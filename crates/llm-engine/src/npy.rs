//! Minimal `.npy` / `.npz` reader for golden-dump comparison.
//!
//! Supports little-endian C-order `f32` and integer arrays produced by
//! `numpy.savez` / `numpy.save`.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

/// An array loaded from a `.npy` member.
#[derive(Clone, Debug)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

#[derive(Clone, Debug)]
pub enum NpyData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl NpyArray {
    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            NpyData::F32(v) => Some(v),
            NpyData::I64(_) => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.data {
            NpyData::I64(v) => Some(v),
            NpyData::F32(_) => None,
        }
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

/// Read one `.npy` blob.
pub fn read_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (header_off, header_len) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (10, len)
        }
        2 | 3 => {
            let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            (12, len)
        }
        v => return Err(format!("unsupported npy version {v}")),
    };
    let header_end = header_off + header_len;
    if bytes.len() < header_end {
        return Err("truncated npy header".into());
    }
    let header = std::str::from_utf8(&bytes[header_off..header_end])
        .map_err(|_| "npy header is not utf-8")?;
    if header.contains("fortran_order': True") || header.contains("fortran_order\": True") {
        return Err("fortran-order arrays are not supported".into());
    }
    let descr = parse_descr(header)?;
    let shape = parse_shape(header)?;
    let numel: usize = shape.iter().copied().product();
    let body = &bytes[header_end..];
    let data = match descr {
        "<f4" | "|f4" => {
            if body.len() < numel * 4 {
                return Err("truncated f32 payload".into());
            }
            let mut v = vec![0.0f32; numel];
            for (dst, chunk) in v.iter_mut().zip(body.as_chunks::<4>().0.iter()) {
                *dst = f32::from_le_bytes(*chunk);
            }
            NpyData::F32(v)
        }
        "<i8" | "|i8" | "<i4" | "|i4" | "<i2" | "|u1" | "|i1" => {
            let width = match &descr[1..] {
                "i8" => 8,
                "i4" => 4,
                "i2" => 2,
                "u1" | "i1" => 1,
                _ => 8,
            };
            if body.len() < numel * width {
                return Err("truncated integer payload".into());
            }
            let mut v = vec![0i64; numel];
            for (i, dst) in v.iter_mut().enumerate() {
                let off = i * width;
                *dst = match width {
                    8 => i64::from_le_bytes(body[off..off + 8].try_into().unwrap()),
                    4 => i32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as i64,
                    2 => i16::from_le_bytes(body[off..off + 2].try_into().unwrap()) as i64,
                    1 => body[off] as i8 as i64,
                    _ => unreachable!(),
                };
            }
            NpyData::I64(v)
        }
        other => return Err(format!("unsupported npy dtype {other}")),
    };
    Ok(NpyArray { shape, data })
}

fn parse_descr(header: &str) -> Result<&'static str, String> {
    const KNOWN: &[&str] = &[
        "<f4", "|f4", "<i8", "|i8", "<i4", "|i4", "<i2", "|u1", "|i1",
    ];
    for d in KNOWN {
        if header.contains(&format!("'descr': '{d}'"))
            || header.contains(&format!("\"descr\": \"{d}\""))
            || header.contains(&format!("'descr': '{d}'"))
        {
            return Ok(*d);
        }
    }
    // Some numpy versions use descr': "<f4"
    if let Some(i) = header.find("descr") {
        let rest = &header[i..];
        for d in KNOWN {
            if rest.contains(d) {
                return Ok(*d);
            }
        }
    }
    Err(format!("could not parse npy descr from {header}"))
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let start = header
        .find("shape")
        .and_then(|i| header[i..].find('('))
        .map(|j| header.find("shape").unwrap() + j)
        .ok_or_else(|| format!("no shape in {header}"))?;
    let rest = &header[start + 1..];
    let end = rest.find(')').ok_or("unterminated shape")?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad shape dim {s}"))
        })
        .collect()
}

/// Read every array in a `.npz` (zip of `.npy` files).
pub fn read_npz(
    path: impl AsRef<Path>,
) -> Result<std::collections::HashMap<String, NpyArray>, String> {
    let file = File::open(path.as_ref()).map_err(|e| e.to_string())?;
    let mut zip = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut out = std::collections::HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().map_err(|e| e.to_string())?.to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let key = name.trim_end_matches(".npy").to_string();
        out.insert(key, read_npy(&buf)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{parse_shape, read_npy};

    #[test]
    fn parses_shape_variants() {
        assert_eq!(
            parse_shape("{'descr': '<f4', 'fortran_order': False, 'shape': (6, 1536), }").unwrap(),
            vec![6, 1536]
        );
        assert_eq!(
            parse_shape("{'descr': '<i8', 'fortran_order': False, 'shape': (6,), }").unwrap(),
            vec![6]
        );
    }

    #[test]
    fn reads_minimal_f32_npy() {
        // Hand-built C-order vector of [1.0, 2.0]
        let mut buf = b"\x93NUMPY".to_vec();
        buf.push(1);
        buf.push(0);
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        // pad to 16-byte alignment: 10 + header_len + data_off
        let mut hdr = header.to_string();
        // header is preceded by 10 bytes; total preamble+header should be multiple of 16
        while !(10 + hdr.len()).is_multiple_of(16) {
            hdr.push(' ');
        }
        let len = hdr.len() as u16;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(hdr.as_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&2.0f32.to_le_bytes());
        let arr = read_npy(&buf).unwrap();
        assert_eq!(arr.shape, vec![2]);
        let d = arr.as_f32().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6 && (d[1] - 2.0).abs() < 1e-6);
    }
}
