//! Safetensors loader: mmap the file and convert BF16/F16/F32 weights to fp32.

use llm_core::ModelConfig;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Error from locating or reading a checkpoint.
#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Safetensors(safetensors::SafeTensorError),
    Message(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Safetensors(e) => write!(f, "{e}"),
            Self::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(e)
    }
}

/// A named fp32 tensor with a shape.
#[derive(Clone, Debug)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorF32 {
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Locate a Hugging Face snapshot directory that contains `config.json` and
/// `model.safetensors`. Checks `LLMHOST_MODEL_DIR`, then the usual Hub caches.
/// Does not download anything.
pub fn find_hf_snapshot(model: &str, revision_prefix: &str) -> Result<PathBuf, LoadError> {
    if let Ok(dir) = std::env::var("LLMHOST_MODEL_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.safetensors").is_file() && p.join("config.json").is_file() {
            return Ok(p);
        }
        return Err(LoadError::Message(format!(
            "LLMHOST_MODEL_DIR={p:?} is missing config.json or model.safetensors"
        )));
    }
    let repo = format!("models--{}", model.replace('/', "--"));
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(c) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(c) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(c).join("hub"));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    for root in roots {
        let snaps = root.join(&repo).join("snapshots");
        let Ok(entries) = std::fs::read_dir(&snaps) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        // Prefer a snapshot whose name starts with the requested prefix.
        dirs.sort_by_key(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with(revision_prefix) {
                0
            } else {
                1
            }
        });
        for dir in dirs {
            let name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !revision_prefix.is_empty() && !name.starts_with(revision_prefix) {
                continue;
            }
            if dir.join("model.safetensors").is_file() && dir.join("config.json").is_file() {
                return Ok(dir);
            }
        }
    }
    Err(LoadError::Message(format!(
        "no local Hugging Face snapshot for {model} (revision {revision_prefix}); \
         set LLMHOST_MODEL_DIR or place the files in the Hub cache"
    )))
}

/// Memory-map a `.safetensors` file and decode every tensor as fp32.
pub fn load_safetensors_f32(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, TensorF32>, LoadError> {
    let file = File::open(path.as_ref())?;
    // SAFETY: we only read the mapping, and the file is not written while we hold it.
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let mut out = HashMap::new();
    for (name, view) in tensors.tensors() {
        out.insert(
            name,
            TensorF32 {
                shape: view.shape().to_vec(),
                data: tensor_to_f32(view.dtype(), view.data())?,
            },
        );
    }
    Ok(out)
}

fn tensor_to_f32(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>, LoadError> {
    match dtype {
        Dtype::F32 => {
            if !raw.len().is_multiple_of(4) {
                return Err(LoadError::Message(
                    "f32 tensor is not 4-byte aligned".into(),
                ));
            }
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        }
        Dtype::BF16 => {
            if !raw.len().is_multiple_of(2) {
                return Err(LoadError::Message(
                    "bf16 tensor is not 2-byte aligned".into(),
                ));
            }
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    let bits = u16::from_le_bytes(*c);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect())
        }
        Dtype::F16 => {
            if !raw.len().is_multiple_of(2) {
                return Err(LoadError::Message(
                    "f16 tensor is not 2-byte aligned".into(),
                ));
            }
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    let bits = u16::from_le_bytes(*c);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        other => Err(LoadError::Message(format!(
            "unsupported safetensors dtype {other:?}"
        ))),
    }
}

/// Read `config.json` next to the weights and parse it as a [`ModelConfig`].
pub fn load_config(dir: impl AsRef<Path>) -> Result<ModelConfig, LoadError> {
    let path = dir.as_ref().join("config.json");
    ModelConfig::from_path(&path).map_err(|e| LoadError::Message(e.to_string()))
}

/// Look up `name` or `model.{name}` in a tensor map.
pub fn get_tensor<'a>(
    tensors: &'a HashMap<String, TensorF32>,
    name: &str,
) -> Result<&'a TensorF32, LoadError> {
    if let Some(t) = tensors.get(name) {
        return Ok(t);
    }
    let alt = if let Some(stripped) = name.strip_prefix("model.") {
        stripped.to_string()
    } else {
        format!("model.{name}")
    };
    tensors
        .get(&alt)
        .ok_or_else(|| LoadError::Message(format!("missing tensor {name}")))
}

/// Squeeze a conv weight `[H, 1, K]` into `[H, K]`.
pub fn squeeze_conv_weight(t: &TensorF32) -> Result<Vec<f32>, LoadError> {
    match t.shape.as_slice() {
        [_h, 1, _k] => Ok(t.data.clone()),
        [_h, _k] => Ok(t.data.clone()),
        other => Err(LoadError::Message(format!(
            "unexpected conv weight shape {other:?}"
        ))),
    }
}
