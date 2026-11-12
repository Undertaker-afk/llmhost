//! f32 matrix-vector product: `out[r] = sum_c matrix[r, c] * vector[c]`.
//!
//! `matrix` is a row-major `rows × cols` slab, `vector` has `cols` elements,
//! and `out` has `rows` elements.

use crate::{CpuFeatures, KernelSet};

/// Scalar reference implementation. Used as the correctness baseline.
pub fn matvec_f32_scalar(matrix: &[f32], vector: &[f32], out: &mut [f32]) {
    let cols = vector.len();
    let rows = out.len();
    assert_eq!(
        matrix.len(),
        rows.checked_mul(cols).expect("rows * cols overflowed"),
        "matrix length must equal rows * cols"
    );
    for (row_idx, dst) in out.iter_mut().enumerate() {
        let row = &matrix[row_idx * cols..(row_idx + 1) * cols];
        let mut acc = 0.0f32;
        for (a, x) in row.iter().zip(vector.iter()) {
            acc += *a * *x;
        }
        *dst = acc;
    }
}

/// AVX2 f32 matvec. Panics if AVX2 is missing on the current CPU.
pub fn matvec_f32_avx2(matrix: &[f32], vector: &[f32], out: &mut [f32]) {
    assert!(
        CpuFeatures::detect().avx2,
        "matvec_f32_avx2 requires AVX2 at runtime"
    );
    // SAFETY: we just checked that the host supports AVX2.
    unsafe { matvec_f32_avx2_impl(matrix, vector, out) }
}

/// Dispatch to the best implemented f32 kernel on this CPU.
pub fn matvec_f32(matrix: &[f32], vector: &[f32], out: &mut [f32]) {
    let features = CpuFeatures::detect();
    match features.decode_kernels() {
        KernelSet::Avx2 | KernelSet::Avx512 | KernelSet::Avx512Vnni | KernelSet::AmxInt8
            if features.avx2 =>
        {
            // VNNI / AMX f32 paths are not implemented yet; AVX2 is the
            // fastest f32 kernel in this skeleton.
            matvec_f32_avx2(matrix, vector, out);
        }
        _ => matvec_f32_scalar(matrix, vector, out),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn matvec_f32_avx2_impl(matrix: &[f32], vector: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let cols = vector.len();
    let rows = out.len();
    assert_eq!(
        matrix.len(),
        rows.checked_mul(cols).expect("rows * cols overflowed"),
        "matrix length must equal rows * cols"
    );

    let chunks = cols / 8;
    let rem_base = chunks * 8;

    for (row_idx, dst) in out.iter_mut().enumerate() {
        let row_ptr = matrix.as_ptr().add(row_idx * cols);
        let mut acc = _mm256_setzero_ps();
        for chunk in 0..chunks {
            let offset = chunk * 8;
            let a = _mm256_loadu_ps(row_ptr.add(offset));
            let x = _mm256_loadu_ps(vector.as_ptr().add(offset));
            // mul + add, not FMA: this function is gated on AVX2 only.
            acc = _mm256_add_ps(acc, _mm256_mul_ps(a, x));
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = 0.0f32;
        for lane in tmp {
            sum += lane;
        }
        for (col, x) in vector.iter().enumerate().skip(rem_base) {
            sum += *row_ptr.add(col) * *x;
        }
        *dst = sum;
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn matvec_f32_avx2_impl(matrix: &[f32], vector: &[f32], out: &mut [f32]) {
    // Unreachable: the public wrapper requires CpuFeatures.avx2, which is
    // always false on non-x86. Keep a scalar body so the crate still typechecks.
    matvec_f32_scalar(matrix, vector, out);
}

#[cfg(test)]
mod tests {
    use super::{matvec_f32_avx2, matvec_f32_scalar};
    use crate::CpuFeatures;

    fn fill_pattern(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.017 + phase).sin())
            .collect()
    }

    fn assert_close(scalar: &[f32], other: &[f32]) {
        assert_eq!(scalar.len(), other.len());
        for (idx, (s, o)) in scalar.iter().zip(other.iter()).enumerate() {
            let diff = (s - o).abs();
            let tol = 1e-4_f32 * (1.0 + s.abs());
            assert!(
                diff <= tol,
                "row {idx}: scalar={s} other={o} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn avx2_matches_scalar() {
        // Odd sizes exercise both the 8-wide body and the scalar tail.
        let rows = 17;
        let cols = 65;
        let matrix = fill_pattern(rows * cols, 0.4);
        let vector = fill_pattern(cols, 1.7);
        let mut scalar_out = vec![0.0; rows];
        matvec_f32_scalar(&matrix, &vector, &mut scalar_out);

        if !CpuFeatures::detect().avx2 {
            // Nothing to compare; the scalar path already ran.
            assert_eq!(scalar_out.len(), rows);
            return;
        }

        let mut avx_out = vec![0.0; rows];
        matvec_f32_avx2(&matrix, &vector, &mut avx_out);
        assert_close(&scalar_out, &avx_out);
    }

    #[test]
    fn scalar_handles_empty_and_aligned() {
        let mut out = vec![];
        matvec_f32_scalar(&[], &[], &mut out);

        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vector = vec![1.0; 8];
        let mut out = vec![0.0; 1];
        matvec_f32_scalar(&matrix, &vector, &mut out);
        assert!((out[0] - 36.0).abs() < 1e-6);
    }
}
