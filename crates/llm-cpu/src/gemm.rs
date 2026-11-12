//! fp32 GEMM / GEMV with runtime AVX2 dispatch and f64 accumulators.
//!
//! `linear` computes `Y = X @ W^T` for a PyTorch-style weight `[out, in]`.
//! Dots accumulate in f64 so large residual activations (seen in some LFM2
//! conv layers) stay under the 1e-4 per-layer gate. When the token axis is
//! small (decode), we parallelize over output columns instead of rows.

use rayon::prelude::*;

/// `out[m, n] = sum_k a[m, k] * b[n, k]` (`b` is stored row-major `[N, K]`).
pub fn gemm_nt(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m.checked_mul(k).expect("m*k overflow"));
    assert_eq!(b.len(), n.checked_mul(k).expect("n*k overflow"));
    assert_eq!(out.len(), m.checked_mul(n).expect("m*n overflow"));
    if m == 0 || n == 0 {
        return;
    }
    let use_avx = is_avx2();
    if m <= 4 {
        // Decode / narrow prefill: one or a few tokens, many output channels
        // (including the 65536-wide tied head). Parallelize over `n`.
        let a_addr = a.as_ptr() as usize;
        let b_addr = b.as_ptr() as usize;
        let out_addr = out.as_mut_ptr() as usize;
        (0..n).into_par_iter().for_each(|col| {
            let b_row =
                unsafe { std::slice::from_raw_parts((b_addr as *const f32).add(col * k), k) };
            for row in 0..m {
                let a_row =
                    unsafe { std::slice::from_raw_parts((a_addr as *const f32).add(row * k), k) };
                let slot = unsafe { &mut *(out_addr as *mut f32).add(row * n + col) };
                *slot = dot(a_row, b_row, use_avx);
            }
        });
        return;
    }
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    out.par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, out_row)| {
            let a_row =
                unsafe { std::slice::from_raw_parts((a_addr as *const f32).add(row * k), k) };
            let b_mat = unsafe { std::slice::from_raw_parts(b_addr as *const f32, n * k) };
            for col in 0..n {
                let b_row = &b_mat[col * k..(col + 1) * k];
                out_row[col] = dot(a_row, b_row, use_avx);
            }
        });
}

/// `out[out_dim] = sum_k x[k] * w[out, k]`.
pub fn gemv_nt(x: &[f32], w: &[f32], out: &mut [f32], out_dim: usize) {
    let k = x.len();
    assert_eq!(w.len(), out_dim * k);
    assert_eq!(out.len(), out_dim);
    gemm_nt(x, w, out, 1, out_dim, k);
}

/// Affine map `Y[T, out] = X[T, in] @ W[out, in]^T` (no bias).
pub fn linear(x: &[f32], w: &[f32], tokens: usize, d_in: usize, d_out: usize, out: &mut [f32]) {
    assert_eq!(x.len(), tokens * d_in);
    assert_eq!(w.len(), d_out * d_in);
    assert_eq!(out.len(), tokens * d_out);
    gemm_nt(x, w, out, tokens, d_out, d_in);
}

/// `dst[t, out] += X[t, in] @ W[out, in]^T`, adding in f64 then writing f32.
/// Used for the FFN down-projection so a large residual does not pick up an
/// extra rounding step in the residual add.
pub fn linear_add(x: &[f32], w: &[f32], tokens: usize, d_in: usize, d_out: usize, dst: &mut [f32]) {
    assert_eq!(x.len(), tokens * d_in);
    assert_eq!(w.len(), d_out * d_in);
    assert_eq!(dst.len(), tokens * d_out);
    let mut tmp = vec![0.0f32; tokens * d_out];
    gemm_nt(x, w, &mut tmp, tokens, d_out, d_in);
    for (d, &s) in dst.iter_mut().zip(tmp.iter()) {
        *d = (f64::from(*d) + f64::from(s)) as f32;
    }
}

fn is_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn dot(a: &[f32], b: &[f32], use_avx: bool) -> f32 {
    if use_avx {
        unsafe { dot_avx2_f64(a, b) }
    } else {
        dot_scalar_f64(a, b)
    }
}

#[cfg(test)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    dot_scalar_f64(a, b)
}

fn dot_scalar_f64(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += f64::from(*x) * f64::from(*y);
    }
    sum as f32
}

/// AVX2 dot product with f64 accumulators (cvt 8-wide f32 loads to two f64 vectors).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2_f64(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / 8;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    for i in 0..chunks {
        let off = i * 8;
        // Convert first, then multiply in f64. An f32 product rounded before the
        // add is enough to miss the 1e-4 residual gate on large conv activations.
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));
        let a_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(va));
        let a_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1));
        let b_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vb));
        let b_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1));
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(a_lo, b_lo));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(a_hi, b_hi));
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for i in (chunks * 8)..n {
        sum += f64::from(a[i]) * f64::from(b[i]);
    }
    sum as f32
}

#[cfg(test)]
mod tests {
    use super::{dot_scalar, gemm_nt};

    #[test]
    fn gemm_nt_matches_scalar() {
        let m = 5;
        let n = 7;
        let k = 17;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.13 - 1.2).sin())
            .collect();
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.07 + 0.4).cos())
            .collect();
        let mut out = vec![0.0; m * n];
        gemm_nt(&a, &b, &mut out, m, n, k);
        for row in 0..m {
            for col in 0..n {
                let got = out[row * n + col];
                let exp = dot_scalar(&a[row * k..(row + 1) * k], &b[col * k..(col + 1) * k]);
                let tol = 1e-5_f32 * (1.0 + exp.abs());
                assert!((got - exp).abs() <= tol, "[{row},{col}] {got} vs {exp}");
            }
        }
    }

    #[test]
    fn gemm_nt_decode_shape_agrees_with_wide() {
        // m=1 path (column-parallel) must match a 5-row computation's first row.
        let n = 11;
        let k = 33;
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.02).sin()).collect();
        let mut out1 = vec![0.0; n];
        let mut out5 = vec![0.0; 5 * n];
        let mut a5 = vec![0.0; 5 * k];
        a5[..k].copy_from_slice(&a);
        gemm_nt(&a, &b, &mut out1, 1, n, k);
        gemm_nt(&a5, &b, &mut out5, 5, n, k);
        for i in 0..n {
            assert!((out1[i] - out5[i]).abs() < 1e-6);
        }
    }
}
