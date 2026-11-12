//! Q4_32 × Q8 GEMV / narrow GEMM for decode and prefill.
//!
//! Weights are [`llm_quant::BlockQ4`] (32 signed nibbles + f16 scale). Activations
//! are quantized per 32-element block to int8 + an f32 scale. The inner product
//! of a block is an int32 dot, then scaled by `scale_w * scale_a`.
//!
//! AVX-512 VNNI (and AVX-VNNI) use `vpdpbusd` with an unsigned-weight offset of
//! 8; AVX2 and scalar paths use signed nibbles directly. All three integer
//! paths are required to agree bit-for-bit.

use llm_quant::{quantize_q4, BlockQ4, BLOCK_SIZE};
use rayon::prelude::*;

/// Quantized activation vector: one Q8-style block per 32 elements.
#[derive(Clone, Debug)]
pub struct ActQ8 {
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
}

impl ActQ8 {
    /// Quantize `x` (length multiple of 32) into signed int8 blocks.
    #[must_use]
    pub fn quantize(x: &[f32]) -> Self {
        assert!(
            x.len().is_multiple_of(BLOCK_SIZE),
            "activation length {} is not a multiple of {BLOCK_SIZE}",
            x.len()
        );
        let n_blocks = x.len() / BLOCK_SIZE;
        let mut qs = vec![0i8; x.len()];
        let mut scales = vec![0.0f32; n_blocks];
        let mut sums = vec![0i32; n_blocks];
        for b in 0..n_blocks {
            let src = &x[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let dst = &mut qs[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let amax = src.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            let mut sum = 0i32;
            for (out, &v) in dst.iter_mut().zip(src.iter()) {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                *out = q;
                sum += i32::from(q);
            }
            scales[b] = scale;
            sums[b] = sum;
        }
        Self { qs, scales, sums }
    }
}

/// A packed Q4 matrix with shape `[n, k]`, `k` a multiple of 32, row-major blocks.
#[derive(Clone, Debug)]
pub struct Q4Matrix {
    pub n: usize,
    pub k: usize,
    pub blocks: Vec<BlockQ4>,
}

impl Q4Matrix {
    /// Quantize a row-major fp32 matrix `[n, k]`.
    #[must_use]
    pub fn from_f32(weight: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(weight.len(), n * k);
        assert!(
            k.is_multiple_of(BLOCK_SIZE),
            "K={k} is not a multiple of {BLOCK_SIZE}"
        );
        Self {
            n,
            k,
            blocks: quantize_q4(weight),
        }
    }

    #[inline]
    fn blocks_per_row(&self) -> usize {
        self.k / BLOCK_SIZE
    }
}

/// Integer-dot kernel used by the Q4 paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Kernel {
    Scalar,
    Avx2,
    Vnni,
}

/// Pick the fastest Q4 integer kernel this CPU can run.
#[must_use]
pub fn detect_q4_kernel() -> Q4Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_vnni_256() {
            return Q4Kernel::Vnni;
        }
        if is_x86_feature_detected!("avx2") {
            return Q4Kernel::Avx2;
        }
    }
    Q4Kernel::Scalar
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_vnni_256() -> bool {
    (is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl"))
        || is_x86_feature_detected!("avxvnni")
}

/// `out[n] = W[n, k] @ x[k]` with Q4 weights and on-the-fly Q8 activations.
pub fn gemv_q4(w: &Q4Matrix, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.k);
    assert_eq!(out.len(), w.n);
    let acts = ActQ8::quantize(x);
    gemv_q4_prep(w, &acts, out, detect_q4_kernel());
}

/// Like [`gemv_q4`] but reuses an already-quantized activation (for fused pairs).
pub fn gemv_q4_prep(w: &Q4Matrix, acts: &ActQ8, out: &mut [f32], kernel: Q4Kernel) {
    assert_eq!(acts.qs.len(), w.k);
    assert_eq!(out.len(), w.n);
    if w.n == 0 {
        return;
    }
    let bpr = w.blocks_per_row();
    let blocks = w.blocks.as_slice();
    // Fat row chunks: 8 workers should not pay a task per output channel.
    let chunk = 32.max(w.n / 64).max(1);
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_idx, outs)| {
            let row0 = chunk_idx * chunk;
            for (i, slot) in outs.iter_mut().enumerate() {
                let row = row0 + i;
                let row_w = &blocks[row * bpr..(row + 1) * bpr];
                *slot = dot_row(row_w, acts, kernel);
            }
        });
}

/// Two GEMVs sharing the same activation (SwiGLU gate and up projections).
pub fn gemv_q4_pair(
    w_gate: &Q4Matrix,
    w_up: &Q4Matrix,
    x: &[f32],
    out_gate: &mut [f32],
    out_up: &mut [f32],
) {
    assert_eq!(w_gate.k, w_up.k);
    assert_eq!(x.len(), w_gate.k);
    let acts = ActQ8::quantize(x);
    let kernel = detect_q4_kernel();
    gemv_q4_prep(w_gate, &acts, out_gate, kernel);
    gemv_q4_prep(w_up, &acts, out_up, kernel);
}

/// Prefill: `out[m, n] = X[m, k] @ W[n, k]^T` with Q4 `W`.
pub fn gemm_q4(w: &Q4Matrix, x: &[f32], out: &mut [f32], m: usize) {
    assert_eq!(x.len(), m * w.k);
    assert_eq!(out.len(), m * w.n);
    if m == 0 || w.n == 0 {
        return;
    }
    if m == 1 {
        gemv_q4(w, x, out);
        return;
    }
    let acts: Vec<ActQ8> = x.chunks_exact(w.k).map(ActQ8::quantize).collect();
    let kernel = detect_q4_kernel();
    let bpr = w.blocks_per_row();
    let n = w.n;
    let blocks = w.blocks.as_slice();
    let chunk = 32.max(n / 64).max(1);
    // Disjoint output columns, so a raw pointer is safe to share across workers.
    let out_addr = out.as_mut_ptr() as usize;
    (0..n).into_par_iter().chunks(chunk).for_each(|rows| {
        let out_ptr = out_addr as *mut f32;
        for row in rows {
            let row_w = &blocks[row * bpr..(row + 1) * bpr];
            for (t, act) in acts.iter().enumerate() {
                let val = dot_row(row_w, act, kernel);
                unsafe {
                    *out_ptr.add(t * n + row) = val;
                }
            }
        }
    });
}

fn dot_row(row_w: &[BlockQ4], acts: &ActQ8, kernel: Q4Kernel) -> f32 {
    let mut acc = 0.0f32;
    for (b, block) in row_w.iter().enumerate() {
        let a = &acts.qs[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let idot = match kernel {
            Q4Kernel::Scalar => dot_q4_q8_scalar(block, a),
            Q4Kernel::Avx2 => avx2_dot(block, a),
            Q4Kernel::Vnni => vnni_dot(block, a, acts.sums[b]),
        };
        acc += idot as f32 * block.scale.to_f32() * acts.scales[b];
    }
    acc
}

fn avx2_dot(block: &BlockQ4, a: &[i8]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_q4_q8_avx2(block, a) };
        }
    }
    dot_q4_q8_scalar(block, a)
}

fn vnni_dot(block: &BlockQ4, a: &[i8], a_sum: i32) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_vnni_256() {
            return unsafe { dot_q4_q8_vnni(block, a, a_sum) };
        }
    }
    dot_q4_q8_scalar(block, a)
}

/// Signed-nibble · int8 dot product of one 32-element block.
#[must_use]
pub fn dot_q4_q8_scalar(block: &BlockQ4, a: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), BLOCK_SIZE);
    let mut sum = 0i32;
    for (i, &act) in a.iter().enumerate() {
        sum += i32::from(block.nibble(i)) * i32::from(act);
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn unpack_signed_weights(block: &BlockQ4) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let packed = _mm_loadu_si128(block.qs.as_ptr().cast());
    let mask = _mm_set1_epi8(0x0f);
    let lo = sign_extend_nibbles(_mm_and_si128(packed, mask));
    let hi = sign_extend_nibbles(_mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    let ev = _mm_unpacklo_epi8(lo, hi);
    let od = _mm_unpackhi_epi8(lo, hi);
    _mm256_set_m128i(od, ev)
}

/// Map each byte's low nibble in `0..15` to a signed i8 in `-8..=7`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sign_extend_nibbles(nibbles: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // If bit 3 is set the nibble is negative: OR in 0xF0 (i.e. -16).
    let eight = _mm_set1_epi8(8);
    let signs = _mm_cmpeq_epi8(_mm_and_si128(nibbles, eight), eight);
    _mm_or_si128(nibbles, _mm_and_si128(signs, _mm_set1_epi8(-16)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_q8_avx2(block: &BlockQ4, a: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let w = unpack_signed_weights(block);
    let aq = _mm256_loadu_si256(a.as_ptr().cast());
    let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
    let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
    let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(aq));
    let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(aq, 1));
    let p0 = _mm256_madd_epi16(w_lo, a_lo);
    let p1 = _mm256_madd_epi16(w_hi, a_hi);
    hsum_i32_256(_mm256_add_epi32(p0, p1))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512vnni", enable = "avx512vl", enable = "avx2")]
unsafe fn dot_q4_q8_vnni(block: &BlockQ4, a: &[i8], a_sum: i32) -> i32 {
    use std::arch::x86_64::*;
    // vpdpbusd does u8 × i8. Convert signed nibbles s to u8 = s+8 (0..=15):
    // sum(s * a) = sum((u-8)*a) = dpbusd(u, a) - 8 * sum(a).
    let packed = _mm_loadu_si128(block.qs.as_ptr().cast());
    let mask = _mm_set1_epi8(0x0f);
    let eight = _mm_set1_epi8(8);
    let lo = _mm_and_si128(_mm_add_epi8(_mm_and_si128(packed, mask), eight), mask);
    let hi = _mm_and_si128(
        _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), mask), eight),
        mask,
    );
    let ev = _mm_unpacklo_epi8(lo, hi);
    let od = _mm_unpackhi_epi8(lo, hi);
    let w = _mm256_set_m128i(od, ev);
    let aq = _mm256_loadu_si256(a.as_ptr().cast());
    let prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, aq);
    hsum_i32_256(prod) - 8 * a_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    _mm_cvtsi128_si32(s)
}

#[cfg(test)]
mod tests {
    use super::{detect_q4_kernel, dot_q4_q8_scalar, gemv_q4_prep, ActQ8, Q4Kernel, Q4Matrix};
    use llm_quant::{BlockQ4, BLOCK_SIZE};

    fn pattern_block(seed: i32) -> ([f32; BLOCK_SIZE], [i8; BLOCK_SIZE]) {
        let mut w = [0.0f32; BLOCK_SIZE];
        let mut a = [0i8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            w[i] = ((i as i32 * 17 + seed) % 21 - 10) as f32 * 0.25;
            a[i] = ((i as i32 * 9 + seed) % 25 - 12) as i8;
        }
        (w, a)
    }

    #[test]
    fn vnni_and_avx2_match_scalar_dot() {
        for seed in 0..16 {
            let (w, a) = pattern_block(seed);
            let block = BlockQ4::quantize(&w);
            let scalar = dot_q4_q8_scalar(&block, &a);
            let avx = super::avx2_dot(&block, &a);
            assert_eq!(avx, scalar, "AVX2 mismatch at seed {seed}");
            let sum: i32 = a.iter().map(|&v| i32::from(v)).sum();
            let vnni = super::vnni_dot(&block, &a, sum);
            assert_eq!(vnni, scalar, "VNNI mismatch at seed {seed}");
        }
    }

    #[test]
    fn gemv_kernels_agree() {
        let n = 17;
        let k = 64;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; k];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = ((i as i32 * 13) % 19 - 9) as f32 * 0.15;
        }
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = ((i as i32 * 7) % 15 - 7) as f32 * 0.2;
        }
        let mat = Q4Matrix::from_f32(&w, n, k);
        let acts = ActQ8::quantize(&x);
        let mut scalar = vec![0.0f32; n];
        let mut avx = vec![0.0f32; n];
        let mut vnni = vec![0.0f32; n];
        // Run serially by using the prep kernels on a 1-thread pool isn't required:
        // par_iter is deterministic for this reduction. Compare max abs.
        gemv_q4_prep(&mat, &acts, &mut scalar, Q4Kernel::Scalar);
        gemv_q4_prep(&mat, &acts, &mut avx, Q4Kernel::Avx2);
        gemv_q4_prep(&mat, &acts, &mut vnni, Q4Kernel::Vnni);
        for i in 0..n {
            let s = scalar[i];
            assert!(
                (avx[i] - s).abs() <= 1e-6 * (1.0 + s.abs()),
                "AVX2 row {i}: {} vs {s}",
                avx[i]
            );
            assert!(
                (vnni[i] - s).abs() <= 1e-6 * (1.0 + s.abs()),
                "VNNI row {i}: {} vs {s}",
                vnni[i]
            );
        }
        let _ = detect_q4_kernel();
    }
}
