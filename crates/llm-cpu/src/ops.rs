//! Elementwise and reduction ops used by the LFM2 fp32 path.

/// RMSNorm as in `Lfm2RMSNorm` / T5LayerNorm:
/// `y = weight * x * rsqrt(mean(x^2) + eps)`.
/// The mean of squares is accumulated in f64 so a sequential sum matches
/// the more accurate reduction PyTorch uses, then the scale is applied in f32
/// the same way as `Lfm2RMSNorm` (`x * rsqrt(var + eps)`, then `weight * x`).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = weight.len();
    assert!(x.len().is_multiple_of(dim));
    assert_eq!(out.len(), x.len());
    let inv_dim = 1.0 / dim as f64;
    for (row, dst) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mut sumsq = 0.0f64;
        for &v in row {
            sumsq += f64::from(v) * f64::from(v);
        }
        let scale = ((sumsq * inv_dim + f64::from(eps)).sqrt().recip()) as f32;
        for ((d, &v), &w) in dst.iter_mut().zip(row.iter()).zip(weight.iter()) {
            *d = w * (v * scale);
        }
    }
}

/// `silu(x) = x * sigmoid(x)` (`F.silu`).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place `silu` over a buffer.
pub fn silu_inplace(xs: &mut [f32]) {
    for x in xs {
        *x = silu(*x);
    }
}

/// `out = a * b` (elementwise), same length.
pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(out.len(), a.len());
    for ((d, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x * y;
    }
}

/// `dst += src`.
pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// `dst = dst + src` aliased as a residual write into a fresh buffer.
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(out.len(), a.len());
    for ((d, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x + y;
    }
}

/// Stable softmax over the last axis of length `n`. `logits` is `[rows, n]`.
pub fn softmax_last_axis(logits: &mut [f32], n: usize) {
    assert!(logits.len().is_multiple_of(n));
    for row in logits.chunks_exact_mut(n) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Argmax of a 1-d logit vector; ties break to the lowest index (PyTorch `argmax`).
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty());
    let mut best_i = 0usize;
    let mut best = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::{argmax, rms_norm, silu, softmax_last_axis};

    #[test]
    fn silu_at_zero_is_zero() {
        assert!((silu(0.0)).abs() < 1e-7);
        assert!(silu(2.0) > 1.0);
        assert!(silu(-2.0) < 0.0);
    }

    #[test]
    fn rms_norm_unit_weight() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut y = [0.0; 2];
        rms_norm(&x, &w, 0.0, &mut y);
        // rms = 5/sqrt(2), y = x / rms
        let rms = ((9.0 + 16.0) / 2.0_f32).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-6);
        assert!((y[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_argmax() {
        let mut z = [1.0f32, 3.0, 2.0];
        softmax_last_axis(&mut z, 3);
        let sum: f32 = z.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }
}
