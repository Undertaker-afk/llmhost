//! LFM2 / LLaMA-style RoPE (`cat((freqs, freqs))` + `rotate_half`).

/// Inverse frequencies: `1 / theta^(i/dim)` for `i` in `0, 2, ..., dim-2`.
/// Computed in fp32 to match `Lfm2RotaryEmbedding.compute_default_rope_parameters`.
#[must_use]
pub fn inv_freq(dim: usize, theta: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim / 2);
    let dim_f = dim as f32;
    for i in (0..dim).step_by(2) {
        let expo = i as f32 / dim_f;
        out.push(1.0 / theta.powf(expo));
    }
    out
}

/// Cos/sin tables for positions `start..start+tokens`, each of length `dim`.
/// Layout: `cos[t * dim + d]`.
pub fn cos_sin(
    inv_freq: &[f32],
    start: usize,
    tokens: usize,
    dim: usize,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    assert_eq!(inv_freq.len(), dim / 2);
    assert_eq!(cos.len(), tokens * dim);
    assert_eq!(sin.len(), tokens * dim);
    let half = dim / 2;
    for t in 0..tokens {
        let pos = (start + t) as f32;
        for i in 0..half {
            let freq = pos * inv_freq[i];
            let c = freq.cos();
            let s = freq.sin();
            // emb = cat(freqs, freqs), so both halves share the same angle.
            cos[t * dim + i] = c;
            cos[t * dim + half + i] = c;
            sin[t * dim + i] = s;
            sin[t * dim + half + i] = s;
        }
    }
}

/// Rotate half: `cat(-x[half:], x[:half])`.
#[inline]
fn rotate_half_into(x: &[f32], out: &mut [f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
}

/// Apply RoPE to a packed `[heads, tokens, dim]` tensor, in place.
pub fn apply_rope_in_place(
    states: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    heads: usize,
    tokens: usize,
    dim: usize,
) {
    assert_eq!(states.len(), heads * tokens * dim);
    assert_eq!(cos.len(), tokens * dim);
    assert_eq!(sin.len(), tokens * dim);
    let mut rotated = vec![0.0f32; dim];
    for h in 0..heads {
        for t in 0..tokens {
            let off = (h * tokens + t) * dim;
            let q = &states[off..off + dim];
            rotate_half_into(q, &mut rotated);
            let c = &cos[t * dim..(t + 1) * dim];
            let s = &sin[t * dim..(t + 1) * dim];
            for d in 0..dim {
                rotated[d] = q[d] * c[d] + rotated[d] * s[d];
            }
            states[off..off + dim].copy_from_slice(&rotated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_rope_in_place, cos_sin, inv_freq};

    #[test]
    fn inv_freq_length_and_first() {
        let freq = inv_freq(64, 1_000_000.0);
        assert_eq!(freq.len(), 32);
        assert!((freq[0] - 1.0).abs() < 1e-6);
        assert!(freq[31] < freq[0]);
    }

    #[test]
    fn rope_is_a_rotation_of_pairs() {
        let dim = 4;
        let freq = inv_freq(dim, 10_000.0);
        let mut cos = vec![0.0; dim];
        let mut sin = vec![0.0; dim];
        cos_sin(&freq, 0, 1, dim, &mut cos, &mut sin);
        // position 0 -> angle 0 -> identity
        let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_rope_in_place(&mut q, &cos, &sin, 1, 1, dim);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 2.0).abs() < 1e-6);
        assert!((q[2] - 3.0).abs() < 1e-6);
        assert!((q[3] - 4.0).abs() < 1e-6);
    }
}
