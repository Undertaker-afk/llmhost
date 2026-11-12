//! Block quantization formats used by the CPU decode kernels.
//!
//! Both [`BlockQ8_0`] and [`BlockQ4`] store 32 elements and a single f16 scale.
//! These are llmhost's own layouts; a GGUF reader may be added later.

use half::f16;

mod formats;
pub use formats::{KvDtype, QuantError, WeightFormat};

/// Number of elements in a Q8_0 or Q4 block.
pub const BLOCK_SIZE: usize = 32;

/// Q8_0 block: 32 int8 values and one f16 scale.
///
/// Dequantized value `i` is `qs[i] * scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    pub scale: f16,
    pub qs: [i8; BLOCK_SIZE],
}

impl BlockQ8_0 {
    /// Quantize 32 f32 values into a Q8_0 block.
    #[must_use]
    pub fn quantize(src: &[f32; BLOCK_SIZE]) -> Self {
        let amax = src.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = f16::from_f32(amax / 127.0);
        let scale_f = scale.to_f32();
        let inv = if scale_f == 0.0 { 0.0 } else { 1.0 / scale_f };
        let mut qs = [0i8; BLOCK_SIZE];
        for (dst, src) in qs.iter_mut().zip(src.iter()) {
            let q = (src * inv).round().clamp(-127.0, 127.0);
            *dst = q as i8;
        }
        Self { scale, qs }
    }

    /// Dequantize this block into 32 f32 values.
    pub fn dequantize(&self, dst: &mut [f32; BLOCK_SIZE]) {
        let scale = self.scale.to_f32();
        for (dst, q) in dst.iter_mut().zip(self.qs.iter()) {
            *dst = f32::from(*q) * scale;
        }
    }
}

/// Q4 block: 32 signed 4-bit values packed into 16 bytes, plus one f16 scale.
///
/// Each nibble is a two's-complement value in `-8..=7`. Even elements live in
/// the low nibble, odd elements in the high nibble. Dequantized value `i` is
/// `q[i] * scale`, with `scale = max(abs(x)) / 8` (so `-8 * scale` reaches
/// the block minimum).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ4 {
    pub scale: f16,
    pub qs: [u8; BLOCK_SIZE / 2],
}

impl BlockQ4 {
    /// Quantize 32 f32 values into a Q4 block.
    #[must_use]
    pub fn quantize(src: &[f32; BLOCK_SIZE]) -> Self {
        let amax = src.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = f16::from_f32(amax / 8.0);
        let scale_f = scale.to_f32();
        let inv = if scale_f == 0.0 { 0.0 } else { 1.0 / scale_f };
        let mut qs = [0u8; BLOCK_SIZE / 2];
        for (i, src) in src.iter().enumerate() {
            let q = (src * inv).round().clamp(-8.0, 7.0) as i8;
            let nibble = (q as u8) & 0x0f;
            if i.is_multiple_of(2) {
                qs[i / 2] = nibble;
            } else {
                qs[i / 2] |= nibble << 4;
            }
        }
        Self { scale, qs }
    }

    /// Unpack nibble `i` as a signed integer in `-8..=7`.
    #[must_use]
    pub fn nibble(&self, i: usize) -> i8 {
        debug_assert!(i < BLOCK_SIZE);
        let packed = self.qs[i / 2];
        let nibble = if i.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        ((nibble as i8) << 4) >> 4
    }

    /// Dequantize this block into 32 f32 values.
    pub fn dequantize(&self, dst: &mut [f32; BLOCK_SIZE]) {
        let scale = self.scale.to_f32();
        for (i, dst) in dst.iter_mut().enumerate() {
            *dst = f32::from(self.nibble(i)) * scale;
        }
    }
}

/// Quantize a slice whose length is a multiple of [`BLOCK_SIZE`].
pub fn quantize_q8_0(input: &[f32]) -> Vec<BlockQ8_0> {
    let (chunks, rem) = input.as_chunks::<BLOCK_SIZE>();
    assert!(
        rem.is_empty(),
        "Q8_0 input length must be a multiple of {BLOCK_SIZE}"
    );
    chunks.iter().map(BlockQ8_0::quantize).collect()
}

/// Dequantize a sequence of Q8_0 blocks.
pub fn dequantize_q8_0(blocks: &[BlockQ8_0], output: &mut [f32]) {
    assert_eq!(output.len(), blocks.len() * BLOCK_SIZE);
    let (chunks, rem) = output.as_chunks_mut::<BLOCK_SIZE>();
    debug_assert!(rem.is_empty());
    for (block, chunk) in blocks.iter().zip(chunks.iter_mut()) {
        block.dequantize(chunk);
    }
}

/// Quantize a slice whose length is a multiple of [`BLOCK_SIZE`].
pub fn quantize_q4(input: &[f32]) -> Vec<BlockQ4> {
    let (chunks, rem) = input.as_chunks::<BLOCK_SIZE>();
    assert!(
        rem.is_empty(),
        "Q4 input length must be a multiple of {BLOCK_SIZE}"
    );
    chunks.iter().map(BlockQ4::quantize).collect()
}

/// Dequantize a sequence of Q4 blocks.
pub fn dequantize_q4(blocks: &[BlockQ4], output: &mut [f32]) {
    assert_eq!(output.len(), blocks.len() * BLOCK_SIZE);
    let (chunks, rem) = output.as_chunks_mut::<BLOCK_SIZE>();
    debug_assert!(rem.is_empty());
    for (block, chunk) in blocks.iter().zip(chunks.iter_mut()) {
        block.dequantize(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dequantize_q4, dequantize_q8_0, quantize_q4, quantize_q8_0, BlockQ4, BlockQ8_0, BLOCK_SIZE,
    };

    fn pattern() -> [f32; BLOCK_SIZE] {
        let mut src = [0.0f32; BLOCK_SIZE];
        for (i, slot) in src.iter_mut().enumerate() {
            *slot = (i as f32) * 0.37 - 5.1;
        }
        // Make sure both signs and a near-zero are present.
        src[0] = -6.0;
        src[1] = 6.0;
        src[2] = 0.0;
        src
    }

    fn max_abs_err(orig: &[f32], back: &[f32]) -> f32 {
        orig.iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn q8_0_roundtrip() {
        let src = pattern();
        let block = BlockQ8_0::quantize(&src);
        let mut back = [0.0f32; BLOCK_SIZE];
        block.dequantize(&mut back);
        let amax = src.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // One Q8 step is amax/127; allow a little slack for f16 scale rounding.
        let bound = amax / 127.0 + 1e-3;
        let err = max_abs_err(&src, &back);
        assert!(err <= bound, "Q8_0 max abs err {err} > bound {bound}");
    }

    #[test]
    fn q4_roundtrip() {
        let src = pattern();
        let block = BlockQ4::quantize(&src);
        let mut back = [0.0f32; BLOCK_SIZE];
        block.dequantize(&mut back);
        let amax = src.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // One Q4 step is amax/8; allow slack for rounding to the grid.
        let bound = amax / 8.0 + 1e-3;
        let err = max_abs_err(&src, &back);
        assert!(err <= bound, "Q4 max abs err {err} > bound {bound}");
        for i in 0..BLOCK_SIZE {
            let q = block.nibble(i);
            assert!((-8..=7).contains(&q), "nibble {i} out of range: {q}");
        }
    }

    #[test]
    fn q8_0_and_q4_zero_block() {
        let src = [0.0f32; BLOCK_SIZE];
        let mut back = [0.0f32; BLOCK_SIZE];
        BlockQ8_0::quantize(&src).dequantize(&mut back);
        assert!(back.iter().all(|v| *v == 0.0));
        BlockQ4::quantize(&src).dequantize(&mut back);
        assert!(back.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn slice_helpers_roundtrip() {
        let mut src = vec![0.0f32; BLOCK_SIZE * 3];
        for (i, slot) in src.iter_mut().enumerate() {
            *slot = (i as f32).sin() * 3.0;
        }
        let mut back = vec![0.0f32; src.len()];
        dequantize_q8_0(&quantize_q8_0(&src), &mut back);
        let err = max_abs_err(&src, &back);
        assert!(err < 0.05, "slice Q8_0 err {err}");

        dequantize_q4(&quantize_q4(&src), &mut back);
        let err = max_abs_err(&src, &back);
        assert!(err < 0.5, "slice Q4 err {err}");
    }
}
