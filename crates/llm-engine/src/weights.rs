//! Linear layers in fp32 or Q4_32, plus an optional Q8_0 tied head.

use crate::load::LoadError;
use llm_cpu::{
    for_ranges, gemm_q4, gemm_q4_add, gemv_q4, gemv_q4_add, gemv_q4_pair, linear, linear_add,
    prefault_bytes, prefault_f32, ActQ8, Q4Matrix,
};
use llm_quant::{quantize_q8_0, BlockQ8_0, WeightFormat, BLOCK_SIZE};

/// A weight matrix used as `Y = X @ W^T` (`W` is `[out, in]`).
pub enum Linear {
    F32 {
        rows: usize,
        cols: usize,
        weight: Vec<f32>,
    },
    Q4(Q4Matrix),
}

impl Linear {
    /// Build a linear layer from a row-major fp32 `[rows, cols]` buffer.
    pub fn from_f32(
        weight: Vec<f32>,
        rows: usize,
        cols: usize,
        format: WeightFormat,
    ) -> Result<Self, LoadError> {
        if weight.len() != rows * cols {
            return Err(LoadError::Message(format!(
                "linear weight length {} != {rows}*{cols}",
                weight.len()
            )));
        }
        match format {
            WeightFormat::F32 => Ok(Self::F32 { rows, cols, weight }),
            WeightFormat::Q4_32 => {
                if !cols.is_multiple_of(BLOCK_SIZE) {
                    return Err(LoadError::Message(format!(
                        "Q4 linear K={cols} is not a multiple of {BLOCK_SIZE}"
                    )));
                }
                Ok(Self::Q4(Q4Matrix::from_f32(&weight, rows, cols)))
            }
            other => Err(LoadError::Message(format!(
                "unsupported linear weight format {other:?}"
            ))),
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } => *rows,
            Self::Q4(m) => m.n,
        }
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } => *cols,
            Self::Q4(m) => m.k,
        }
    }

    /// Fault in packed or fp32 weight pages.
    pub fn prefault(&self) {
        match self {
            Self::F32 { weight, .. } => prefault_f32(weight),
            Self::Q4(m) => m.prefault(),
        }
    }

    /// `out[tokens, rows] = x[tokens, cols] @ W^T`.
    pub fn apply(&self, x: &[f32], tokens: usize, out: &mut [f32]) {
        assert_eq!(x.len(), tokens * self.cols());
        assert_eq!(out.len(), tokens * self.rows());
        match self {
            Self::F32 { rows, cols, weight } => linear(x, weight, tokens, *cols, *rows, out),
            Self::Q4(m) => {
                if tokens == 1 {
                    gemv_q4(m, x, out);
                } else {
                    gemm_q4(m, x, out, tokens);
                }
            }
        }
    }

    /// `dst[tokens, rows] += x[tokens, cols] @ W^T` (f64 residual add).
    pub fn apply_add(&self, x: &[f32], tokens: usize, dst: &mut [f32]) {
        match self {
            Self::F32 { rows, cols, weight } => linear_add(x, weight, tokens, *cols, *rows, dst),
            Self::Q4(m) => {
                if tokens == 1 {
                    gemv_q4_add(m, x, dst);
                } else {
                    gemm_q4_add(m, x, dst, tokens);
                }
            }
        }
    }
}

/// Fused SwiGLU gate (w1) and up (w3) projections sharing an activation.
pub fn apply_pair(
    w1: &Linear,
    w3: &Linear,
    x: &[f32],
    tokens: usize,
    out1: &mut [f32],
    out3: &mut [f32],
) {
    match (w1, w3) {
        (Linear::Q4(a), Linear::Q4(b)) if tokens == 1 => {
            gemv_q4_pair(a, b, x, out1, out3);
        }
        _ => {
            w1.apply(x, tokens, out1);
            w3.apply(x, tokens, out3);
        }
    }
}

/// Optional Q8_0 tied language-model head (same math as a Q8 GEMV).
pub struct Q8Matrix {
    pub n: usize,
    pub k: usize,
    pub blocks: Vec<BlockQ8_0>,
}

impl Q8Matrix {
    #[must_use]
    pub fn from_f32(weight: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(weight.len(), n * k);
        assert!(k.is_multiple_of(BLOCK_SIZE));
        Self {
            n,
            k,
            blocks: quantize_q8_0(weight),
        }
    }

    pub fn prefault(&self) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(self.blocks.as_slice()),
            )
        };
        prefault_bytes(bytes);
    }
}

/// Tied output head: same storage as a [`Linear`], plus a Q8 option.
pub enum OutputHead {
    Linear(Linear),
    Q8(Q8Matrix),
}

impl OutputHead {
    pub fn from_embed(
        embed: &[f32],
        vocab: usize,
        hidden: usize,
        format: WeightFormat,
    ) -> Result<Self, LoadError> {
        match format {
            WeightFormat::Q8_0 => Ok(Self::Q8(Q8Matrix::from_f32(embed, vocab, hidden))),
            other => Ok(Self::Linear(Linear::from_f32(
                embed.to_vec(),
                vocab,
                hidden,
                other,
            )?)),
        }
    }

    pub fn prefault(&self) {
        match self {
            Self::Linear(l) => l.prefault(),
            Self::Q8(m) => m.prefault(),
        }
    }

    pub fn apply(&self, x: &[f32], tokens: usize, out: &mut [f32]) {
        match self {
            Self::Linear(l) => l.apply(x, tokens, out),
            Self::Q8(m) => q8_apply(m, x, tokens, out),
        }
    }
}

fn q8_apply(m: &Q8Matrix, x: &[f32], tokens: usize, out: &mut [f32]) {
    let bpr = m.k / BLOCK_SIZE;
    if tokens == 1 {
        let acts = ActQ8::quantize(x);
        let out_addr = out.as_mut_ptr() as usize;
        for_ranges(m.n, |_tid, r0, r1| {
            let out_ptr = out_addr as *mut f32;
            for row in r0..r1 {
                let mut acc = 0.0f32;
                let row_w = &m.blocks[row * bpr..(row + 1) * bpr];
                for (b, block) in row_w.iter().enumerate() {
                    let a = &acts.qs[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                    let mut s = 0i32;
                    for (&wq, &aq) in block.qs.iter().zip(a.iter()) {
                        s += i32::from(wq) * i32::from(aq);
                    }
                    acc += s as f32 * block.scale.to_f32() * acts.scales[b];
                }
                unsafe {
                    *out_ptr.add(row) = acc;
                }
            }
        });
        return;
    }
    for t in 0..tokens {
        q8_apply(
            m,
            &x[t * m.k..(t + 1) * m.k],
            1,
            &mut out[t * m.n..(t + 1) * m.n],
        );
    }
}
