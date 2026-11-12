//! CPU backend: runtime feature detection and kernels.
//!
//! Decode prefers AVX-512 VNNI GEMV; prefill prefers AMX int8 tiles. Both
//! fall back to AVX2, then to a scalar reference. Selection is always at
//! runtime; builds must not use `-C target-cpu=native`.

mod conv;
mod features;
mod gemm;
mod gemv_q4;
mod matvec;
mod ops;
mod pool;
mod rope;

pub use conv::{causal_dwconv_prefill, causal_dwconv_update};
pub use features::{CpuFeatures, KernelSet};
pub use gemm::{gemm_nt, gemv_nt, linear, linear_add};
pub use gemv_q4::{
    detect_q4_kernel, gemm_q4, gemv_q4, gemv_q4_pair, gemv_q4_prep, ActQ8, Q4Kernel, Q4Matrix,
};
pub use matvec::{matvec_f32, matvec_f32_avx2, matvec_f32_scalar};
pub use ops::{add, add_inplace, argmax, mul, rms_norm, silu, silu_inplace, softmax_last_axis};
pub use pool::with_threads;
pub use rope::{apply_rope_in_place, cos_sin, inv_freq};

use llm_core::Backend;

/// CPU backend. Kernel dispatch is based on [`CpuFeatures::detect`].
#[derive(Clone, Debug)]
pub struct CpuBackend {
    /// Detected instruction-set flags.
    pub features: CpuFeatures,
    /// Preferred kernel set for decode (VNNI GEMV when present).
    pub decode: KernelSet,
    /// Preferred kernel set for prefill (AMX when present).
    pub prefill: KernelSet,
}

impl CpuBackend {
    /// Detect features on this machine and pick kernel sets.
    #[must_use]
    pub fn new() -> Self {
        let features = CpuFeatures::detect();
        Self {
            decode: features.decode_kernels(),
            prefill: features.prefill_kernels(),
            features,
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn alloc(&self, len: usize) -> Self::Buffer {
        vec![0.0; len]
    }
}

#[cfg(test)]
mod tests {
    use super::CpuBackend;
    use llm_core::Backend;

    #[test]
    fn backend_allocates() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.alloc(4).len(), 4);
    }
}
