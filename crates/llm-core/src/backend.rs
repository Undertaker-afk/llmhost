/// Compute backend (CPU, wgpu, CUDA, Metal).
///
/// Methods for copies and kernels land with the first real backend.
pub trait Backend: Send + Sync {
    /// Device-side buffer type for this backend.
    type Buffer: Send + Sync;

    /// Short name, such as `"cpu"` or `"wgpu"`.
    fn name(&self) -> &'static str;

    /// Allocate a buffer that can hold `len` f32 elements.
    fn alloc(&self, len: usize) -> Self::Buffer;
}
