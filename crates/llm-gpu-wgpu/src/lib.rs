//! Portable GPU backend for llmhost, implemented with wgpu.
//!
//! This is the default GPU path: WGSL compute shaders running on Vulkan,
//! Metal, or DX12. It will be used for correctness via lavapipe (software Vulkan) in
//! milestone 4. Real throughput numbers need a physical GPU.
//!
//! No dependencies are declared yet; shaders and the `wgpu` crate land with
//! the first kernels.

/// Marker for the future wgpu backend.
pub struct WgpuBackend;
