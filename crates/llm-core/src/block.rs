use crate::{LayerKind, Tensor};

/// One model layer (conv, attention, or a future LFM3 type such as MoE).
///
/// The engine walks a `Vec` of blocks built from [`crate::ModelConfig::layers`].
pub trait Block {
    /// Kind of this layer, used to size the KV cache vs the rolling conv state.
    fn kind(&self) -> LayerKind;

    /// Apply the block to `input` and write into `output`.
    ///
    /// Residual wiring and exact shapes are defined by the concrete block.
    fn forward(&self, input: &Tensor, output: &mut Tensor);
}
