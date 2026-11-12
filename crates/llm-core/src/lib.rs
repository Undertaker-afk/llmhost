//! Core types for the llmhost inference engine.
//!
//! The model graph is config-driven: a Hugging Face `config.json` becomes a
//! [`Vec<LayerKind>`] so LFM2 and LFM3 can share the same runtime.

mod backend;
mod block;
mod config;
mod model;
mod shape;
mod tensor;

pub use backend::Backend;
pub use block::Block;
pub use config::LayerKind;
pub use config::{ConfigError, ModelConfig};
pub use model::Model;
pub use shape::Shape;
pub use tensor::{Tensor, TensorError};
