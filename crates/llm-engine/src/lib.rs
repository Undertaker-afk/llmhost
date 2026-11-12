//! Inference engine: sessions, caches, batching, and sampling.
//!
//! The batch-first API in [`batch`] returns logits for every token position
//! and can accept or roll back a speculative tree by token index (the
//! accepted path need not be a prefix). Temperature / nucleus sampling is
//! still a stub; [`Sampler::sample`] is greedy argmax so tests stay
//! deterministic.

mod batch;
mod cache;
mod draft;
mod lfm2;
mod load;
mod npy;
mod schedule;
mod tree;
mod weights;

pub use batch::{Engine, EngineError, ForwardBatch, ForwardOutput, SeqInput, SeqLogits};
pub use cache::{ConvSnapshot, ConvState, PagedKv, KV_BLOCK_TOKENS};
pub use draft::{DraftModel, DraftProposal};
pub use lfm2::{LayerTraces, Lfm2Cache, Lfm2Model};
pub use load::{find_hf_snapshot, load_config, load_safetensors_f32, LoadError};
pub use npy::{read_npy, read_npz, NpyArray, NpyData};
pub use schedule::Scheduler;
pub use tree::{
    effective_tree, linear_tree, validate_accept_path, validate_tree, validate_tree_for_tokens,
    TreeError,
};

/// Sampling parameters for a generation run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerConfig {
    /// Softmax temperature. `0.0` is treated as greedy in later milestones.
    pub temperature: f32,
    /// Optional top-k cutoff (keep the k largest logits).
    pub top_k: Option<u32>,
    /// Optional nucleus (top-p) cutoff in `(0, 1]`.
    pub top_p: Option<f32>,
    /// Optional min-p cutoff.
    pub min_p: Option<f32>,
    /// Seed for the (not yet implemented) RNG.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: 0,
        }
    }
}

/// Sampler. Nucleus / temperature sampling is not implemented yet; [`Sampler::sample`]
/// is greedy argmax so tests stay deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampler {
    pub config: SamplerConfig,
}

impl Sampler {
    #[must_use]
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Pick a token id from `logits`. Currently always greedy (see [`greedy_argmax`]).
    #[must_use]
    pub fn sample(&self, logits: &[f32]) -> usize {
        let _ = self.config;
        greedy_argmax(logits)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(SamplerConfig::default())
    }
}

/// Return the index of the largest logit. Ties break to the lowest index.
///
/// This is deterministic and does not depend on a seed.
#[must_use]
pub fn greedy_argmax(logits: &[f32]) -> usize {
    assert!(
        !logits.is_empty(),
        "greedy_argmax requires at least one logit"
    );
    let mut best_i = 0usize;
    let mut best = logits[0];
    for (i, &value) in logits.iter().enumerate().skip(1) {
        if value > best {
            best = value;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::{greedy_argmax, Sampler, SamplerConfig};

    #[test]
    fn greedy_argmax_is_deterministic_and_breaks_ties_low() {
        let logits = [0.1, 2.5, 2.5, -4.0, 1.0];
        assert_eq!(greedy_argmax(&logits), 1);
        assert_eq!(greedy_argmax(&logits), 1);

        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_k: Some(5),
            top_p: Some(0.9),
            min_p: Some(0.05),
            seed: 123,
        });
        assert_eq!(sampler.sample(&logits), 1);
        assert_eq!(Sampler::default().sample(&[-1.0, -0.5, -3.0]), 1);
    }
}
