//! Placeholder paged KV cache and rolling conv state.
//!
//! No real kernels live here. The types track occupancy and snapshots so
//! accept / rollback of a speculative tree can be tested.

use llm_core::LayerKind;
use llm_quant::KvDtype;

/// Tokens stored in one KV page.
pub const KV_BLOCK_TOKENS: usize = 16;

/// Paged key/value cache for attention layers (blocks of 16 tokens).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagedKv {
    pub dtype: KvDtype,
    token_count: usize,
    allocated_blocks: usize,
}

impl PagedKv {
    #[must_use]
    pub fn new(dtype: KvDtype) -> Self {
        Self {
            dtype,
            token_count: 0,
            allocated_blocks: 0,
        }
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    #[must_use]
    pub fn allocated_blocks(&self) -> usize {
        self.allocated_blocks
    }

    /// Number of 16-token blocks needed to hold `tokens` tokens.
    #[must_use]
    pub fn blocks_for(tokens: usize) -> usize {
        tokens.div_ceil(KV_BLOCK_TOKENS)
    }

    /// Resize occupancy to `tokens` tokens (grows or shrinks the page count).
    pub fn set_token_count(&mut self, tokens: usize) {
        self.token_count = tokens;
        self.allocated_blocks = Self::blocks_for(tokens);
    }
}

/// Snapshot of a [`ConvState`] taken after a forwarded token, used to roll
/// back rejected branches of a speculative tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvSnapshot {
    steps: usize,
    /// Token ids applied up to this snapshot (placeholder for the rolling buffer).
    applied: Vec<u32>,
}

/// Rolling conv-state for every conv layer in the graph.
///
/// The real buffer length is `kernel - 1` per layer. This placeholder only
/// records which tokens have been applied so accept/rollback can restore it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvState {
    /// One slot per conv layer, holding the kernel size for later kernels.
    pub kernels: Vec<usize>,
    steps: usize,
    applied: Vec<u32>,
}

impl ConvState {
    /// Build an empty state, one entry per [`LayerKind::Conv`] in `layers`.
    #[must_use]
    pub fn from_layers(layers: &[LayerKind]) -> Self {
        let kernels = layers
            .iter()
            .filter_map(|kind| match kind {
                LayerKind::Conv { kernel } => Some(*kernel),
                _ => None,
            })
            .collect();
        Self {
            kernels,
            steps: 0,
            applied: Vec::new(),
        }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            kernels: Vec::new(),
            steps: 0,
            applied: Vec::new(),
        }
    }

    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Token ids applied so far, in order. Test/debug bookkeeping only.
    #[must_use]
    pub fn applied(&self) -> &[u32] {
        &self.applied
    }

    /// Apply one forwarded token (placeholder: just record the id).
    pub fn apply_token(&mut self, token: u32) {
        self.applied.push(token);
        self.steps += 1;
    }

    #[must_use]
    pub fn snapshot(&self) -> ConvSnapshot {
        ConvSnapshot {
            steps: self.steps,
            applied: self.applied.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &ConvSnapshot) {
        self.steps = snapshot.steps;
        self.applied.clone_from(&snapshot.applied);
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvState, PagedKv, KV_BLOCK_TOKENS};
    use llm_core::LayerKind;
    use llm_quant::KvDtype;

    #[test]
    fn paged_kv_uses_16_token_blocks() {
        assert_eq!(KV_BLOCK_TOKENS, 16);
        assert_eq!(PagedKv::blocks_for(0), 0);
        assert_eq!(PagedKv::blocks_for(1), 1);
        assert_eq!(PagedKv::blocks_for(16), 1);
        assert_eq!(PagedKv::blocks_for(17), 2);
        let mut kv = PagedKv::new(KvDtype::F16);
        kv.set_token_count(17);
        assert_eq!(kv.token_count(), 17);
        assert_eq!(kv.allocated_blocks(), 2);
        kv.set_token_count(16);
        assert_eq!(kv.allocated_blocks(), 1);
    }

    #[test]
    fn conv_state_snapshot_restore() {
        let layers = [
            LayerKind::Conv { kernel: 3 },
            LayerKind::Attention {
                sliding_window: None,
                rope: llm_core::RopeParams::default(),
            },
            LayerKind::Conv { kernel: 7 },
        ];
        let mut state = ConvState::from_layers(&layers);
        assert_eq!(state.kernels, vec![3, 7]);
        state.apply_token(10);
        let snap = state.snapshot();
        state.apply_token(11);
        assert_eq!(state.applied(), &[10, 11]);
        state.restore(&snap);
        assert_eq!(state.applied(), &[10]);
        assert_eq!(state.steps(), 1);
    }
}
