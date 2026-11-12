//! Hook for a speculative draft model.
//!
//! The draft proposes a token list plus a parent-index tree. The engine
//! verifies the tree with [`crate::validate_tree`] and later accepts any
//! valid path (not necessarily a prefix) after tree verification.

use crate::batch::SeqInput;

/// Draft tokens and the parent-index tree that produced them.
///
/// `tree[i]` is the parent of `tokens[i]` (`-1` = root). `tree` and
/// `tokens` must have the same length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftProposal {
    pub tokens: Vec<u32>,
    pub tree: Vec<i32>,
}

/// A (usually smaller, faster) model that proposes a continuation tree.
pub trait DraftModel {
    /// Propose draft tokens and a tree for `seq`.
    fn propose(&self, seq: &SeqInput) -> DraftProposal;
}

#[cfg(test)]
mod tests {
    use super::{DraftModel, DraftProposal};
    use crate::batch::SeqInput;
    use crate::tree::validate_tree_for_tokens;

    struct BranchingDraft;

    impl DraftModel for BranchingDraft {
        fn propose(&self, seq: &SeqInput) -> DraftProposal {
            let last = *seq.tokens.last().unwrap_or(&0);
            DraftProposal {
                tokens: vec![last + 1, last + 2, last + 3],
                // 0
                // ├─ 1
                // └─ 2
                tree: vec![-1, 0, 0],
            }
        }
    }

    #[test]
    fn draft_hook_returns_a_valid_tree() {
        let draft = BranchingDraft;
        let seq = SeqInput::linear(1, vec![9], 0);
        let proposal = draft.propose(&seq);
        validate_tree_for_tokens(&proposal.tree, proposal.tokens.len()).unwrap();
        assert_eq!(proposal.tokens, vec![10, 11, 12]);
    }
}
