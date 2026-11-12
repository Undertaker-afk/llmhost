//! Speculative-decoding tree masks.
//!
//! Each entry is the parent index of that token in the same list. `-1` marks
//! a root (attached to the already-committed prefix). `None` on a
//! [`crate::SeqInput`] means a linear chain: `[-1, 0, 1, ...]`.

use std::fmt;

/// Error from an invalid parent-index tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeError {
    /// `tree.len()` did not match the token list.
    LengthMismatch { tree: usize, tokens: usize },
    /// Parent index is not `-1` and not a valid index into the tree.
    ParentOutOfRange { child: usize, parent: i32 },
    /// Parent index is greater than or equal to the child index.
    /// Parents must precede children so a tree can be applied left to right.
    ParentDoesNotPrecede { child: usize, parent: i32 },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { tree, tokens } => {
                write!(f, "tree length {tree} does not match token length {tokens}")
            }
            Self::ParentOutOfRange { child, parent } => {
                write!(f, "token {child} has out-of-range parent {parent}")
            }
            Self::ParentDoesNotPrecede { child, parent } => {
                write!(f, "parent {parent} does not precede child {child}")
            }
        }
    }
}

impl std::error::Error for TreeError {}

/// A linear left-to-right chain of `len` tokens: `[-1, 0, 1, …, len-2]`.
#[must_use]
pub fn linear_tree(len: usize) -> Vec<i32> {
    (0..len)
        .map(|i| if i == 0 { -1 } else { (i - 1) as i32 })
        .collect()
}

/// Check that every parent is either `-1` (a root) or an earlier index.
pub fn validate_tree(tree: &[i32]) -> Result<(), TreeError> {
    for (child, &parent) in tree.iter().enumerate() {
        if parent == -1 {
            continue;
        }
        if parent < 0 || (parent as usize) >= tree.len() {
            return Err(TreeError::ParentOutOfRange { child, parent });
        }
        if (parent as usize) >= child {
            return Err(TreeError::ParentDoesNotPrecede { child, parent });
        }
    }
    Ok(())
}

/// Like [`validate_tree`], but also requires `tree.len() == tokens`.
pub fn validate_tree_for_tokens(tree: &[i32], tokens: usize) -> Result<(), TreeError> {
    if tree.len() != tokens {
        return Err(TreeError::LengthMismatch {
            tree: tree.len(),
            tokens,
        });
    }
    validate_tree(tree)
}

/// Resolve an optional tree: `None` becomes a linear chain of `tokens` length.
pub fn effective_tree(tree: Option<&[i32]>, tokens: usize) -> Result<Vec<i32>, TreeError> {
    match tree {
        None => Ok(linear_tree(tokens)),
        Some(tree) => {
            validate_tree_for_tokens(tree, tokens)?;
            Ok(tree.to_vec())
        }
    }
}

/// Check that `accepted` is a connected path in `tree`.
///
/// The accepted set need not be a prefix of the token list (a branch can be
/// chosen), but every accepted token's parent must also be accepted, or be a
/// root (`-1`).
pub fn validate_accept_path(tree: &[i32], accepted: &[usize]) -> Result<(), TreeError> {
    use std::collections::HashSet;
    let set: HashSet<usize> = accepted.iter().copied().collect();
    if set.len() != accepted.len() {
        // Treat a duplicate as a self-parenting / ill-formed path.
        return Err(TreeError::ParentDoesNotPrecede {
            child: accepted
                .iter()
                .copied()
                .find(|i| accepted.iter().filter(|j| *j == i).count() > 1)
                .unwrap_or(0),
            parent: -2,
        });
    }
    for &child in accepted {
        if child >= tree.len() {
            return Err(TreeError::ParentOutOfRange {
                child,
                parent: child as i32,
            });
        }
        let parent = tree[child];
        if parent == -1 {
            continue;
        }
        if parent < 0 || (parent as usize) >= tree.len() {
            return Err(TreeError::ParentOutOfRange { child, parent });
        }
        if !set.contains(&(parent as usize)) {
            return Err(TreeError::ParentDoesNotPrecede { child, parent });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        effective_tree, linear_tree, validate_accept_path, validate_tree, validate_tree_for_tokens,
        TreeError,
    };

    #[test]
    fn linear_chain_is_valid() {
        let tree = linear_tree(4);
        assert_eq!(tree, vec![-1, 0, 1, 2]);
        validate_tree(&tree).unwrap();
        assert_eq!(effective_tree(None, 4).unwrap(), tree);
    }

    #[test]
    fn branching_tree_parents_precede_children() {
        // 0
        // ├─ 1
        // │  └─ 3
        // └─ 2
        let tree = [-1, 0, 0, 1];
        validate_tree(&tree).unwrap();
        validate_accept_path(&tree, &[0, 1, 3]).unwrap();
        validate_accept_path(&tree, &[0, 2]).unwrap();
        // Not a path: 3's parent 1 is missing.
        assert!(validate_accept_path(&tree, &[0, 3]).is_err());
    }

    #[test]
    fn parent_must_precede_child() {
        assert_eq!(
            validate_tree(&[-1, 2, 0]).unwrap_err(),
            TreeError::ParentDoesNotPrecede {
                child: 1,
                parent: 2
            }
        );
        assert_eq!(
            validate_tree(&[0]).unwrap_err(),
            TreeError::ParentDoesNotPrecede {
                child: 0,
                parent: 0
            }
        );
        assert_eq!(
            validate_tree(&[-1, -3]).unwrap_err(),
            TreeError::ParentOutOfRange {
                child: 1,
                parent: -3
            }
        );
        assert_eq!(
            validate_tree_for_tokens(&[-1, 0], 3).unwrap_err(),
            TreeError::LengthMismatch { tree: 2, tokens: 3 }
        );
    }
}
