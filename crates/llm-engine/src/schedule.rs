//! Continuous-batching scheduler (placeholder).
//!
//! Tracks live sequence ids and packs them into a [`ForwardBatch`]. Filling
//! tokens, budgets, and preemption lands with the real engine.

use crate::batch::{ForwardBatch, SeqInput};

/// In-flight sequences packed into one forward batch per step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scheduler {
    seqs: Vec<u64>,
}

impl Scheduler {
    #[must_use]
    pub fn new() -> Self {
        Self { seqs: Vec::new() }
    }

    pub fn add(&mut self, seq_id: u64) {
        if !self.seqs.contains(&seq_id) {
            self.seqs.push(seq_id);
        }
    }

    pub fn remove(&mut self, seq_id: u64) {
        self.seqs.retain(|id| *id != seq_id);
    }

    /// Sequence ids currently eligible to run.
    #[must_use]
    pub fn running(&self) -> &[u64] {
        &self.seqs
    }

    /// Build a batch for the running set. Tokens and block tables are empty
    /// in this skeleton; the engine fills them in a later milestone.
    #[must_use]
    pub fn schedule(&self) -> ForwardBatch {
        ForwardBatch {
            seqs: self
                .seqs
                .iter()
                .map(|&seq_id| SeqInput::linear(seq_id, Vec::new(), 0))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Scheduler;

    #[test]
    fn packs_running_sequences() {
        let mut sched = Scheduler::new();
        sched.add(1);
        sched.add(2);
        sched.add(1);
        assert_eq!(sched.running(), &[1, 2]);
        let batch = sched.schedule();
        assert_eq!(batch.seqs.len(), 2);
        assert_eq!(batch.seqs[0].seq_id, 1);
        sched.remove(1);
        assert_eq!(sched.running(), &[2]);
    }
}
