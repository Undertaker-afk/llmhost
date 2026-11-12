//! Causal depthwise conv1d, kernel along the last axis of each channel.
//!
//! Weight layout matches `nn.Conv1d` after `squeeze(1)`: `w[channel, k]`,
//! with `w[..., 0]` multiplying the oldest sample (left / past).

/// Prefill: `out[t, h] = sum_k w[h, k] * x[t - (K-1) + k, h]`, with x<0 treated as 0.
/// `x` and `out` are `[tokens, hidden]`; `w` is `[hidden, kernel]`.
///
/// After the call, `state` (length `hidden * (kernel-1)`) holds the last
/// `kernel-1` inputs of each channel, oldest first, for incremental decode.
pub fn causal_dwconv_prefill(
    x: &[f32],
    w: &[f32],
    tokens: usize,
    hidden: usize,
    kernel: usize,
    out: &mut [f32],
    state: &mut [f32],
) {
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(out.len(), tokens * hidden);
    assert_eq!(w.len(), hidden * kernel);
    let history = kernel - 1;
    assert_eq!(state.len(), hidden * history);

    for t in 0..tokens {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..kernel {
                let src = t as isize - history as isize + k as isize;
                let v = if src >= 0 {
                    x[src as usize * hidden + h]
                } else {
                    0.0
                };
                acc = w[h * kernel + k].mul_add(v, acc);
            }
            out[t * hidden + h] = acc;
        }
    }
    // Last `history` tokens of x (pad with zeros if the prompt is shorter).
    state.fill(0.0);
    for i in 0..history {
        let src = tokens as isize - history as isize + i as isize;
        if src >= 0 {
            let t = src as usize;
            for h in 0..hidden {
                state[h * history + i] = x[t * hidden + h];
            }
        }
    }
}

/// Incremental decode of one token: `x` and `out` are `[hidden]`.
/// Updates `state` in place (`[hidden, history]`, oldest first).
pub fn causal_dwconv_update(
    x: &[f32],
    w: &[f32],
    hidden: usize,
    kernel: usize,
    out: &mut [f32],
    state: &mut [f32],
) {
    assert_eq!(x.len(), hidden);
    assert_eq!(out.len(), hidden);
    assert_eq!(w.len(), hidden * kernel);
    let history = kernel - 1;
    assert_eq!(state.len(), hidden * history);
    for h in 0..hidden {
        let mut acc = 0.0f32;
        for k in 0..history {
            acc = w[h * kernel + k].mul_add(state[h * history + k], acc);
        }
        acc = w[h * kernel + history].mul_add(x[h], acc);
        out[h] = acc;
        // Shift state and append x.
        for k in 0..history.saturating_sub(1) {
            state[h * history + k] = state[h * history + k + 1];
        }
        if history > 0 {
            state[h * history + history - 1] = x[h];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{causal_dwconv_prefill, causal_dwconv_update};

    #[test]
    fn prefill_then_update_matches_one_shot() {
        let hidden = 2;
        let kernel = 3;
        // 4 tokens
        let x = vec![
            1.0, 10.0, // t0
            2.0, 20.0, // t1
            3.0, 30.0, // t2
            4.0, 40.0, // t3
        ];
        let w = vec![
            0.1, 0.2, 0.3, // ch0
            1.0, 0.0, -1.0, // ch1
        ];
        let mut full = vec![0.0; 8];
        let mut state = vec![0.0; hidden * 2];
        causal_dwconv_prefill(&x, &w, 4, hidden, kernel, &mut full, &mut state);

        // Prefill first 3, then update the 4th.
        let mut prefix = vec![0.0; 6];
        let mut st = vec![0.0; hidden * 2];
        causal_dwconv_prefill(&x[..6], &w, 3, hidden, kernel, &mut prefix, &mut st);
        let mut last = [0.0f32; 2];
        causal_dwconv_update(&x[6..], &w, hidden, kernel, &mut last, &mut st);
        assert!((last[0] - full[6]).abs() < 1e-6);
        assert!((last[1] - full[7]).abs() < 1e-6);
        assert_eq!(st, state);
    }
}
