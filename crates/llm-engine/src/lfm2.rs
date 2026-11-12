//! LFM2 fp32 inference: prefill over a full prompt and incremental decode.
//!
//! The implementation follows `transformers.models.lfm2.modeling_lfm2` (5.22.2):
//! RMSNorm, short conv (`in_proj` → B,C,x gating, depthwise causal conv,
//! `out_proj`), GQA attention with q/k RMSNorm and RoPE, SwiGLU FFN,
//! `embedding_norm`, and a tied `lm_head`.

use crate::load::{
    find_hf_snapshot, get_tensor, load_config, load_safetensors_f32, squeeze_conv_weight,
    LoadError, TensorF32,
};
use llm_core::{LayerKind, ModelConfig};
use llm_cpu::{
    add_inplace, apply_rope_in_place, argmax, causal_dwconv_prefill, causal_dwconv_update, cos_sin,
    inv_freq, linear, mul, rms_norm, silu_inplace, softmax_last_axis,
};
use std::collections::HashMap;
use std::path::Path;

/// Intermediate residual (and other) activations captured during a prefill.
#[derive(Clone, Debug, Default)]
pub struct LayerTraces {
    pub embed: Vec<f32>,
    pub layers: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

struct Rms {
    weight: Vec<f32>,
}

struct Ffn {
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    ffn_norm: Rms,
}

struct ConvBlock {
    operator_norm: Rms,
    in_proj: Vec<f32>,
    conv_w: Vec<f32>,
    out_proj: Vec<f32>,
    ffn: Ffn,
    kernel: usize,
}

struct AttnBlock {
    operator_norm: Rms,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    out_proj: Vec<f32>,
    q_norm: Rms,
    k_norm: Rms,
    ffn: Ffn,
}

enum Block {
    Conv(ConvBlock),
    Attn(AttnBlock),
}

/// Rolling conv state and contiguous fp32 KV cache for incremental decode.
pub struct Lfm2Cache {
    /// Tokens already committed (prompt + generated).
    pub seq_len: usize,
    /// Per conv layer: `[hidden * (kernel-1)]`, oldest first. `None` for attention layers.
    conv: Vec<Option<Vec<f32>>>,
    /// Per attention layer: keys `[n_kv, seq, head_dim]`.
    k: Vec<Option<Vec<f32>>>,
    /// Per attention layer: values `[n_kv, seq, head_dim]`.
    v: Vec<Option<Vec<f32>>>,
    /// Reserved capacity along the time axis of K/V (so greedy decode does not realloc).
    kv_cap: usize,
}

/// LFM2 model with fp32 weights.
pub struct Lfm2Model {
    pub config: ModelConfig,
    embed: Vec<f32>,
    embedding_norm: Rms,
    blocks: Vec<Block>,
    inv_freq: Vec<f32>,
}

impl Lfm2Model {
    /// Load from a snapshot directory that has `config.json` + `model.safetensors`.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = load_config(dir)?;
        let tensors = load_safetensors_f32(dir.join("model.safetensors"))?;
        Self::from_tensors(config, &tensors)
    }

    /// Locate the LFM2-700M snapshot in the local Hub cache (no download) and load it.
    pub fn load_lfm2_700m() -> Result<Self, LoadError> {
        let dir = find_hf_snapshot("LiquidAI/LFM2-700M", "86f49fc9")?;
        Self::load_dir(dir)
    }

    fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, TensorF32>,
    ) -> Result<Self, LoadError> {
        let h = config.hidden_size;
        let embed = get_tensor(tensors, "model.embed_tokens.weight")?;
        expect_shape(&embed.shape, &[config.vocab_size, h], "embed_tokens")?;
        let embedding_norm = rms(get_tensor(tensors, "model.embedding_norm.weight")?, h)?;
        let mut blocks = Vec::with_capacity(config.layers.len());
        for (i, kind) in config.layers.iter().enumerate() {
            let prefix = format!("model.layers.{i}");
            let ffn = Ffn {
                w1: named(tensors, &format!("{prefix}.feed_forward.w1.weight"))?
                    .data
                    .clone(),
                w2: named(tensors, &format!("{prefix}.feed_forward.w2.weight"))?
                    .data
                    .clone(),
                w3: named(tensors, &format!("{prefix}.feed_forward.w3.weight"))?
                    .data
                    .clone(),
                ffn_norm: rms(
                    get_tensor(tensors, &format!("{prefix}.ffn_norm.weight"))?,
                    h,
                )?,
            };
            match kind {
                LayerKind::Conv { kernel } => {
                    let conv = get_tensor(tensors, &format!("{prefix}.conv.conv.weight"))?;
                    blocks.push(Block::Conv(ConvBlock {
                        operator_norm: rms(
                            get_tensor(tensors, &format!("{prefix}.operator_norm.weight"))?,
                            h,
                        )?,
                        in_proj: named(tensors, &format!("{prefix}.conv.in_proj.weight"))?
                            .data
                            .clone(),
                        conv_w: squeeze_conv_weight(conv)?,
                        out_proj: named(tensors, &format!("{prefix}.conv.out_proj.weight"))?
                            .data
                            .clone(),
                        ffn,
                        kernel: *kernel,
                    }));
                }
                LayerKind::Attention { .. } => {
                    blocks.push(Block::Attn(AttnBlock {
                        operator_norm: rms(
                            get_tensor(tensors, &format!("{prefix}.operator_norm.weight"))?,
                            h,
                        )?,
                        q_proj: named(tensors, &format!("{prefix}.self_attn.q_proj.weight"))?
                            .data
                            .clone(),
                        k_proj: named(tensors, &format!("{prefix}.self_attn.k_proj.weight"))?
                            .data
                            .clone(),
                        v_proj: named(tensors, &format!("{prefix}.self_attn.v_proj.weight"))?
                            .data
                            .clone(),
                        out_proj: named(tensors, &format!("{prefix}.self_attn.out_proj.weight"))?
                            .data
                            .clone(),
                        q_norm: rms(
                            get_tensor(tensors, &format!("{prefix}.self_attn.q_layernorm.weight"))?,
                            config.head_dim,
                        )?,
                        k_norm: rms(
                            get_tensor(tensors, &format!("{prefix}.self_attn.k_layernorm.weight"))?,
                            config.head_dim,
                        )?,
                        ffn,
                    }));
                }
                other => {
                    return Err(LoadError::Message(format!(
                        "unsupported LFM2 layer {i}: {other:?}"
                    )));
                }
            }
        }
        let inv = inv_freq(
            config.head_dim,
            config
                .layers
                .iter()
                .find_map(|k| match k {
                    LayerKind::Attention { rope, .. } => Some(rope.theta as f32),
                    _ => None,
                })
                .unwrap_or(1_000_000.0),
        );
        Ok(Self {
            config,
            embed: embed.data.clone(),
            embedding_norm,
            blocks,
            inv_freq: inv,
        })
    }

    #[must_use]
    pub fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    #[must_use]
    pub fn vocab(&self) -> usize {
        self.config.vocab_size
    }

    /// Prefill `tokens`. If `traces` is `Some`, fills embed / per-layer residual /
    /// final_norm / logits for every position. Returns a cache ready for decode.
    pub fn prefill(
        &self,
        tokens: &[u32],
        mut traces: Option<&mut LayerTraces>,
        kv_reserve: usize,
    ) -> Lfm2Cache {
        let t = tokens.len();
        let h = self.hidden();
        let v = self.vocab();
        let mut hidden = vec![0.0f32; t * h];
        for (i, &id) in tokens.iter().enumerate() {
            let id = id as usize;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.embed[id * h..(id + 1) * h]);
        }
        if let Some(tr) = traces.as_mut() {
            tr.embed = hidden.clone();
            tr.layers.clear();
        }

        let cap = kv_reserve.max(t);
        let mut cache = Lfm2Cache {
            seq_len: t,
            conv: Vec::with_capacity(self.blocks.len()),
            k: Vec::with_capacity(self.blocks.len()),
            v: Vec::with_capacity(self.blocks.len()),
            kv_cap: cap,
        };

        for block in &self.blocks {
            match block {
                Block::Conv(conv) => {
                    let mut state = vec![0.0f32; h * conv.kernel.saturating_sub(1)];
                    self.conv_prefill(conv, &mut hidden, t, &mut state);
                    self.ffn_residual(&conv.ffn, &mut hidden, t);
                    cache.conv.push(Some(state));
                    cache.k.push(None);
                    cache.v.push(None);
                }
                Block::Attn(attn) => {
                    let (k, v) = self.attn_prefill(attn, &mut hidden, t, 0, cap);
                    self.ffn_residual(&attn.ffn, &mut hidden, t);
                    cache.conv.push(None);
                    cache.k.push(Some(k));
                    cache.v.push(Some(v));
                }
            }
            if let Some(tr) = traces.as_mut() {
                tr.layers.push(hidden.clone());
            }
        }

        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            &hidden,
            &self.embedding_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut logits = vec![0.0f32; t * v];
        linear(&normed, &self.embed, t, h, v, &mut logits);
        if let Some(tr) = traces.as_mut() {
            tr.final_norm = normed;
            tr.logits = logits;
        }
        cache
    }

    /// Decode one new token, appending to `cache`. Returns vocab logits for that step.
    pub fn decode_one(&self, token: u32, cache: &mut Lfm2Cache) -> Vec<f32> {
        let h = self.hidden();
        let v = self.vocab();
        let id = token as usize;
        let mut hidden = self.embed[id * h..(id + 1) * h].to_vec();
        let pos = cache.seq_len;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Conv(conv) => {
                    let state = cache.conv[layer_idx]
                        .as_mut()
                        .expect("conv state for conv layer");
                    self.conv_update(conv, &mut hidden, state);
                    self.ffn_residual(&conv.ffn, &mut hidden, 1);
                }
                Block::Attn(attn) => {
                    self.attn_decode(
                        attn,
                        &mut hidden,
                        pos,
                        cache.k[layer_idx].as_mut().unwrap(),
                        cache.v[layer_idx].as_mut().unwrap(),
                        cache.kv_cap,
                    );
                    self.ffn_residual(&attn.ffn, &mut hidden, 1);
                }
            }
        }
        cache.seq_len += 1;
        let mut normed = vec![0.0f32; h];
        rms_norm(
            &hidden,
            &self.embedding_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut logits = vec![0.0f32; v];
        linear(&normed, &self.embed, 1, h, v, &mut logits);
        logits
    }

    /// Greedy incremental decode of `steps` tokens, starting from the last
    /// prefill logits (the last row of `prefill_logits`, length `vocab`).
    pub fn greedy_decode(
        &self,
        cache: &mut Lfm2Cache,
        prefill_logits: &[f32],
        steps: usize,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let v = self.vocab();
        assert_eq!(prefill_logits.len() % v, 0);
        let last = &prefill_logits[prefill_logits.len() - v..];
        let mut ids = Vec::with_capacity(steps);
        let mut all_logits = Vec::with_capacity(steps);
        let mut next = argmax(last) as u32;
        for _ in 0..steps {
            let logits = self.decode_one(next, cache);
            ids.push(next);
            let chosen = argmax(&logits) as u32;
            all_logits.push(logits);
            next = chosen;
        }
        (ids, all_logits)
    }

    fn conv_prefill(&self, conv: &ConvBlock, hidden: &mut [f32], t: usize, state: &mut [f32]) {
        let h = self.hidden();
        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            hidden,
            &conv.operator_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut bcx = vec![0.0f32; t * 3 * h];
        linear(&normed, &conv.in_proj, t, h, 3 * h, &mut bcx);
        // Split B, C, x along the last axis of size 3H: [T, 3H] = concat_last(B, C, x)
        let mut gate_b = vec![0.0f32; t * h];
        let mut gate_c = vec![0.0f32; t * h];
        let mut gate_x = vec![0.0f32; t * h];
        for i in 0..t {
            let row = &bcx[i * 3 * h..(i + 1) * 3 * h];
            gate_b[i * h..(i + 1) * h].copy_from_slice(&row[0..h]);
            gate_c[i * h..(i + 1) * h].copy_from_slice(&row[h..2 * h]);
            gate_x[i * h..(i + 1) * h].copy_from_slice(&row[2 * h..3 * h]);
        }
        let mut u = vec![0.0f32; t * h];
        mul(&gate_b, &gate_x, &mut u);
        let mut conv_out = vec![0.0f32; t * h];
        causal_dwconv_prefill(&u, &conv.conv_w, t, h, conv.kernel, &mut conv_out, state);
        let mut y = vec![0.0f32; t * h];
        mul(&gate_c, &conv_out, &mut y);
        let mut projected = vec![0.0f32; t * h];
        linear(&y, &conv.out_proj, t, h, h, &mut projected);
        // residual: hidden = hidden + conv(norm(hidden))
        add_inplace(hidden, &projected);
    }

    fn conv_update(&self, conv: &ConvBlock, hidden: &mut [f32], state: &mut [f32]) {
        let h = self.hidden();
        let mut normed = vec![0.0f32; h];
        rms_norm(
            hidden,
            &conv.operator_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut bcx = vec![0.0f32; 3 * h];
        linear(&normed, &conv.in_proj, 1, h, 3 * h, &mut bcx);
        let (b, rest) = bcx.split_at(h);
        let (c, x) = rest.split_at(h);
        let mut u = vec![0.0f32; h];
        mul(b, x, &mut u);
        let mut conv_out = vec![0.0f32; h];
        causal_dwconv_update(&u, &conv.conv_w, h, conv.kernel, &mut conv_out, state);
        let mut y = vec![0.0f32; h];
        mul(c, &conv_out, &mut y);
        let mut projected = vec![0.0f32; h];
        linear(&y, &conv.out_proj, 1, h, h, &mut projected);
        add_inplace(hidden, &projected);
    }

    fn attn_prefill(
        &self,
        attn: &AttnBlock,
        hidden: &mut [f32],
        t: usize,
        start_pos: usize,
        kv_cap: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let h = self.hidden();
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let d = self.config.head_dim;
        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            hidden,
            &attn.operator_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );

        let mut q = vec![0.0f32; t * n_q * d];
        let mut k = vec![0.0f32; t * n_kv * d];
        let mut v = vec![0.0f32; t * n_kv * d];
        linear(&normed, &attn.q_proj, t, h, n_q * d, &mut q);
        linear(&normed, &attn.k_proj, t, h, n_kv * d, &mut k);
        linear(&normed, &attn.v_proj, t, h, n_kv * d, &mut v);
        // q/k RMSNorm over head_dim
        rms_norm_heads(&mut q, &attn.q_norm.weight, self.config.norm_eps, n_q, t, d);
        rms_norm_heads(
            &mut k,
            &attn.k_norm.weight,
            self.config.norm_eps,
            n_kv,
            t,
            d,
        );

        // Reshape to [heads, tokens, dim] for RoPE + attention.
        let mut q_htd = transpose_heads_in(&q, n_q, t, d);
        let mut k_htd = transpose_heads_in(&k, n_kv, t, d);
        let v_htd = transpose_heads_in(&v, n_kv, t, d);

        let mut cos = vec![0.0f32; t * d];
        let mut sin = vec![0.0f32; t * d];
        cos_sin(&self.inv_freq, start_pos, t, d, &mut cos, &mut sin);
        apply_rope_in_place(&mut q_htd, &cos, &sin, n_q, t, d);
        apply_rope_in_place(&mut k_htd, &cos, &sin, n_kv, t, d);

        let attn_out = attention_forward(&q_htd, &k_htd, &v_htd, n_q, n_kv, t, t, d, true);
        let flat = transpose_heads_out(&attn_out, n_q, t, d);
        let mut projected = vec![0.0f32; t * h];
        linear(&flat, &attn.out_proj, t, n_q * d, h, &mut projected);
        add_inplace(hidden, &projected);

        // Store K/V at capacity `kv_cap` along time: layout [n_kv, cap, d], first t filled.
        let mut k_store = vec![0.0f32; n_kv * kv_cap * d];
        let mut v_store = vec![0.0f32; n_kv * kv_cap * d];
        for head in 0..n_kv {
            for tok in 0..t {
                let src = (head * t + tok) * d;
                let dst = (head * kv_cap + tok) * d;
                k_store[dst..dst + d].copy_from_slice(&k_htd[src..src + d]);
                v_store[dst..dst + d].copy_from_slice(&v_htd[src..src + d]);
            }
        }
        (k_store, v_store)
    }

    fn attn_decode(
        &self,
        attn: &AttnBlock,
        hidden: &mut [f32],
        pos: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        kv_cap: usize,
    ) {
        let h = self.hidden();
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let d = self.config.head_dim;
        let kv_len = pos + 1;
        assert!(kv_len <= kv_cap, "KV cache overflow: {kv_len} > {kv_cap}");

        let mut normed = vec![0.0f32; h];
        rms_norm(
            hidden,
            &attn.operator_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut q = vec![0.0f32; n_q * d];
        let mut k = vec![0.0f32; n_kv * d];
        let mut v = vec![0.0f32; n_kv * d];
        linear(&normed, &attn.q_proj, 1, h, n_q * d, &mut q);
        linear(&normed, &attn.k_proj, 1, h, n_kv * d, &mut k);
        linear(&normed, &attn.v_proj, 1, h, n_kv * d, &mut v);
        rms_norm_heads(&mut q, &attn.q_norm.weight, self.config.norm_eps, n_q, 1, d);
        rms_norm_heads(
            &mut k,
            &attn.k_norm.weight,
            self.config.norm_eps,
            n_kv,
            1,
            d,
        );

        // Already [heads, 1, dim]
        let mut cos = vec![0.0f32; d];
        let mut sin = vec![0.0f32; d];
        cos_sin(&self.inv_freq, pos, 1, d, &mut cos, &mut sin);
        apply_rope_in_place(&mut q, &cos, &sin, n_q, 1, d);
        apply_rope_in_place(&mut k, &cos, &sin, n_kv, 1, d);

        // Append K/V at time `pos`.
        for head in 0..n_kv {
            let dst = (head * kv_cap + pos) * d;
            k_cache[dst..dst + d].copy_from_slice(&k[head * d..(head + 1) * d]);
            v_cache[dst..dst + d].copy_from_slice(&v[head * d..(head + 1) * d]);
        }

        // Pack the filled prefix of the cache into a dense [n_kv, kv_len, d] for attention.
        let mut k_full = vec![0.0f32; n_kv * kv_len * d];
        let mut v_full = vec![0.0f32; n_kv * kv_len * d];
        for head in 0..n_kv {
            for tok in 0..kv_len {
                let src = (head * kv_cap + tok) * d;
                let dst = (head * kv_len + tok) * d;
                k_full[dst..dst + d].copy_from_slice(&k_cache[src..src + d]);
                v_full[dst..dst + d].copy_from_slice(&v_cache[src..src + d]);
            }
        }
        let attn_out = attention_forward(&q, &k_full, &v_full, n_q, n_kv, 1, kv_len, d, false);
        let mut projected = vec![0.0f32; h];
        linear(&attn_out, &attn.out_proj, 1, n_q * d, h, &mut projected);
        add_inplace(hidden, &projected);
    }

    fn ffn_residual(&self, ffn: &Ffn, hidden: &mut [f32], t: usize) {
        let h = self.hidden();
        let inter = self.config.intermediate_size;
        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            hidden,
            &ffn.ffn_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut g = vec![0.0f32; t * inter];
        let mut u = vec![0.0f32; t * inter];
        linear(&normed, &ffn.w1, t, h, inter, &mut g);
        linear(&normed, &ffn.w3, t, h, inter, &mut u);
        silu_inplace(&mut g);
        for (a, b) in g.iter_mut().zip(u.iter()) {
            *a *= *b;
        }
        let mut down = vec![0.0f32; t * h];
        linear(&g, &ffn.w2, t, inter, h, &mut down);
        add_inplace(hidden, &down);
    }
}

fn named<'a>(
    tensors: &'a HashMap<String, TensorF32>,
    name: &str,
) -> Result<&'a TensorF32, LoadError> {
    get_tensor(tensors, name)
}

fn rms(t: &TensorF32, dim: usize) -> Result<Rms, LoadError> {
    expect_shape(&t.shape, &[dim], "rmsnorm")?;
    Ok(Rms {
        weight: t.data.clone(),
    })
}

fn expect_shape(got: &[usize], want: &[usize], what: &str) -> Result<(), LoadError> {
    if got != want {
        Err(LoadError::Message(format!(
            "{what}: expected shape {want:?}, got {got:?}"
        )))
    } else {
        Ok(())
    }
}

/// RMSNorm over the last axis (`head_dim`) of a `[tokens, heads, dim]` (token-major) buffer.
fn rms_norm_heads(
    buf: &mut [f32],
    weight: &[f32],
    eps: f32,
    heads: usize,
    tokens: usize,
    dim: usize,
) {
    assert_eq!(buf.len(), tokens * heads * dim);
    let mut tmp = vec![0.0f32; dim];
    for t in 0..tokens {
        for h in 0..heads {
            let off = (t * heads + h) * dim;
            rms_norm(&buf[off..off + dim], weight, eps, &mut tmp);
            buf[off..off + dim].copy_from_slice(&tmp);
        }
    }
}

/// `[tokens, heads, dim]` → `[heads, tokens, dim]`.
fn transpose_heads_in(src: &[f32], heads: usize, tokens: usize, dim: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; heads * tokens * dim];
    for t in 0..tokens {
        for h in 0..heads {
            let s = (t * heads + h) * dim;
            let d = (h * tokens + t) * dim;
            dst[d..d + dim].copy_from_slice(&src[s..s + dim]);
        }
    }
    dst
}

/// `[heads, tokens, dim]` → `[tokens, heads, dim]`.
fn transpose_heads_out(src: &[f32], heads: usize, tokens: usize, dim: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; tokens * heads * dim];
    for t in 0..tokens {
        for h in 0..heads {
            let s = (h * tokens + t) * dim;
            let d = (t * heads + h) * dim;
            dst[d..d + dim].copy_from_slice(&src[s..s + dim]);
        }
    }
    dst
}

/// Eager GQA attention. `q` is `[n_q, q_len, dim]`, `k`/`v` are `[n_kv, kv_len, dim]`.
/// Returns `[n_q, q_len, dim]` (decode: already `[n_q, dim]` when q_len=1, same layout).
#[allow(clippy::too_many_arguments)]
fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    q_len: usize,
    kv_len: usize,
    dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = (dim as f32).sqrt().recip();
    let n_rep = n_q / n_kv;
    let mut out = vec![0.0f32; n_q * q_len * dim];
    for qh in 0..n_q {
        let kv_h = qh / n_rep;
        let mut scores = vec![0.0f32; q_len * kv_len];
        for t in 0..q_len {
            let qv = &q[(qh * q_len + t) * dim..(qh * q_len + t) * dim + dim];
            for s in 0..kv_len {
                if causal && s > t {
                    scores[t * kv_len + s] = f32::NEG_INFINITY;
                    continue;
                }
                let kv = &k[(kv_h * kv_len + s) * dim..(kv_h * kv_len + s) * dim + dim];
                let mut acc = 0.0f32;
                for d in 0..dim {
                    acc = qv[d].mul_add(kv[d], acc);
                }
                scores[t * kv_len + s] = acc * scale;
            }
        }
        softmax_last_axis(&mut scores, kv_len);
        for t in 0..q_len {
            let dst = &mut out[(qh * q_len + t) * dim..(qh * q_len + t) * dim + dim];
            dst.fill(0.0);
            for s in 0..kv_len {
                let w = scores[t * kv_len + s];
                if w == 0.0 {
                    continue;
                }
                let vv = &v[(kv_h * kv_len + s) * dim..(kv_h * kv_len + s) * dim + dim];
                for d in 0..dim {
                    dst[d] = w.mul_add(vv[d], dst[d]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::transpose_heads_in;

    #[test]
    fn transpose_roundtrip_shape() {
        let heads = 2;
        let tokens = 3;
        let dim = 4;
        let src: Vec<f32> = (0..heads * tokens * dim).map(|i| i as f32).collect();
        // src is token-major [t, h, d] if we fill that way — just check length.
        let t = transpose_heads_in(&src, heads, tokens, dim);
        assert_eq!(t.len(), src.len());
    }
}
