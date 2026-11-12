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
use crate::weights::{apply_pair, Linear, OutputHead};
use llm_core::{LayerKind, ModelConfig};
use llm_cpu::{
    add_inplace, apply_rope_in_place, argmax, causal_dwconv_prefill, causal_dwconv_update, cos_sin,
    inv_freq, mul, prefault_f32, rms_norm, silu_inplace, softmax_last_axis,
};
use llm_quant::WeightFormat;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Intermediate residual (and other) activations captured during a prefill.
#[derive(Clone, Debug, Default)]
pub struct LayerTraces {
    pub embed: Vec<f32>,
    pub layers: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Per-class nanoseconds accumulated over a decode run.
#[derive(Clone, Debug, Default)]
pub struct DecodeProfile {
    /// Q4/Q8/fp32 projections (includes the pool barrier).
    pub gemv_ns: u64,
    /// Attention scores, softmax, and value mix (not QKV / out_proj).
    pub attn_ns: u64,
    /// Depthwise conv and B,C,x gates (not in/out projections).
    pub conv_ns: u64,
    /// RMSNorm (including q/k head norms).
    pub norm_ns: u64,
    /// Everything else (embed gather, residual adds, RoPE, silu, ...).
    pub other_ns: u64,
    /// Number of tokens accumulated in this profile.
    pub tokens: u64,
}

impl DecodeProfile {
    /// Share of each class as a fraction of the summed class times.
    #[must_use]
    pub fn fractions(&self) -> [(&'static str, u64, f64); 5] {
        let total = self.gemv_ns + self.attn_ns + self.conv_ns + self.norm_ns + self.other_ns;
        let tot = total.max(1) as f64;
        [
            ("gemv", self.gemv_ns, self.gemv_ns as f64 / tot),
            ("attn", self.attn_ns, self.attn_ns as f64 / tot),
            ("conv", self.conv_ns, self.conv_ns as f64 / tot),
            ("norm", self.norm_ns, self.norm_ns as f64 / tot),
            ("other", self.other_ns, self.other_ns as f64 / tot),
        ]
    }
}

/// Load options for a Q4 body (and an optional Q8_0 tied head).
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4LoadOpts {
    /// Quantize the tied LM head to Q8_0 instead of Q4_32.
    pub q8_head: bool,
}

/// Reusable buffers for allocation-free incremental decode.
pub struct DecodeScratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    bcx: Vec<f32>,
    tmp_h: Vec<f32>,
    tmp_h2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    scores: Vec<f32>,
    attn_out: Vec<f32>,
    projected: Vec<f32>,
    /// Logits for the last decoded token (length `vocab`).
    pub logits: Vec<f32>,
    /// Accumulated profile; written when `collect_profile` is set.
    pub profile: DecodeProfile,
    pub collect_profile: bool,
}

impl DecodeScratch {
    /// Allocate buffers large enough for `model` and a KV cap of `kv_cap`.
    #[must_use]
    pub fn for_model(model: &Lfm2Model, kv_cap: usize) -> Self {
        let h = model.hidden();
        let v = model.vocab();
        let n_q = model.config.num_attention_heads;
        let n_kv = model.config.num_key_value_heads;
        let d = model.config.head_dim;
        let inter = model.config.intermediate_size;
        Self {
            hidden: vec![0.0; h],
            normed: vec![0.0; h],
            bcx: vec![0.0; 3 * h],
            tmp_h: vec![0.0; h],
            tmp_h2: vec![0.0; h],
            q: vec![0.0; n_q * d],
            k: vec![0.0; n_kv * d],
            v: vec![0.0; n_kv * d],
            cos: vec![0.0; d],
            sin: vec![0.0; d],
            g: vec![0.0; inter],
            u: vec![0.0; inter],
            scores: vec![0.0; n_q * kv_cap],
            attn_out: vec![0.0; n_q * d],
            projected: vec![0.0; h],
            logits: vec![0.0; v],
            profile: DecodeProfile::default(),
            collect_profile: false,
        }
    }
}

#[derive(Clone, Copy)]
enum Prof {
    Gemv,
    Attn,
    Conv,
    Norm,
    Other,
}

fn tick(scratch: &mut DecodeScratch, class: Prof, start: Instant) {
    if !scratch.collect_profile {
        return;
    }
    let ns = start.elapsed().as_nanos() as u64;
    match class {
        Prof::Gemv => scratch.profile.gemv_ns += ns,
        Prof::Attn => scratch.profile.attn_ns += ns,
        Prof::Conv => scratch.profile.conv_ns += ns,
        Prof::Norm => scratch.profile.norm_ns += ns,
        Prof::Other => scratch.profile.other_ns += ns,
    }
}

struct Rms {
    weight: Vec<f32>,
}

struct Ffn {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    ffn_norm: Rms,
}

struct ConvBlock {
    operator_norm: Rms,
    in_proj: Linear,
    conv_w: Vec<f32>,
    out_proj: Linear,
    ffn: Ffn,
    kernel: usize,
}

struct AttnBlock {
    operator_norm: Rms,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
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
    /// Per-position key mask (`true` = attend). Prompt pads may be false;
    /// newly decoded tokens are always valid, matching HF `generate`.
    key_valid: Vec<bool>,
}

impl Lfm2Cache {
    /// Reserved time capacity of the KV cache.
    #[must_use]
    pub fn kv_capacity(&self) -> usize {
        self.kv_cap
    }
}

/// LFM2 model. Linear layers are fp32 or Q4_32 depending on how it was loaded.
pub struct Lfm2Model {
    pub config: ModelConfig,
    pub format: WeightFormat,
    embed: Vec<f32>,
    head: OutputHead,
    embedding_norm: Rms,
    blocks: Vec<Block>,
    inv_freq: Vec<f32>,
}

impl Lfm2Model {
    /// Load from a snapshot directory that has `config.json` + `model.safetensors`.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        Self::load_dir_with(dir, WeightFormat::F32)
    }

    /// Load a snapshot, quantizing every linear layer to `format` at load time.
    /// Embeddings, RMSNorms, and the depthwise conv kernel stay fp32. The tied
    /// LM head uses the same format as the other linears (Q4_32 or fp32).
    pub fn load_dir_with(dir: impl AsRef<Path>, format: WeightFormat) -> Result<Self, LoadError> {
        Self::load_dir_split(dir, format, format)
    }

    /// Load a snapshot with a body format and a possibly different tied head.
    pub fn load_dir_split(
        dir: impl AsRef<Path>,
        body: WeightFormat,
        head: WeightFormat,
    ) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = load_config(dir)?;
        let tensors = load_safetensors_f32(dir.join("model.safetensors"))?;
        Self::from_tensors(config, &tensors, body, head)
    }

    /// Locate the LFM2-700M snapshot in the local Hub cache (no download) and load it.
    pub fn load_lfm2_700m() -> Result<Self, LoadError> {
        let dir = find_hf_snapshot("LiquidAI/LFM2-700M", "86f49fc9")?;
        Self::load_dir(dir)
    }

    /// LFM2-700M with Q4_32 linear layers (and a Q4 tied head).
    pub fn load_lfm2_700m_q4() -> Result<Self, LoadError> {
        Self::load_lfm2_700m_q4_opts(Q4LoadOpts { q8_head: false })
    }

    /// LFM2-700M with Q4_32 body linears and an optional Q8_0 tied head.
    pub fn load_lfm2_700m_q4_opts(opts: Q4LoadOpts) -> Result<Self, LoadError> {
        let dir = find_hf_snapshot("LiquidAI/LFM2-700M", "86f49fc9")?;
        let head = if opts.q8_head {
            WeightFormat::Q8_0
        } else {
            WeightFormat::Q4_32
        };
        Self::load_dir_split(dir, WeightFormat::Q4_32, head)
    }

    fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, TensorF32>,
        format: WeightFormat,
        head_format: WeightFormat,
    ) -> Result<Self, LoadError> {
        let h = config.hidden_size;
        let embed = get_tensor(tensors, "model.embed_tokens.weight")?;
        expect_shape(&embed.shape, &[config.vocab_size, h], "embed_tokens")?;
        let embedding_norm = rms(get_tensor(tensors, "model.embedding_norm.weight")?, h)?;
        let mut blocks = Vec::with_capacity(config.layers.len());
        for (i, kind) in config.layers.iter().enumerate() {
            let prefix = format!("model.layers.{i}");
            let ffn = Ffn {
                w1: linear_named(tensors, &format!("{prefix}.feed_forward.w1.weight"), format)?,
                w2: linear_named(tensors, &format!("{prefix}.feed_forward.w2.weight"), format)?,
                w3: linear_named(tensors, &format!("{prefix}.feed_forward.w3.weight"), format)?,
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
                        in_proj: linear_named(
                            tensors,
                            &format!("{prefix}.conv.in_proj.weight"),
                            format,
                        )?,
                        conv_w: squeeze_conv_weight(conv)?,
                        out_proj: linear_named(
                            tensors,
                            &format!("{prefix}.conv.out_proj.weight"),
                            format,
                        )?,
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
                        q_proj: linear_named(
                            tensors,
                            &format!("{prefix}.self_attn.q_proj.weight"),
                            format,
                        )?,
                        k_proj: linear_named(
                            tensors,
                            &format!("{prefix}.self_attn.k_proj.weight"),
                            format,
                        )?,
                        v_proj: linear_named(
                            tensors,
                            &format!("{prefix}.self_attn.v_proj.weight"),
                            format,
                        )?,
                        out_proj: linear_named(
                            tensors,
                            &format!("{prefix}.self_attn.out_proj.weight"),
                            format,
                        )?,
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
        let head = OutputHead::from_embed(&embed.data, config.vocab_size, h, head_format)?;
        let model = Self {
            config,
            format,
            embed: embed.data.clone(),
            head,
            embedding_norm,
            blocks,
            inv_freq: inv,
        };
        model.prefault();
        Ok(model)
    }

    /// Touch weight pages so the first decode/prefill is not dominated by faults.
    pub fn prefault(&self) {
        prefault_f32(&self.embed);
        prefault_f32(&self.embedding_norm.weight);
        prefault_f32(&self.inv_freq);
        self.head.prefault();
        for block in &self.blocks {
            match block {
                Block::Conv(c) => {
                    prefault_f32(&c.operator_norm.weight);
                    c.in_proj.prefault();
                    prefault_f32(&c.conv_w);
                    c.out_proj.prefault();
                    prefault_ffn(&c.ffn);
                }
                Block::Attn(a) => {
                    prefault_f32(&a.operator_norm.weight);
                    a.q_proj.prefault();
                    a.k_proj.prefault();
                    a.v_proj.prefault();
                    a.out_proj.prefault();
                    prefault_f32(&a.q_norm.weight);
                    prefault_f32(&a.k_norm.weight);
                    prefault_ffn(&a.ffn);
                }
            }
        }
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
        traces: Option<&mut LayerTraces>,
        kv_reserve: usize,
    ) -> Lfm2Cache {
        self.prefill_masked(tokens, traces, kv_reserve, None)
    }

    /// Prefill with an optional pad id. Positions equal to `pad_token_id` are
    /// treated like Hugging Face `generate` when it is given that pad id:
    /// conv inputs at those positions are zeroed and attention does not
    /// attend to them. Newly decoded tokens stay visible even if they equal
    /// the pad id. Layer traces for an unpadded prefill should keep
    /// `pad_token_id` as `None`.
    pub fn prefill_masked(
        &self,
        tokens: &[u32],
        mut traces: Option<&mut LayerTraces>,
        kv_reserve: usize,
        pad_token_id: Option<u32>,
    ) -> Lfm2Cache {
        let t = tokens.len();
        let h = self.hidden();
        let v = self.vocab();
        let mut hidden = vec![0.0f32; t * h];
        for (i, &id) in tokens.iter().enumerate() {
            let id = id as usize;
            hidden[i * h..(i + 1) * h].copy_from_slice(&self.embed[id * h..(id + 1) * h]);
        }
        let key_valid: Vec<bool> = tokens.iter().map(|&id| pad_token_id != Some(id)).collect();
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
            key_valid: Vec::new(),
        };

        for block in &self.blocks {
            match block {
                Block::Conv(conv) => {
                    let mut state = vec![0.0f32; h * conv.kernel.saturating_sub(1)];
                    self.conv_prefill(conv, &mut hidden, t, &mut state, &key_valid);
                    self.ffn_residual(&conv.ffn, &mut hidden, t);
                    cache.conv.push(Some(state));
                    cache.k.push(None);
                    cache.v.push(None);
                }
                Block::Attn(attn) => {
                    let (k, v) = self.attn_prefill(attn, &mut hidden, t, 0, cap, &key_valid);
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
        cache.key_valid = key_valid;

        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            &hidden,
            &self.embedding_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        let mut logits = vec![0.0f32; t * v];
        self.head.apply(&normed, t, &mut logits);
        if let Some(tr) = traces.as_mut() {
            tr.final_norm = normed;
            tr.logits = logits;
        }
        cache
    }

    /// Decode one new token, appending to `cache`. Returns vocab logits for that step.
    pub fn decode_one(&self, token: u32, cache: &mut Lfm2Cache) -> Vec<f32> {
        let mut scratch = DecodeScratch::for_model(self, cache.kv_cap);
        self.decode_one_in(token, cache, &mut scratch).to_vec()
    }

    /// Decode into reusable `scratch`. The returned slice is `scratch.logits`.
    pub fn decode_one_in<'a>(
        &self,
        token: u32,
        cache: &mut Lfm2Cache,
        scratch: &'a mut DecodeScratch,
    ) -> &'a [f32] {
        let h = self.hidden();
        let id = token as usize;
        scratch
            .hidden
            .copy_from_slice(&self.embed[id * h..(id + 1) * h]);
        let pos = cache.seq_len;
        // Generated tokens are never treated as padding.
        cache.key_valid.push(true);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Conv(conv) => {
                    let state = cache.conv[layer_idx]
                        .as_mut()
                        .expect("conv state for conv layer");
                    self.conv_update(conv, state, scratch);
                    self.ffn_residual_in(&conv.ffn, scratch);
                }
                Block::Attn(attn) => {
                    self.attn_decode(
                        attn,
                        pos,
                        cache.k[layer_idx].as_mut().unwrap(),
                        cache.v[layer_idx].as_mut().unwrap(),
                        cache.kv_cap,
                        &cache.key_valid,
                        scratch,
                    );
                    self.ffn_residual_in(&attn.ffn, scratch);
                }
            }
        }
        cache.seq_len += 1;
        let t0 = Instant::now();
        rms_norm(
            &scratch.hidden,
            &self.embedding_norm.weight,
            self.config.norm_eps,
            &mut scratch.normed,
        );
        tick(scratch, Prof::Norm, t0);
        let t0 = Instant::now();
        self.head.apply(&scratch.normed, 1, &mut scratch.logits);
        tick(scratch, Prof::Gemv, t0);
        if scratch.collect_profile {
            scratch.profile.tokens += 1;
        }
        &scratch.logits
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
        let mut scratch = DecodeScratch::for_model(self, cache.kv_cap);
        for _ in 0..steps {
            let logits = self.decode_one_in(next, cache, &mut scratch);
            ids.push(next);
            let chosen = argmax(logits) as u32;
            all_logits.push(logits.to_vec());
            next = chosen;
        }
        (ids, all_logits)
    }

    fn conv_prefill(
        &self,
        conv: &ConvBlock,
        hidden: &mut [f32],
        t: usize,
        state: &mut [f32],
        key_valid: &[bool],
    ) {
        let h = self.hidden();
        let mut normed = vec![0.0f32; t * h];
        rms_norm(
            hidden,
            &conv.operator_norm.weight,
            self.config.norm_eps,
            &mut normed,
        );
        // Match Lfm2ShortConv: zero padding positions before in_proj.
        if key_valid.len() == t {
            for (i, valid) in key_valid.iter().enumerate() {
                if !valid {
                    normed[i * h..(i + 1) * h].fill(0.0);
                }
            }
        }
        let mut bcx = vec![0.0f32; t * 3 * h];
        conv.in_proj.apply(&normed, t, &mut bcx);
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
        conv.out_proj.apply(&y, t, &mut projected);
        // residual: hidden = hidden + conv(norm(hidden))
        add_inplace(hidden, &projected);
    }

    fn conv_update(&self, conv: &ConvBlock, state: &mut [f32], scratch: &mut DecodeScratch) {
        let h = self.hidden();
        let t0 = Instant::now();
        rms_norm(
            &scratch.hidden,
            &conv.operator_norm.weight,
            self.config.norm_eps,
            &mut scratch.normed,
        );
        tick(scratch, Prof::Norm, t0);
        let t0 = Instant::now();
        conv.in_proj.apply(&scratch.normed, 1, &mut scratch.bcx);
        tick(scratch, Prof::Gemv, t0);
        let t0 = Instant::now();
        // u = B * x, then depthwise conv, then y = C * conv(u). tmp_h/tmp_h2 hold u/y.
        for i in 0..h {
            scratch.tmp_h[i] = scratch.bcx[i] * scratch.bcx[2 * h + i];
        }
        causal_dwconv_update(
            &scratch.tmp_h,
            &conv.conv_w,
            h,
            conv.kernel,
            &mut scratch.tmp_h2,
            state,
        );
        for i in 0..h {
            scratch.tmp_h[i] = scratch.bcx[h + i] * scratch.tmp_h2[i];
        }
        tick(scratch, Prof::Conv, t0);
        let t0 = Instant::now();
        conv.out_proj
            .apply(&scratch.tmp_h, 1, &mut scratch.projected);
        tick(scratch, Prof::Gemv, t0);
        add_inplace(&mut scratch.hidden, &scratch.projected);
    }

    fn attn_prefill(
        &self,
        attn: &AttnBlock,
        hidden: &mut [f32],
        t: usize,
        start_pos: usize,
        kv_cap: usize,
        key_valid: &[bool],
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
        attn.q_proj.apply(&normed, t, &mut q);
        attn.k_proj.apply(&normed, t, &mut k);
        attn.v_proj.apply(&normed, t, &mut v);
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

        let attn_out = attention_forward(
            &q_htd,
            &k_htd,
            &v_htd,
            n_q,
            n_kv,
            t,
            t,
            d,
            true,
            Some(key_valid),
        );
        let flat = transpose_heads_out(&attn_out, n_q, t, d);
        let mut projected = vec![0.0f32; t * h];
        attn.out_proj.apply(&flat, t, &mut projected);
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

    #[allow(clippy::too_many_arguments)]
    fn attn_decode(
        &self,
        attn: &AttnBlock,
        pos: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        kv_cap: usize,
        key_valid: &[bool],
        scratch: &mut DecodeScratch,
    ) {
        let h = self.hidden();
        let n_q = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let d = self.config.head_dim;
        let kv_len = pos + 1;
        assert!(kv_len <= kv_cap, "KV cache overflow: {kv_len} > {kv_cap}");

        let t0 = Instant::now();
        rms_norm(
            &scratch.hidden,
            &attn.operator_norm.weight,
            self.config.norm_eps,
            &mut scratch.normed,
        );
        tick(scratch, Prof::Norm, t0);
        let t0 = Instant::now();
        attn.q_proj.apply(&scratch.normed, 1, &mut scratch.q);
        attn.k_proj.apply(&scratch.normed, 1, &mut scratch.k);
        attn.v_proj.apply(&scratch.normed, 1, &mut scratch.v);
        tick(scratch, Prof::Gemv, t0);
        let t0 = Instant::now();
        rms_norm_heads(
            &mut scratch.q,
            &attn.q_norm.weight,
            self.config.norm_eps,
            n_q,
            1,
            d,
        );
        rms_norm_heads(
            &mut scratch.k,
            &attn.k_norm.weight,
            self.config.norm_eps,
            n_kv,
            1,
            d,
        );
        tick(scratch, Prof::Norm, t0);

        let t0 = Instant::now();
        cos_sin(
            &self.inv_freq,
            pos,
            1,
            d,
            &mut scratch.cos,
            &mut scratch.sin,
        );
        apply_rope_in_place(&mut scratch.q, &scratch.cos, &scratch.sin, n_q, 1, d);
        apply_rope_in_place(&mut scratch.k, &scratch.cos, &scratch.sin, n_kv, 1, d);
        tick(scratch, Prof::Other, t0);

        for head in 0..n_kv {
            let dst = (head * kv_cap + pos) * d;
            k_cache[dst..dst + d].copy_from_slice(&scratch.k[head * d..(head + 1) * d]);
            v_cache[dst..dst + d].copy_from_slice(&scratch.v[head * d..(head + 1) * d]);
        }

        let need = n_q * kv_len;
        if scratch.scores.len() < need {
            scratch.scores.resize(need, 0.0);
        }
        let t0 = Instant::now();
        attention_decode_cached(
            &scratch.q,
            k_cache,
            v_cache,
            n_q,
            n_kv,
            kv_len,
            kv_cap,
            d,
            key_valid,
            &mut scratch.scores[..need],
            &mut scratch.attn_out,
        );
        tick(scratch, Prof::Attn, t0);
        let t0 = Instant::now();
        attn.out_proj
            .apply(&scratch.attn_out, 1, &mut scratch.projected);
        tick(scratch, Prof::Gemv, t0);
        add_inplace(&mut scratch.hidden, &scratch.projected);
        let _ = h;
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
        apply_pair(&ffn.w1, &ffn.w3, &normed, t, &mut g, &mut u);
        silu_inplace(&mut g);
        for (a, b) in g.iter_mut().zip(u.iter()) {
            *a *= *b;
        }
        // Down-projection is added in f64 so a large residual (chat_basic L09)
        // does not pick up an extra f32 rounding step.
        ffn.w2.apply_add(&g, t, hidden);
    }

    fn ffn_residual_in(&self, ffn: &Ffn, scratch: &mut DecodeScratch) {
        let t0 = Instant::now();
        rms_norm(
            &scratch.hidden,
            &ffn.ffn_norm.weight,
            self.config.norm_eps,
            &mut scratch.normed,
        );
        tick(scratch, Prof::Norm, t0);
        let t0 = Instant::now();
        apply_pair(
            &ffn.w1,
            &ffn.w3,
            &scratch.normed,
            1,
            &mut scratch.g,
            &mut scratch.u,
        );
        tick(scratch, Prof::Gemv, t0);
        let t0 = Instant::now();
        silu_inplace(&mut scratch.g);
        for (a, b) in scratch.g.iter_mut().zip(scratch.u.iter()) {
            *a *= *b;
        }
        tick(scratch, Prof::Other, t0);
        let t0 = Instant::now();
        ffn.w2.apply_add(&scratch.g, 1, &mut scratch.hidden);
        tick(scratch, Prof::Gemv, t0);
    }
}

fn prefault_ffn(ffn: &Ffn) {
    ffn.w1.prefault();
    ffn.w2.prefault();
    ffn.w3.prefault();
    prefault_f32(&ffn.ffn_norm.weight);
}

fn named<'a>(
    tensors: &'a HashMap<String, TensorF32>,
    name: &str,
) -> Result<&'a TensorF32, LoadError> {
    get_tensor(tensors, name)
}

fn linear_named(
    tensors: &HashMap<String, TensorF32>,
    name: &str,
    format: WeightFormat,
) -> Result<Linear, LoadError> {
    let t = named(tensors, name)?;
    if t.shape.len() != 2 {
        return Err(LoadError::Message(format!(
            "{name}: expected a 2-D weight, got {:?}",
            t.shape
        )));
    }
    Linear::from_f32(t.data.clone(), t.shape[0], t.shape[1], format)
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

/// Decode-time attention against a strided KV cache `[n_kv, cap, dim]`.
#[allow(clippy::too_many_arguments)]
fn attention_decode_cached(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_q: usize,
    n_kv: usize,
    kv_len: usize,
    kv_cap: usize,
    dim: usize,
    key_valid: &[bool],
    scores: &mut [f32],
    out: &mut [f32],
) {
    let scale = (dim as f32).sqrt().recip();
    let n_rep = n_q / n_kv;
    assert_eq!(q.len(), n_q * dim);
    assert_eq!(scores.len(), n_q * kv_len);
    assert_eq!(out.len(), n_q * dim);
    out.fill(0.0);
    for qh in 0..n_q {
        let kv_h = qh / n_rep;
        let qv = &q[qh * dim..(qh + 1) * dim];
        let row = &mut scores[qh * kv_len..(qh + 1) * kv_len];
        for s in 0..kv_len {
            if key_valid.get(s) == Some(&false) {
                row[s] = f32::NEG_INFINITY;
                continue;
            }
            let kv = &k_cache[(kv_h * kv_cap + s) * dim..(kv_h * kv_cap + s) * dim + dim];
            let mut acc = 0.0f32;
            for d in 0..dim {
                acc = qv[d].mul_add(kv[d], acc);
            }
            row[s] = acc * scale;
        }
        softmax_last_axis(row, kv_len);
        let dst = &mut out[qh * dim..(qh + 1) * dim];
        for s in 0..kv_len {
            let w = row[s];
            if w == 0.0 {
                continue;
            }
            let vv = &v_cache[(kv_h * kv_cap + s) * dim..(kv_h * kv_cap + s) * dim + dim];
            for d in 0..dim {
                dst[d] = w.mul_add(vv[d], dst[d]);
            }
        }
    }
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
    key_valid: Option<&[bool]>,
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
                if (causal && s > t) || key_valid.is_some_and(|m| !m[s]) {
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
