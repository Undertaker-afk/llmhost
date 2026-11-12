use serde::Deserialize;
use serde_json::Value;
use std::fmt;
use std::path::Path;

/// Rotary embedding parameters for one attention layer.
///
/// LFM3 uses a different `theta` (and sometimes a different
/// `partial_rotary_factor`) for full vs sliding attention; see
/// `rope_parameters` in the Hugging Face `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// RoPE base frequency (`rope_theta`).
    pub theta: f64,
    /// Fraction of the head dimension that is rotated. `1.0` means full RoPE.
    pub partial_rotary_factor: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
        }
    }
}

/// A layer in the config-driven model graph.
///
/// LFM2 and LFM3 are both sequences of these kinds. `Moe` is reserved for
/// a future LFM3 (or later) mixture-of-experts block; unknown type names
/// land in [`LayerKind::Other`] so a new family can be described before
/// its kernels exist.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    /// Causal depthwise short convolution. `kernel` is `conv_L_cache`
    /// (3 on LFM2, 7 on LFM3). The rolling conv state is `kernel - 1`.
    Conv { kernel: usize },
    /// Grouped-query (or compatible) attention.
    ///
    /// `sliding_window` is `Some` for LFM3 `sliding_attention` layers and
    /// `None` for full attention. `rope` is filled per layer from
    /// `rope_parameters` (LFM3) or the top-level `rope_theta` (LFM2).
    Attention {
        sliding_window: Option<usize>,
        rope: RopeParams,
    },
    /// Mixture-of-experts block (not yet implemented).
    Moe,
    /// Unrecognized or not-yet-implemented layer type, as written in the config.
    Other(String),
}

impl LayerKind {
    /// Classify a Hugging Face layer-type string, using the supplied conv
    /// kernel and the full / sliding rope parameters from the same config.
    #[must_use]
    pub fn from_parts(
        name: &str,
        conv_kernel: usize,
        sliding_window: Option<usize>,
        full_rope: RopeParams,
        sliding_rope: RopeParams,
    ) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "conv" | "convolution" | "short_conv" | "shortconv" | "short-conv" => Self::Conv {
                kernel: conv_kernel,
            },
            "attention" | "attn" | "gqa" | "mha" | "mqa" | "full_attention" | "full-attention" => {
                Self::Attention {
                    sliding_window: None,
                    rope: full_rope,
                }
            }
            "sliding_attention" | "sliding-attention" | "sliding" => Self::Attention {
                sliding_window,
                rope: sliding_rope,
            },
            "moe" | "mixture_of_experts" | "mixture-of-experts" => Self::Moe,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Error from parsing a Hugging Face `config.json`.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The contents were not valid JSON or missed the expected fields.
    Json(serde_json::Error),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read config: {err}"),
            Self::Json(err) => write!(f, "failed to parse config.json: {err}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

/// Subset of a Hugging Face `config.json` that the engine needs.
///
/// Unknown fields are ignored so LFM2 and LFM3 (and future revisions) can
/// share this parser. Vision configs (`lfm2_vl`, `lfm3_vl`) are read from
/// the nested `text_config` object.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// `model_type` from the (text) config, for example `"lfm2"` or `"lfm3"`.
    pub model_type: String,
    /// Hidden / model dimension.
    pub hidden_size: usize,
    /// Number of blocks in the stack.
    pub num_hidden_layers: usize,
    /// Per-layer kinds, in stack order. This is what builds the graph.
    pub layers: Vec<LayerKind>,
    /// Vocabulary size (`vocab_size`).
    pub vocab_size: usize,
    /// Query heads (`num_attention_heads` / `num_heads`).
    pub num_attention_heads: usize,
    /// Key/value heads (`num_key_value_heads`).
    pub num_key_value_heads: usize,
    /// Per-head dimension (`head_dim`, or `hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// SwiGLU intermediate size, after `block_auto_adjust_ff_dim` rounding.
    pub intermediate_size: usize,
    /// RMSNorm epsilon (`norm_eps`).
    pub norm_eps: f32,
    /// Whether conv / in_proj / out_proj have a bias (`conv_bias`).
    pub conv_bias: bool,
    /// Whether `lm_head` shares `embed_tokens` (`tie_word_embeddings`).
    pub tie_word_embeddings: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            hidden_size: 0,
            num_hidden_layers: 0,
            layers: Vec::new(),
            vocab_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            norm_eps: 1e-5,
            conv_bias: false,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct HfConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    hidden_size: usize,
    #[serde(default)]
    num_hidden_layers: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    block_types: Option<Vec<String>>,
    /// LFM2 (700M / 1.2B / 350M): attention layer indices; the rest are conv.
    #[serde(default, alias = "full_attention_idxs")]
    full_attn_idxs: Option<Vec<usize>>,
    /// Depthwise conv kernel / rolling cache length (`conv_L_cache`).
    #[serde(
        default,
        rename = "conv_L_cache",
        alias = "conv_l_cache",
        alias = "conv_kernel"
    )]
    conv_l_cache: Option<usize>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    /// Either a flat `{rope_theta, partial_rotary_factor}` object or a
    /// per-attention-type map (`full_attention` / `sliding_attention`).
    #[serde(default)]
    rope_parameters: Option<Value>,
    #[serde(default)]
    sliding_window: Option<usize>,
    /// Language-tower config on LFM2-VL / LFM3-VL.
    #[serde(default)]
    text_config: Option<Box<HfConfig>>,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    block_ff_dim: Option<usize>,
    #[serde(default)]
    block_auto_adjust_ff_dim: Option<bool>,
    #[serde(default)]
    block_ffn_dim_multiplier: Option<f64>,
    #[serde(default)]
    block_multiple_of: Option<usize>,
    #[serde(default)]
    norm_eps: Option<f32>,
    #[serde(default)]
    block_norm_eps: Option<f32>,
    #[serde(default)]
    conv_bias: Option<bool>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    tie_embedding: Option<bool>,
}

impl HfConfig {
    fn effective(self) -> Self {
        match self.text_config {
            Some(text) => *text,
            None => self,
        }
    }

    fn default_theta(&self) -> f64 {
        self.rope_theta.or(self.theta).unwrap_or(10_000.0)
    }

    fn default_partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor.unwrap_or(1.0)
    }

    fn rope_for(&self, kind: &str) -> RopeParams {
        let fallback = RopeParams {
            theta: self.default_theta(),
            partial_rotary_factor: self.default_partial_rotary_factor(),
        };
        let Some(value) = self.rope_parameters.as_ref() else {
            return fallback;
        };
        let Some(obj) = value.as_object() else {
            return fallback;
        };
        if let Some(entry) = obj.get(kind) {
            return rope_from_value(entry, fallback);
        }
        if obj.contains_key("rope_theta") || obj.contains_key("partial_rotary_factor") {
            return rope_from_value(value, fallback);
        }
        fallback
    }

    fn layer_type_names(&self) -> Vec<String> {
        if let Some(names) = self.layer_types.as_ref().or(self.block_types.as_ref()) {
            return names.clone();
        }
        if self.num_hidden_layers > 0 {
            if let Some(idxs) = self.full_attn_idxs.as_ref() {
                return (0..self.num_hidden_layers)
                    .map(|i| {
                        if idxs.contains(&i) {
                            "full_attention".to_string()
                        } else {
                            "conv".to_string()
                        }
                    })
                    .collect();
            }
        }
        Vec::new()
    }
}

fn rope_from_value(value: &Value, fallback: RopeParams) -> RopeParams {
    let theta = value
        .get("rope_theta")
        .or_else(|| value.get("theta"))
        .and_then(json_f64)
        .unwrap_or(fallback.theta);
    let partial_rotary_factor = value
        .get("partial_rotary_factor")
        .and_then(json_f32)
        .unwrap_or(fallback.partial_rotary_factor);
    RopeParams {
        theta,
        partial_rotary_factor,
    }
}

fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .or_else(|| value.as_u64().map(|v| v as f64))
}

fn json_f32(value: &Value) -> Option<f32> {
    json_f64(value).map(|v| v as f32)
}

impl ModelConfig {
    /// Parse a `config.json` document.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let raw: HfConfig = serde_json::from_str(json)?;
        Ok(Self::from_hf(raw.effective()))
    }

    /// Read and parse a `config.json` file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    fn from_hf(raw: HfConfig) -> Self {
        let type_names = raw.layer_type_names();
        let conv_kernel = raw.conv_l_cache.unwrap_or(0);
        let sliding_window = raw.sliding_window;
        let full_rope = raw.rope_for("full_attention");
        let sliding_rope = raw.rope_for("sliding_attention");
        let mut layers: Vec<LayerKind> = type_names
            .iter()
            .map(|name| {
                LayerKind::from_parts(name, conv_kernel, sliding_window, full_rope, sliding_rope)
            })
            .collect();
        if layers.is_empty() && raw.num_hidden_layers > 0 {
            layers = vec![LayerKind::Other("unspecified".to_string()); raw.num_hidden_layers];
        }
        let num_hidden_layers = if raw.num_hidden_layers > 0 {
            raw.num_hidden_layers
        } else {
            layers.len()
        };
        let num_attention_heads = raw.num_attention_heads.or(raw.num_heads).unwrap_or(0);
        let num_key_value_heads = if raw.num_key_value_heads > 0 {
            raw.num_key_value_heads
        } else {
            num_attention_heads
        };
        let head_dim = raw.head_dim.unwrap_or_else(|| {
            raw.hidden_size
                .checked_div(num_attention_heads)
                .unwrap_or(0)
        });
        let intermediate_size = compute_intermediate_size(&raw);
        let norm_eps = raw.norm_eps.or(raw.block_norm_eps).unwrap_or(1e-5);
        let tie_word_embeddings = raw
            .tie_word_embeddings
            .or(raw.tie_embedding)
            .unwrap_or(true);
        Self {
            model_type: raw.model_type,
            hidden_size: raw.hidden_size,
            num_hidden_layers,
            layers,
            vocab_size: raw.vocab_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            norm_eps,
            conv_bias: raw.conv_bias.unwrap_or(false),
            tie_word_embeddings,
        }
    }
}

/// Apply LFM2 `block_auto_adjust_ff_dim` (same arithmetic as `Lfm2MLP`).
fn compute_intermediate_size(raw: &HfConfig) -> usize {
    let mut size = raw.intermediate_size.or(raw.block_ff_dim).unwrap_or(0);
    if size == 0 {
        return 0;
    }
    if raw.block_auto_adjust_ff_dim.unwrap_or(false) {
        size = (2 * size) / 3;
        if let Some(mult) = raw.block_ffn_dim_multiplier {
            size = (mult * size as f64) as usize;
        }
        let multiple = raw.block_multiple_of.unwrap_or(256);
        if multiple > 0 {
            size = multiple * size.div_ceil(multiple);
        }
    }
    size
}

#[cfg(test)]
mod tests {
    use super::{LayerKind, ModelConfig, RopeParams};

    fn full_rope(theta: f64, partial: f32) -> RopeParams {
        RopeParams {
            theta,
            partial_rotary_factor: partial,
        }
    }

    fn is_conv(kind: &LayerKind, kernel: usize) -> bool {
        matches!(kind, LayerKind::Conv { kernel: k } if *k == kernel)
    }

    fn is_full(kind: &LayerKind, rope: RopeParams) -> bool {
        matches!(
            kind,
            LayerKind::Attention {
                sliding_window: None,
                rope: r,
            } if *r == rope
        )
    }

    fn is_sliding(kind: &LayerKind, window: usize, rope: RopeParams) -> bool {
        matches!(
            kind,
            LayerKind::Attention {
                sliding_window: Some(w),
                rope: r,
            } if *w == window && *r == rope
        )
    }

    #[test]
    fn parses_layer_types_and_moe() {
        let json = r#"{
            "model_type": "lfm2",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "conv_L_cache": 3,
            "rope_theta": 1000000.0,
            "layer_types": ["conv", "short_conv", "attention", "moe"],
            "unused_field": true
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        let rope = full_rope(1_000_000.0, 1.0);
        assert_eq!(cfg.model_type, "lfm2");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert!(is_conv(&cfg.layers[0], 3));
        assert!(is_conv(&cfg.layers[1], 3));
        assert!(is_full(&cfg.layers[2], rope));
        assert_eq!(cfg.layers[3], LayerKind::Moe);
    }

    #[test]
    fn accepts_block_types_alias() {
        let json = r#"{"model_type":"lfm3","conv_L_cache":7,"block_types":["gqa","conv"]}"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm3");
        assert_eq!(cfg.num_hidden_layers, 2);
        assert!(is_full(&cfg.layers[0], RopeParams::default()));
        assert!(is_conv(&cfg.layers[1], 7));
    }

    #[test]
    fn fills_unspecified_layers_from_count() {
        let json = r#"{"num_hidden_layers": 3, "hidden_size": 16}"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.layers.len(), 3);
        assert!(cfg
            .layers
            .iter()
            .all(|k| *k == LayerKind::Other("unspecified".to_string())));
    }

    #[test]
    fn parses_lfm2_700m_shaped_full_attn_idxs() {
        // Shaped like LiquidAI/LFM2-700M: no layer_types, attention at
        // full_attn_idxs, conv_L_cache = 3, a single rope_theta.
        let json = r#"{
            "architectures": ["Lfm2ForCausalLM"],
            "conv_L_cache": 3,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "hidden_size": 1536,
            "max_position_embeddings": 128000,
            "model_type": "lfm2",
            "num_attention_heads": 24,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rope_theta": 1000000.0,
            "vocab_size": 65536
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm2");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.layers.len(), 16);
        let rope = full_rope(1_000_000.0, 1.0);
        let attn = [2usize, 5, 8, 10, 12, 14];
        for (i, kind) in cfg.layers.iter().enumerate() {
            if attn.contains(&i) {
                assert!(is_full(kind, rope), "layer {i} should be full attention");
            } else {
                assert!(is_conv(kind, 3), "layer {i} should be conv");
            }
        }
    }

    #[test]
    fn parses_lfm3_shaped_sliding_and_per_layer_rope() {
        // Shaped like LiquidAI/LFM3-1.7B: layer_types with sliding_attention,
        // conv_L_cache = 7, sliding_window = 256, per-type rope_parameters.
        let json = r#"{
            "architectures": ["Lfm3ForCausalLM"],
            "conv_L_cache": 7,
            "hidden_size": 2048,
            "layer_types": [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv",
                "full_attention",
                "conv"
            ],
            "model_type": "lfm3",
            "num_hidden_layers": 8,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 1000000.0,
                    "rope_type": "default"
                },
                "sliding_attention": {
                    "rope_theta": 10000,
                    "rope_type": "default"
                }
            },
            "sliding_window": 256,
            "vocab_size": 128000
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm3");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.layers.len(), 8);
        let full = full_rope(1_000_000.0, 0.5);
        let sliding = full_rope(10_000.0, 1.0);
        assert!(is_conv(&cfg.layers[0], 7));
        assert!(is_conv(&cfg.layers[1], 7));
        assert!(is_full(&cfg.layers[2], full));
        assert!(is_conv(&cfg.layers[3], 7));
        assert!(is_sliding(&cfg.layers[4], 256, sliding));
        assert!(is_conv(&cfg.layers[5], 7));
        assert!(is_full(&cfg.layers[6], full));
        assert!(is_conv(&cfg.layers[7], 7));
    }

    #[test]
    fn computes_lfm2_700m_intermediate_size() {
        let json = r#"{
            "model_type": "lfm2",
            "hidden_size": 1536,
            "num_hidden_layers": 16,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "block_ff_dim": 10240,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "vocab_size": 65536,
            "norm_eps": 1e-05,
            "conv_L_cache": 3,
            "rope_theta": 1000000.0
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.intermediate_size, 6912);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.vocab_size, 65536);
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn parses_vl_text_config() {
        let json = r#"{
            "model_type": "lfm3_vl",
            "text_config": {
                "model_type": "lfm3",
                "hidden_size": 2048,
                "conv_L_cache": 7,
                "layer_types": ["conv", "sliding_attention"],
                "rope_parameters": {
                    "sliding_attention": { "rope_theta": 10000 }
                },
                "sliding_window": 256,
                "num_hidden_layers": 2
            }
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm3");
        assert!(is_conv(&cfg.layers[0], 7));
        assert!(is_sliding(&cfg.layers[1], 256, full_rope(10_000.0, 1.0)));
    }
}
