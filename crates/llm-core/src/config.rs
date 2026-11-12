use serde::Deserialize;
use std::fmt;
use std::path::Path;

/// A layer in the config-driven model graph.
///
/// `Other` holds a type name that is not wired yet (for example an LFM3 MoE
/// block) so a new family can be described before its kernels exist.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Causal depthwise short convolution (LFM2/LFM3 conv blocks).
    Conv,
    /// Grouped-query (or compatible) attention block.
    Attention,
    /// Unrecognized or not-yet-implemented layer type, as written in the config.
    Other(String),
}

impl LayerKind {
    /// Map a Hugging Face layer-type string to a [`LayerKind`].
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "conv" | "convolution" | "short_conv" | "shortconv" | "short-conv" => Self::Conv,
            "attention" | "attn" | "gqa" | "mha" | "mqa" | "full_attention" | "full-attention" => {
                Self::Attention
            }
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
/// share this parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    /// `model_type` from the config, for example `"lfm2"` or `"lfm3"`.
    pub model_type: String,
    /// Hidden / model dimension.
    pub hidden_size: usize,
    /// Number of blocks in the stack.
    pub num_hidden_layers: usize,
    /// Per-layer kinds, in stack order. This is what builds the graph.
    pub layers: Vec<LayerKind>,
}

#[derive(Debug, Deserialize)]
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
}

impl ModelConfig {
    /// Parse a `config.json` document.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let raw: HfConfig = serde_json::from_str(json)?;
        let type_names = raw.layer_types.or(raw.block_types).unwrap_or_default();
        let mut layers: Vec<LayerKind> = type_names
            .iter()
            .map(|name| LayerKind::from_name(name))
            .collect();
        if layers.is_empty() && raw.num_hidden_layers > 0 {
            layers = vec![LayerKind::Other("unspecified".to_string()); raw.num_hidden_layers];
        }
        let num_hidden_layers = if raw.num_hidden_layers > 0 {
            raw.num_hidden_layers
        } else {
            layers.len()
        };
        Ok(Self {
            model_type: raw.model_type,
            hidden_size: raw.hidden_size,
            num_hidden_layers,
            layers,
        })
    }

    /// Read and parse a `config.json` file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerKind, ModelConfig};

    #[test]
    fn parses_layer_types_from_hf_json() {
        let json = r#"{
            "model_type": "lfm2",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "layer_types": ["conv", "short_conv", "attention", "moe"],
            "unused_field": true
        }"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm2");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(
            cfg.layers,
            vec![
                LayerKind::Conv,
                LayerKind::Conv,
                LayerKind::Attention,
                LayerKind::Other("moe".to_string()),
            ]
        );
    }

    #[test]
    fn accepts_block_types_alias() {
        let json = r#"{"model_type":"lfm3","block_types":["gqa","conv"]}"#;
        let cfg = ModelConfig::from_json(json).unwrap();
        assert_eq!(cfg.model_type, "lfm3");
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.layers, vec![LayerKind::Attention, LayerKind::Conv]);
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
}
