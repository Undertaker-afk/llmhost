use crate::{LayerKind, ModelConfig};

/// Placeholder model. Weights, blocks, and a tokenizer are added in M1.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// Architecture and layer graph for this model.
    pub config: ModelConfig,
}

impl Model {
    /// Wrap a parsed config. Loading weights is a later milestone.
    #[must_use]
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Layer kinds in stack order.
    #[must_use]
    pub fn layers(&self) -> &[LayerKind] {
        &self.config.layers
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::{LayerKind, ModelConfig};

    #[test]
    fn exposes_layer_graph() {
        let config = ModelConfig {
            model_type: "lfm2".into(),
            hidden_size: 8,
            num_hidden_layers: 2,
            layers: vec![
                LayerKind::Conv { kernel: 3 },
                LayerKind::Attention {
                    sliding_window: None,
                    rope: crate::RopeParams::default(),
                },
            ],
        };
        let model = Model::new(config);
        assert_eq!(model.layers().len(), 2);
    }
}
