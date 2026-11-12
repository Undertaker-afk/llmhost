//! Weight and KV-cache type tags.
//!
//! These are format identifiers only. Quantization kernels for Q8_0 / Q4 live
//! in this crate; NVFP4 is reserved for the later GPU path.

use std::fmt;

/// How model weights are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    F32,
    F16,
    BF16,
    /// 32-element int8 blocks with an f16 scale ([`crate::BlockQ8_0`]).
    Q8_0,
    /// 32-element 4-bit blocks with an f16 scale ([`crate::BlockQ4`]).
    Q4_32,
    /// Reserved for the later GPU path (RTX 5090-class, 27B). Not implemented.
    Nvfp4,
}

/// Error from a format that this crate cannot (yet) produce or consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantError {
    /// The requested format is recognized but not supported.
    Unsupported {
        format: WeightFormat,
        reason: &'static str,
    },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { format, reason } => {
                write!(f, "unsupported weight format {format:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for QuantError {}

impl WeightFormat {
    /// Return `Ok(())` for implemented formats.
    ///
    /// [`WeightFormat::Nvfp4`] is reserved and returns [`QuantError::Unsupported`].
    pub fn ensure_supported(self) -> Result<(), QuantError> {
        match self {
            Self::Nvfp4 => Err(QuantError::Unsupported {
                format: self,
                reason: "NVFP4 is reserved for the GPU path and is not implemented",
            }),
            Self::F32 | Self::F16 | Self::BF16 | Self::Q8_0 | Self::Q4_32 => Ok(()),
        }
    }
}

/// Storage type for the paged KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvDtype {
    /// Full f16 keys and values.
    F16,
    /// 8-bit keys and values.
    Q8,
    /// 8-bit keys and 4-bit values.
    K8V4,
}

#[cfg(test)]
mod tests {
    use super::{KvDtype, QuantError, WeightFormat};

    #[test]
    fn implemented_weight_formats_are_supported() {
        for format in [
            WeightFormat::F32,
            WeightFormat::F16,
            WeightFormat::BF16,
            WeightFormat::Q8_0,
            WeightFormat::Q4_32,
        ] {
            format.ensure_supported().expect("implemented format");
        }
    }

    #[test]
    fn nvfp4_is_reserved_and_returns_unsupported() {
        let err = WeightFormat::Nvfp4
            .ensure_supported()
            .expect_err("NVFP4 must be unsupported");
        assert_eq!(
            err,
            QuantError::Unsupported {
                format: WeightFormat::Nvfp4,
                reason: "NVFP4 is reserved for the GPU path and is not implemented",
            }
        );
        assert!(err.to_string().contains("NVFP4"));
    }

    #[test]
    fn kv_dtype_variants_exist() {
        // Smoke test so the tags stay part of the public API.
        let tags = [KvDtype::F16, KvDtype::Q8, KvDtype::K8V4];
        assert_eq!(tags.len(), 3);
        assert_ne!(KvDtype::K8V4, KvDtype::Q8);
    }
}
