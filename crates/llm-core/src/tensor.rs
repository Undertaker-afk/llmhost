use crate::Shape;
use std::fmt;

/// Error from building a tensor whose data does not match its shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// `data.len()` was not equal to `shape.numel()`.
    LengthMismatch { shape: Vec<usize>, len: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { shape, len } => {
                write!(f, "tensor data length {len} does not match shape {shape:?}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Owned, contiguous f32 tensor. Layout is a placeholder for milestone 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
}

impl Tensor {
    /// A tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.numel()];
        Self { shape, data }
    }

    /// A tensor from existing data. The vector length must equal `shape.numel()`.
    pub fn from_vec(shape: Shape, data: Vec<f32>) -> Result<Self, TensorError> {
        if data.len() != shape.numel() {
            return Err(TensorError::LengthMismatch {
                shape: shape.dims().to_vec(),
                len: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::Shape;

    #[test]
    fn rejects_length_mismatch() {
        let err = Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            super::TensorError::LengthMismatch { len: 2, .. }
        ));
    }
}
