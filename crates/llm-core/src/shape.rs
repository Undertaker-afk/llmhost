/// Named tensor shape (row-major, contiguous).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from a list of dimensions.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Dimension list, slowest-moving first.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Product of the dimensions (number of elements).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().copied().product()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn numel_is_product() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }
}
