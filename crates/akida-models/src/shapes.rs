//! Shape and dimension parsing
//!
//! Extracts tensor shapes from Akida model files.

use crate::error::Result;

/// Tensor shape (dimensions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Dimensions (e.g., [batch, height, width, channels])
    pub dims: Vec<usize>,
}

impl Shape {
    /// Create new shape
    pub const fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get total number of elements
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get number of dimensions
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Check if shape is scalar (0-D)
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Check if shape is 1-D
    #[must_use]
    pub const fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

/// Extract shapes from model data
///
/// Looks for common shape patterns in `FlatBuffers` metadata.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn extract_shapes(data: &[u8]) -> Result<Vec<Shape>> {
    let mut shapes = Vec::new();

    // Look for shape patterns
    // Common patterns: sequences of small integers (1-256) representing dimensions

    let mut i = 0;
    while i + 8 < data.len() {
        // Look for potential shape sequences
        if let Some(shape) = try_extract_shape_at(data, i) {
            if is_valid_shape(&shape) {
                tracing::debug!("Found shape: {}", shape);
                shapes.push(shape);
            }
        }
        i += 1;
    }

    Ok(shapes)
}

/// Try to extract shape at specific offset
fn try_extract_shape_at(data: &[u8], offset: usize) -> Option<Shape> {
    let slice = &data[offset..];

    // Look for sequences of 1-4 byte integers
    let mut dims = Vec::new();
    let mut pos = 0;

    // Try to read up to 4 dimensions (most common in neural nets)
    for _ in 0..4 {
        if pos >= slice.len() {
            break;
        }

        // Try reading as little-endian u32
        if pos + 4 <= slice.len() {
            let value =
                u32::from_le_bytes([slice[pos], slice[pos + 1], slice[pos + 2], slice[pos + 3]]);

            // Valid dimension range: 1-4096 (reasonable for neural nets)
            if value > 0 && value < 4096 {
                dims.push(value as usize);
                pos += 4;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if dims.is_empty() {
        None
    } else {
        Some(Shape::new(dims))
    }
}

/// Check if shape looks valid
fn is_valid_shape(shape: &Shape) -> bool {
    // Must have 1-4 dimensions (typical for neural nets)
    if shape.dims.is_empty() || shape.dims.len() > 4 {
        return false;
    }

    // All dimensions must be > 0
    if shape.dims.contains(&0) {
        return false;
    }

    // Total elements should be reasonable (< 10M)
    let total = shape.total_elements();
    if total > 10_000_000 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_total_elements() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.total_elements(), 24);
    }

    #[test]
    fn test_shape_rank() {
        let shape = Shape::new(vec![1, 28, 28, 1]);
        assert_eq!(shape.rank(), 4);
    }

    #[test]
    fn test_shape_display() {
        let shape = Shape::new(vec![1, 224, 224, 3]);
        assert_eq!(format!("{shape}"), "[1, 224, 224, 3]");
    }

    #[test]
    fn test_is_vector() {
        assert!(Shape::new(vec![10]).is_vector());
        assert!(!Shape::new(vec![10, 10]).is_vector());
    }
}
