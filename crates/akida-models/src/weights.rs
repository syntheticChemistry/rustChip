//! Weight data extraction and parsing
//!
//! Handles parsing of quantized weight data from Akida models.

use crate::error::{AkidaModelError, Result};
use bytes::Bytes;

/// Weight quantization configuration
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    /// Number of bits per weight
    pub bits: u8,

    /// Scale factor
    pub scale: f32,

    /// Zero-point offset
    pub offset: i32,
}

/// Parsed weight data
#[derive(Debug, Clone)]
pub struct WeightData {
    /// Quantization configuration
    pub quantization: QuantizationConfig,

    /// Raw weight bytes (Bytes enables zero-copy cloning for large model weights)
    pub data: Bytes,

    /// Weight dimensions (if available)
    pub shape: Option<Vec<usize>>,
}

impl WeightData {
    /// Create new weight data
    pub fn new(quantization: QuantizationConfig, data: impl Into<Bytes>) -> Self {
        Self {
            quantization,
            data: data.into(),
            shape: None,
        }
    }

    /// Set weight shape
    #[must_use]
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Get total number of weights
    #[must_use]
    pub fn weight_count(&self) -> usize {
        if let Some(shape) = &self.shape {
            shape.iter().product()
        } else {
            // Estimate from data size
            self.data.len() * 8 / self.quantization.bits as usize
        }
    }

    /// Decode quantized weights to f32
    ///
    /// # Errors
    ///
    /// Returns error if data is malformed.
    pub fn decode(&self) -> Result<Vec<f32>> {
        let weight_count = self.weight_count();
        let mut weights = Vec::with_capacity(weight_count);

        match self.quantization.bits {
            1 => self.decode_1bit(&mut weights),
            2 => self.decode_2bit(&mut weights),
            4 => self.decode_4bit(&mut weights),
            8 => self.decode_8bit(&mut weights),
            _ => {
                return Err(AkidaModelError::parse_error(format!(
                    "Unsupported bit width: {}",
                    self.quantization.bits
                )));
            }
        }

        Ok(weights)
    }

    /// Decode 1-bit weights
    fn decode_1bit(&self, weights: &mut Vec<f32>) {
        for &byte in self.data.as_ref() {
            for bit_idx in 0..8 {
                let bit = (byte >> bit_idx) & 1;
                let quantized = i32::from(bit) - self.quantization.offset;
                #[allow(clippy::cast_precision_loss)]
                let weight = quantized as f32 * self.quantization.scale;
                weights.push(weight);
            }
        }
    }

    /// Decode 2-bit weights
    fn decode_2bit(&self, weights: &mut Vec<f32>) {
        for &byte in self.data.as_ref() {
            for shift in (0..8).step_by(2) {
                let value = (byte >> shift) & 0b11;
                let quantized = i32::from(value) - self.quantization.offset;
                #[allow(clippy::cast_precision_loss)]
                let weight = quantized as f32 * self.quantization.scale;
                weights.push(weight);
            }
        }
    }

    /// Decode 4-bit weights
    fn decode_4bit(&self, weights: &mut Vec<f32>) {
        for &byte in self.data.as_ref() {
            // Low nibble
            let low = byte & 0x0F;
            let quantized_low = i32::from(low) - self.quantization.offset;
            #[allow(clippy::cast_precision_loss)]
            let weight_low = quantized_low as f32 * self.quantization.scale;
            weights.push(weight_low);

            // High nibble
            let high = (byte >> 4) & 0x0F;
            let quantized_high = i32::from(high) - self.quantization.offset;
            #[allow(clippy::cast_precision_loss)]
            let weight_high = quantized_high as f32 * self.quantization.scale;
            weights.push(weight_high);
        }
    }

    /// Decode 8-bit weights
    fn decode_8bit(&self, weights: &mut Vec<f32>) {
        for &byte in self.data.as_ref() {
            let quantized = i32::from(byte) - self.quantization.offset;
            #[allow(clippy::cast_precision_loss)]
            let weight = quantized as f32 * self.quantization.scale;
            weights.push(weight);
        }
    }
}

/// Extract weight data from model bytes
///
/// # Errors
///
/// Returns error if weight parsing fails.
pub fn extract_weights(data: &[u8]) -> Result<Vec<WeightData>> {
    let mut weights = Vec::new();

    // Look for weight data patterns
    // Pattern: fe 01 00 repeated (common in Akida models)
    let weight_pattern = [0xfe, 0x01, 0x00];

    let mut i = 0;
    while i + weight_pattern.len() < data.len() {
        if data[i..i + 3] == weight_pattern {
            tracing::debug!("Found weight pattern at offset 0x{:x}", i);

            // Extract weight block (simplified heuristic)
            let block_start = i;
            let mut block_end = i + 3;

            // Find end of repeated pattern
            while block_end + 3 < data.len() && data[block_end..block_end + 3] == weight_pattern {
                block_end += 3;
            }

            // Create weight data (default 4-bit quantization)
            let weight_block = Bytes::copy_from_slice(&data[block_start..block_end]);
            let quant = QuantizationConfig {
                bits: 4,
                scale: 1.0,
                offset: 0,
            };

            weights.push(WeightData::new(quant, weight_block));

            i = block_end;
        } else {
            i += 1;
        }
    }

    tracing::info!("Extracted {} weight block(s)", weights.len());
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1bit_decode() {
        let quant = QuantizationConfig {
            bits: 1,
            scale: 1.0,
            offset: 0,
        };

        let data = vec![0b1010_0101]; // 8 bits
        let weights = WeightData::new(quant, data);

        let decoded = weights.decode().unwrap();
        assert_eq!(decoded.len(), 8);
        assert!((decoded[0] - 1.0).abs() < 0.01); // bit 0
        assert!((decoded[1] - 0.0).abs() < 0.01); // bit 1
    }

    #[test]
    fn test_4bit_decode() {
        let quant = QuantizationConfig {
            bits: 4,
            scale: 0.1,
            offset: 8,
        };

        let data = vec![0x12]; // 0x1 and 0x2
        let weights = WeightData::new(quant, data);

        let decoded = weights.decode().unwrap();
        assert_eq!(decoded.len(), 2);
        // First nibble: (2 - 8) * 0.1 = -0.6
        // Second nibble: (1 - 8) * 0.1 = -0.7
        assert!((decoded[0] - (-0.6)).abs() < 0.01);
        assert!((decoded[1] - (-0.7)).abs() < 0.01);
    }

    #[test]
    fn test_weight_count() {
        let quant = QuantizationConfig {
            bits: 4,
            scale: 1.0,
            offset: 0,
        };

        let data = vec![0u8; 10]; // 10 bytes
        let weights = WeightData::new(quant, data);

        // 10 bytes * 8 bits / 4 bits per weight = 20 weights
        assert_eq!(weights.weight_count(), 20);
    }
}
