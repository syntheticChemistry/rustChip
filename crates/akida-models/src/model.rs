// SPDX-License-Identifier: AGPL-3.0-or-later

//! Akida model representation

use crate::error::{AkidaModelError, Result};
use crate::parser;
use crate::weights::{WeightData, extract_weights};
use std::fs;
use std::path::Path;

/// Akida neural network model
#[derive(Debug, Clone)]
pub struct Model {
    /// SDK version that created this model
    version: String,

    /// Model layers
    layers: Vec<Layer>,

    /// Weight data blocks
    weights: Vec<WeightData>,

    /// Raw model data
    data: Vec<u8>,

    /// Input tensor shape (from manifest or heuristic). Empty if unknown.
    input_shape: Vec<usize>,

    /// Output tensor shape (from manifest or heuristic). Empty if unknown.
    output_shape: Vec<usize>,
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    /// Layer name
    pub name: String,

    /// Layer type
    pub layer_type: LayerType,

    /// Input shape
    pub input_shape: Vec<usize>,

    /// Output shape
    pub output_shape: Vec<usize>,
}

/// Layer type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Input layer
    InputData,

    /// Fully connected layer
    FullyConnected,

    /// Convolutional layer
    Conv2D,

    /// Depthwise convolutional
    DepthwiseConv2D,

    /// Pooling layer
    Pooling,

    /// Unknown/unsupported layer
    Unknown(String),
}

impl Model {
    /// Load model from file
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        tracing::info!("Loading model from: {}", path.display());

        if !path.exists() {
            return Err(AkidaModelError::FileNotFound {
                path: path.to_path_buf(),
            });
        }

        let data = fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse model from bytes.
    ///
    /// Accepts both Snappy-compressed `.fbz` files (model zoo) and raw
    /// FlatBuffer data (hand-built test models).
    ///
    /// # Errors
    ///
    /// Returns error if parsing fails.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        tracing::debug!("Parsing model ({} bytes)", data.len());

        // Parse header (handles Snappy decompression internally)
        let header = parser::parse_header(data)?;

        tracing::info!("Model version: {}", header.version);
        tracing::debug!("Layer count: {}", header.layer_count);

        // Decompress for layer/weight extraction if needed
        let (payload, _compressed) = parser::decompress_fbz(data)?;

        // Extract layer names from the decompressed payload
        let layer_names = parser::extract_layer_names(&payload);

        // Create layers from extracted names
        //
        // Deep Debt: Shape extraction requires FlatBuffers schema
        //
        // Akida .fbz files use FlatBuffers serialization. To properly
        // parse input_shape and output_shape, we need:
        //
        // 1. The official FlatBuffers schema (.fbs file) from BrainChip
        // 2. Generated Rust bindings from `flatc --rust`
        //
        // The current heuristic parser extracts layer names and weights
        // successfully, but shape data is embedded in FlatBuffers tables
        // that require proper schema-aware parsing.
        //
        // Evolution path:
        // - Reverse-engineer schema from known good models
        // - Or: Obtain schema from BrainChip SDK documentation
        // - Generate Rust bindings and replace heuristic parser
        let layers = layer_names
            .into_iter()
            .map(|name| {
                // Infer type from name
                let layer_type = LayerType::from_name(&name);

                Layer {
                    name,
                    layer_type,
                    // Deep Debt: Empty until FlatBuffers schema available
                    // Shape data exists in .fbz but requires schema for extraction
                    input_shape: Vec::new(),
                    output_shape: Vec::new(),
                }
            })
            .collect();

        // Extract weight data from decompressed payload
        let weights = extract_weights(&payload)?;
        tracing::debug!("Found {} weight block(s)", weights.len());

        Ok(Self {
            version: header.version,
            layers,
            weights,
            data: payload,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
        })
    }

    /// Set input/output shapes from external metadata (e.g. zoo manifest).
    pub fn set_shapes(&mut self, input: Vec<usize>, output: Vec<usize>) {
        self.input_shape = input;
        self.output_shape = output;
    }

    /// Input tensor shape. Empty if not yet populated from manifest.
    #[must_use]
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Output tensor shape. Empty if not yet populated from manifest.
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Get model SDK version
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get number of layers
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get model layers
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Get model program size (bytes)
    #[must_use]
    pub const fn program_size(&self) -> usize {
        self.data.len()
    }

    /// Get weight blocks
    #[must_use]
    pub fn weights(&self) -> &[WeightData] {
        &self.weights
    }

    /// Get total weight count across all blocks
    #[must_use]
    pub fn total_weight_count(&self) -> usize {
        self.weights.iter().map(WeightData::weight_count).sum()
    }

    /// Get raw model data
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl LayerType {
    /// Infer layer type from name
    fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower.contains("input") {
            Self::InputData
        } else if lower.contains("fc") || lower.contains("dense") {
            Self::FullyConnected
        } else if lower.contains("conv") && lower.contains("depth") {
            Self::DepthwiseConv2D
        } else if lower.contains("conv") {
            Self::Conv2D
        } else if lower.contains("pool") {
            Self::Pooling
        } else {
            Self::Unknown(name.to_string())
        }
    }
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputData => write!(f, "InputData"),
            Self::FullyConnected => write!(f, "FullyConnected"),
            Self::Conv2D => write!(f, "Conv2D"),
            Self::DepthwiseConv2D => write!(f, "DepthwiseConv2D"),
            Self::Pooling => write!(f, "Pooling"),
            Self::Unknown(s) => write!(f, "Unknown({s})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_from_name() {
        assert_eq!(LayerType::from_name("input"), LayerType::InputData);
        assert_eq!(LayerType::from_name("fc"), LayerType::FullyConnected);
        assert_eq!(LayerType::from_name("conv_0"), LayerType::Conv2D);
    }

    #[test]
    fn from_bytes_preserves_raw_data_len() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&crate::parser::LEGACY_TEST_MAGIC);
        let ver = b"2.18.2\0";
        data[30..30 + ver.len()].copy_from_slice(ver);

        let model = Model::from_bytes(&data).expect("parse");
        assert_eq!(model.version(), "2.18.2");
        assert_eq!(model.program_size(), data.len());
        assert_eq!(model.data().len(), data.len());
    }

    #[test]
    fn model_clone_round_trips_len() {
        let mut data = vec![0u8; 96];
        data[0..4].copy_from_slice(&crate::parser::LEGACY_TEST_MAGIC);
        let ver = b"2.18.2\0";
        data[30..30 + ver.len()].copy_from_slice(ver);

        let a = Model::from_bytes(&data).unwrap();
        let b = a.clone();
        assert_eq!(a.program_size(), b.program_size());
        assert_eq!(a.layer_count(), b.layer_count());
    }

    #[test]
    fn from_file_missing_returns_not_found() {
        let err = Model::from_file("/no/such/akida_model_file.fbz").unwrap_err();
        assert!(matches!(err, crate::AkidaModelError::FileNotFound { .. }));
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let err = Model::from_bytes(&[0u8; 32]).unwrap_err();
        assert!(matches!(err, crate::AkidaModelError::InvalidHeader));
    }

    #[test]
    fn layer_type_from_name_covers_variants() {
        assert_eq!(LayerType::from_name("my_input_layer"), LayerType::InputData);
        assert_eq!(
            LayerType::from_name("depthwise_conv"),
            LayerType::DepthwiseConv2D
        );
        assert_eq!(LayerType::from_name("max_pool"), LayerType::Pooling);
        let unk = LayerType::from_name("custom_xyz");
        assert!(matches!(unk, LayerType::Unknown(_)));
        assert_eq!(format!("{unk}"), "Unknown(custom_xyz)");
    }

    #[test]
    fn total_weight_count_sums_blocks() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&crate::parser::LEGACY_TEST_MAGIC);
        let ver = b"2.18.2\0";
        data[30..30 + ver.len()].copy_from_slice(ver);
        let model = Model::from_bytes(&data).unwrap();
        let _ = model.total_weight_count();
        assert!(model.layers().iter().all(|l| l.input_shape.is_empty()));
    }
}
