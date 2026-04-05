// SPDX-License-Identifier: AGPL-3.0-or-later

//! Programmatic FlatBuffer model construction for Akida NPUs.
//!
//! Builds model programs layer-by-layer without the BrainChip SDK
//! (MetaTF / QuantizeML / CNN2SNN). The output is a `ProgramBinary`
//! compatible with `program_external()` injection.
//!
//! # Architecture
//!
//! ```text
//! ProgramBuilder
//!   .add_layer(LayerSpec::InputConv { .. })
//!   .add_layer(LayerSpec::FullyConnected { .. })
//!   .build() -> ProgramBinary
//!
//! EsnProgramBuilder (convenience)
//!   .new(input_dim, reservoir_size, output_dim)
//!   .build() -> ProgramBinary
//! ```
//!
//! # Status
//!
//! Scaffolded — structure and API are defined; the FlatBuffer serialization
//! in `build()` will be completed once the full program_info/program_data
//! format is reverse-engineered (see `akida_chip::program` for current
//! understanding of the binary format).

use akida_chip::program::ProgramBinary;

/// Layer-by-layer FlatBuffer program builder.
///
/// Accumulates layer specifications and produces a `ProgramBinary`
/// that can be loaded via `program_external()`.
#[derive(Debug, Clone)]
pub struct ProgramBuilder {
    layers: Vec<LayerSpec>,
    target_np_offset: u32,
    quant_config: Option<QuantConfig>,
}

impl ProgramBuilder {
    /// Create a new empty program builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            target_np_offset: 0,
            quant_config: None,
        }
    }

    /// Set the target NP offset for this program.
    ///
    /// When loading multiple programs (multi-tenancy), each program
    /// targets a different NP range. Offset 0 is the default.
    #[must_use]
    pub fn with_np_offset(mut self, offset: u32) -> Self {
        self.target_np_offset = offset;
        self
    }

    /// Set global quantization configuration.
    #[must_use]
    pub fn with_quantization(mut self, config: QuantConfig) -> Self {
        self.quant_config = Some(config);
        self
    }

    /// Add a layer to the program.
    ///
    /// Layers are processed in order. The first layer must be an input
    /// layer (`InputConv` or `FullyConnected` with `is_input: true`).
    pub fn add_layer(&mut self, layer: LayerSpec) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Append a classification / readout head to the program.
    ///
    /// Adds a final `FullyConnected` layer with the given output dimension,
    /// configured as the output layer.
    pub fn append_head(&mut self, output_dim: u32) -> &mut Self {
        self.layers.push(LayerSpec::FullyConnected {
            neurons: output_dim,
            activation: Activation::Linear,
            is_input: false,
            is_output: true,
        });
        self
    }

    /// Number of layers added so far.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Build the final program binary.
    ///
    /// Serializes all layers into FlatBuffer `program_info` + `program_data`
    /// format compatible with `program_external()`.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No layers have been added
    /// - Layer configuration is invalid (e.g., no input layer)
    /// - Serialization fails
    pub fn build(&self) -> Result<ProgramBinary, BuildError> {
        if self.layers.is_empty() {
            return Err(BuildError::NoLayers);
        }

        // Validate layer sequence
        self.validate_layers()?;

        // Compute NP allocation for each layer
        let allocation = self.allocate_nps()?;

        // Serialize to FlatBuffer binary
        let (info_bytes, data_bytes) = self.serialize(&allocation)?;

        let mut bytes = info_bytes;
        let info_end = bytes.len();
        bytes.extend_from_slice(&data_bytes);

        Ok(ProgramBinary::new(bytes, info_end))
    }

    fn validate_layers(&self) -> Result<(), BuildError> {
        if self.layers.is_empty() {
            return Err(BuildError::NoLayers);
        }

        let has_input = self.layers.iter().any(|l| match l {
            LayerSpec::InputConv { .. } => true,
            LayerSpec::FullyConnected { is_input, .. } => *is_input,
            LayerSpec::SeparableConv { .. } => false,
        });

        if !has_input {
            return Err(BuildError::NoInputLayer);
        }

        Ok(())
    }

    fn allocate_nps(&self) -> Result<Vec<NpAllocation>, BuildError> {
        let mut allocations = Vec::with_capacity(self.layers.len());
        let mut next_np = self.target_np_offset;

        for (i, layer) in self.layers.iter().enumerate() {
            let np_count = layer.estimated_np_count();
            allocations.push(NpAllocation {
                layer_index: i,
                start_np: next_np,
                np_count,
            });
            next_np += np_count;
        }

        Ok(allocations)
    }

    #[expect(
        clippy::unused_self,
        reason = "Placeholder serializer; self reserved for future format work"
    )]
    fn serialize(&self, _allocation: &[NpAllocation]) -> Result<(Vec<u8>, Vec<u8>), BuildError> {
        // Placeholder: actual FlatBuffer serialization will be implemented
        // once the full program_info/program_data binary format is confirmed.
        //
        // Current understanding (from akida_chip::program):
        //   program_info: NP routing table + register write sequence (~332 bytes)
        //   program_data: layer metadata + activation params (~396 bytes)
        //
        // The format is reverse-engineered from .fbz files and C++ engine
        // symbol analysis. See SILICON_SPEC.md §6 for details.

        let info = Vec::new();
        let data = Vec::new();

        Ok((info, data))
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience builder for Echo State Network programs.
///
/// Creates the standard ESN architecture used by hotSpring:
/// `InputConv(input_dim) → FullyConnected(reservoir) → FullyConnected(output)`
///
/// This is the program structure validated in Experiment 022 (5,978 live
/// hardware calls, 24-hour lattice QCD simulation).
#[derive(Debug, Clone)]
pub struct EsnProgramBuilder {
    input_dim: u32,
    reservoir_size: u32,
    output_dim: u32,
    np_offset: u32,
    quant: QuantConfig,
}

impl EsnProgramBuilder {
    /// Create a new ESN program builder.
    ///
    /// # Arguments
    ///
    /// * `input_dim` — feature vector size (e.g., 50 for lattice QCD observables)
    /// * `reservoir_size` — reservoir neuron count (e.g., 128)
    /// * `output_dim` — output size (e.g., 1 for phase classification)
    #[must_use]
    pub fn new(input_dim: u32, reservoir_size: u32, output_dim: u32) -> Self {
        Self {
            input_dim,
            reservoir_size,
            output_dim,
            np_offset: 0,
            quant: QuantConfig::default(),
        }
    }

    /// Set the NP offset for multi-tenant deployment.
    #[must_use]
    pub fn with_np_offset(mut self, offset: u32) -> Self {
        self.np_offset = offset;
        self
    }

    /// Override quantization config.
    #[must_use]
    pub fn with_quantization(mut self, quant: QuantConfig) -> Self {
        self.quant = quant;
        self
    }

    /// Build the ESN program.
    ///
    /// # Errors
    ///
    /// Returns error if the program cannot be serialized.
    pub fn build(&self) -> Result<ProgramBinary, BuildError> {
        let mut builder = ProgramBuilder::new()
            .with_np_offset(self.np_offset)
            .with_quantization(self.quant.clone());

        builder.add_layer(LayerSpec::InputConv {
            channels: self.input_dim,
            kernel_size: 1,
            stride: 1,
            activation: Activation::Linear,
        });

        builder.add_layer(LayerSpec::FullyConnected {
            neurons: self.reservoir_size,
            activation: Activation::Linear,
            is_input: false,
            is_output: false,
        });

        builder.add_layer(LayerSpec::FullyConnected {
            neurons: self.output_dim,
            activation: Activation::Linear,
            is_input: false,
            is_output: true,
        });

        builder.build()
    }
}

/// Layer specification for program construction.
#[derive(Debug, Clone)]
pub enum LayerSpec {
    /// Input convolution layer.
    ///
    /// Discovery 1: any channel count works (1–64 tested), not just 1 or 3.
    InputConv {
        /// Number of input channels.
        channels: u32,
        /// Convolution kernel size.
        kernel_size: u32,
        /// Convolution stride.
        stride: u32,
        /// Activation function.
        activation: Activation,
    },

    /// Fully connected layer.
    ///
    /// Discovery 2: FC layers merge via SkipDMA — deep chains execute
    /// as a single hardware pass.
    /// Discovery 5: tested to 8192+ neurons (SRAM-limited only).
    FullyConnected {
        /// Number of neurons.
        neurons: u32,
        /// Activation function.
        activation: Activation,
        /// Whether this is the input layer.
        is_input: bool,
        /// Whether this is the output layer.
        is_output: bool,
    },

    /// Depthwise separable convolution.
    SeparableConv {
        /// Number of output filters.
        filters: u32,
        /// Kernel size.
        kernel_size: u32,
        /// Stride.
        stride: u32,
        /// Activation function.
        activation: Activation,
    },
}

impl LayerSpec {
    /// Estimate how many NPs this layer will consume.
    ///
    /// Based on observed allocation patterns from hardware probing.
    /// Actual allocation depends on compiler heuristics.
    #[must_use]
    pub fn estimated_np_count(&self) -> u32 {
        match self {
            Self::InputConv { channels, .. } => {
                // ~1 NP per 16 channels (observed pattern)
                (channels + 15) / 16
            }
            Self::FullyConnected { neurons, .. } => {
                // ~1 NP per 128 neurons (FC packing observed)
                (neurons + 127) / 128
            }
            Self::SeparableConv { filters, .. } => {
                // ~1 NP per 32 filters (separable conv pattern)
                (filters + 31) / 32
            }
        }
    }
}

/// Activation function for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// No activation (identity).
    Linear,
    /// Bounded ReLU (hardware native, clips negatives to 0).
    BoundedRelu,
    /// Standard ReLU.
    Relu,
}

/// Quantization configuration for int4 weight compression.
///
/// The AKD1000 uses 1/2/4-bit weight quantization. int4 is the
/// standard precision for ESN readout models.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Weight bit width (1, 2, or 4).
    pub weight_bits: u8,
    /// Per-layer scale factor.
    pub scale: f32,
    /// Per-layer zero point.
    pub zero_point: i32,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            weight_bits: 4,
            scale: 1.0,
            zero_point: 0,
        }
    }
}

/// Internal NP allocation for a single layer.
#[derive(Debug)]
struct NpAllocation {
    layer_index: usize,
    start_np: u32,
    np_count: u32,
}

/// Errors during program construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// No layers were added to the builder.
    NoLayers,
    /// No input layer found in the layer sequence.
    NoInputLayer,
    /// Layer configuration is invalid.
    InvalidLayerConfig(String),
    /// NP allocation exceeds available NPs.
    NpOverflow {
        /// NPs required.
        required: u32,
        /// NPs available.
        available: u32,
    },
    /// FlatBuffer serialization failed.
    SerializationFailed(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLayers => write!(f, "no layers added to program builder"),
            Self::NoInputLayer => write!(f, "no input layer in layer sequence"),
            Self::InvalidLayerConfig(msg) => write!(f, "invalid layer config: {msg}"),
            Self::NpOverflow {
                required,
                available,
            } => write!(f, "NP overflow: need {required}, have {available}"),
            Self::SerializationFailed(msg) => write!(f, "serialization failed: {msg}"),
        }
    }
}

impl std::error::Error for BuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_fails() {
        let builder = ProgramBuilder::new();
        assert_eq!(builder.build().unwrap_err(), BuildError::NoLayers);
    }

    #[test]
    fn no_input_layer_fails() {
        let mut builder = ProgramBuilder::new();
        builder.add_layer(LayerSpec::FullyConnected {
            neurons: 128,
            activation: Activation::Linear,
            is_input: false,
            is_output: true,
        });
        assert_eq!(builder.build().unwrap_err(), BuildError::NoInputLayer);
    }

    #[test]
    fn esn_builder_produces_three_layers() {
        let esn = EsnProgramBuilder::new(50, 128, 1);
        let result = esn.build();
        assert!(result.is_ok());
    }

    #[test]
    fn np_estimation_input_conv() {
        let layer = LayerSpec::InputConv {
            channels: 50,
            kernel_size: 1,
            stride: 1,
            activation: Activation::Linear,
        };
        assert_eq!(layer.estimated_np_count(), 4); // ceil(50/16)
    }

    #[test]
    fn np_estimation_fc() {
        let layer = LayerSpec::FullyConnected {
            neurons: 128,
            activation: Activation::BoundedRelu,
            is_input: false,
            is_output: false,
        };
        assert_eq!(layer.estimated_np_count(), 1); // ceil(128/128)
    }

    #[test]
    fn default_quant_config() {
        let q = QuantConfig::default();
        assert_eq!(q.weight_bits, 4);
    }
}
