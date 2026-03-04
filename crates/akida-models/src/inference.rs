// SPDX-License-Identifier: AGPL-3.0-or-later

//! Model inference integration
//!
//! Bridges model metadata with device inference execution.

use crate::{AkidaModelError, Model, Result};

impl Model {
    /// Run inference on device
    ///
    /// **Deep Debt**: Complete implementation, self-knowledge!
    ///
    /// This method:
    /// 1. Extracts input/output shapes from model
    /// 2. Creates inference configuration
    /// 3. Executes on device NPUs
    /// 4. Returns results with metrics
    ///
    /// # Example
    ///
    /// ```no_run
    /// use akida_models::Model;
    /// use akida_driver::DeviceManager;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Load model
    /// let model = Model::from_file("model.fbz")?;
    ///
    /// // Get device
    /// let manager = DeviceManager::discover()?;
    /// let mut device = manager.open_first()?;
    ///
    /// // Load model to device
    /// model.load_to_device(&mut device)?;
    ///
    /// // Run inference
    /// let input = vec![0u8; model.input_size()];
    /// let result = model.infer(&input, &mut device)?;
    ///
    /// println!("Inference: {:?} in {:?}",
    ///          result.output.len(), result.total_duration);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input size doesn't match model
    /// - Inference fails
    /// - Output cannot be retrieved
    pub fn infer(
        &self,
        input: &[u8],
        device: &mut akida_driver::AkidaDevice,
    ) -> Result<akida_driver::InferenceResult> {
        use akida_driver::{InferenceConfig, InferenceExecutor};

        tracing::info!("Running inference on device {}", device.index());

        let (input_shape, output_shape) = self.io_shapes();

        // Create inference config from model metadata
        let config = InferenceConfig::new(
            input_shape,
            output_shape,
            1, // uint8 input (standard for Akida)
            1, // uint8 output (Akida uses quantized outputs)
        );

        tracing::debug!(
            "Inference config: input={} bytes, output={} bytes",
            config.input_size_bytes(),
            config.output_size_bytes()
        );

        // Execute inference
        let executor = InferenceExecutor::new(config);
        let result = executor
            .infer(input, device)
            .map_err(|e| AkidaModelError::loading_failed(format!("Inference failed: {e}")))?;

        tracing::info!("✅ Inference complete: {:?}", result.total_duration);
        Ok(result)
    }

    /// Get input size in bytes.
    ///
    /// Uses model layer metadata when available, falls back to
    /// program size heuristic when FlatBuffers shapes aren't parsed.
    pub fn input_size(&self) -> usize {
        let (input_shape, _) = self.io_shapes();
        input_shape.iter().product()
    }

    /// Get output size in bytes.
    ///
    /// Uses model layer metadata when available, falls back to
    /// program size heuristic when FlatBuffers shapes aren't parsed.
    pub fn output_size(&self) -> usize {
        let (_, output_shape) = self.io_shapes();
        output_shape.iter().product()
    }

    /// Extract input/output shapes from model structure.
    ///
    /// Traverses the layer graph: the first layer's input shape is
    /// the model input; the last layer's output shape is the model output.
    /// When shapes are not yet parsed (pending FlatBuffers schema), derives
    /// a reasonable estimate from program size and layer count.
    fn io_shapes(&self) -> (Vec<usize>, Vec<usize>) {
        let layers = self.layers();

        // Try to get shapes from first/last layers
        let input_shape = layers
            .first()
            .filter(|l| !l.input_shape.is_empty())
            .map(|l| l.input_shape.clone());

        let output_shape = layers
            .last()
            .filter(|l| !l.output_shape.is_empty())
            .map(|l| l.output_shape.clone());

        if let (Some(inp), Some(out)) = (input_shape, output_shape) {
            return (inp, out);
        }

        // Heuristic fallback: derive from program size and weight count.
        // AKD1000 standard Akida models use uint8 I/O with shapes derivable
        // from the weight geometry. Until FlatBuffers schema parsing is
        // complete, use program metadata to estimate.
        let total_weights = self.total_weight_count();
        let estimated_input = if total_weights > 0 {
            // Infer from first weight block dimension
            self.weights()
                .first()
                .and_then(|w| w.shape.as_ref())
                .map_or_else(
                    || vec![self.program_size().min(1024)],
                    |s| {
                        if s.len() >= 2 {
                            vec![s[s.len() - 1]]
                        } else {
                            vec![s[0]]
                        }
                    },
                )
        } else {
            vec![self.program_size().min(1024)]
        };

        let estimated_output = self
            .weights()
            .last()
            .and_then(|w| w.shape.as_ref())
            .map_or(vec![self.layer_count().max(1)], |s| vec![s[0]]);

        tracing::debug!(
            "Estimated I/O shapes from model metadata: input={:?}, output={:?}",
            estimated_input,
            estimated_output
        );

        (estimated_input, estimated_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_output_sizes() {
        // Create minimal test model with version
        let data = vec![0x80, 0x44, 0x04, 0x10]; // FlatBuffers header
        let mut data = [&data[..], &vec![0x00; 1000]].concat();

        // Add version string at offset 30
        let version = b"2.18.2\0";
        data[30..30 + version.len()].copy_from_slice(version);

        let model = Model::from_bytes(&data).expect("Failed to create model");

        let input_size = model.input_size();
        let output_size = model.output_size();

        assert!(input_size > 0);
        assert!(output_size > 0);
    }
}
