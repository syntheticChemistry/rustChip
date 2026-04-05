// SPDX-License-Identifier: AGPL-3.0-or-later

#![deny(unsafe_code)]
#![warn(clippy::expect_used, clippy::unwrap_used)]

//! Akida neural network model parser
//!
//! This crate provides parsing and loading capabilities for Akida `.fbz` model files.
//!
//! # Format
//!
//! Akida models are stored in `FlatBuffers` binary format with the following structure:
//!
//! - **Header** (16 bytes): `FlatBuffers` magic and table offsets
//! - **Version**: SDK version string (e.g., "2.18.2")
//! - **Metadata**: Model configuration and layer information
//! - **Layers**: Array of layer definitions
//! - **Weights**: Quantized weight data
//!
//! # Example
//!
//! ```no_run
//! use akida_models::Model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load model from file
//! let model = Model::from_file("model.fbz")?;
//!
//! println!("Model version: {}", model.version());
//! println!("Layers: {}", model.layer_count());
//! println!("Program size: {} bytes", model.program_size());
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod builder;
mod error;
mod inference;
mod loading;
mod model;
mod parser;
mod shapes;
mod weights;
pub mod zoo;

pub use error::{AkidaModelError, Result};
pub use model::{Layer, LayerType, Model};
pub use shapes::{Shape, extract_shapes};
pub use weights::{QuantizationConfig, WeightData, extract_weights};
pub use zoo::{ModelTask, ModelZoo, ZooModel};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{Layer, LayerType, Model, ModelZoo, Result, ZooModel};
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn prelude_exports_resolve() {
        let _: Result<()> = Ok(());
        let _ = core::mem::size_of::<Model>();
        let _ = core::mem::size_of::<Layer>();
        let _ = core::mem::size_of::<LayerType>();
        let _ = core::mem::size_of::<ModelZoo>();
        let _ = core::mem::size_of::<ZooModel>();
    }
}
