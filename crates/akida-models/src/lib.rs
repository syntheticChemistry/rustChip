// SPDX-License-Identifier: AGPL-3.0-or-later

#![deny(unsafe_code)]
#![warn(clippy::expect_used, clippy::unwrap_used)]

//! Akida neural network model parser
//!
//! This crate provides parsing and loading capabilities for Akida `.fbz` model files.
//!
//! # Format
//!
//! Akida `.fbz` files are **Snappy-compressed FlatBuffers with zero-padding**:
//!
//! ```text
//! [varint: uncompressed_size] [snappy_chunks] [zero_padding]
//! ```
//!
//! - **Varint**: standard Snappy/LEB128 varint encoding the uncompressed
//!   FlatBuffer size (1–5 bytes)
//! - **Snappy chunks**: compressed FlatBuffer data in Snappy raw/block format
//! - **Zero-padding**: trailing zero bytes to an alignment boundary (BrainChip
//!   SDK artifact). These must be stripped before decompression since `0x00`
//!   is a valid 1-byte Snappy literal that would overflow the output buffer.
//!
//! After decompression, the payload is a standard FlatBuffer binary:
//! - Bytes `[0..4]`: root table offset (u32 LE, typically 0x10)
//! - Version string (e.g., "2.18.2") via vtable traversal
//! - Layer definitions, metadata, and quantized weight data
//!
//! Validated against the full BrainChip model zoo (v1 + v2) and hand-built
//! test models from `ProgramBuilder`.
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
pub mod import;
mod inference;
mod loading;
mod model;
mod parser;
pub mod quantize;
pub mod schema;
pub mod schema_parser;
mod shapes;
mod weights;
pub mod guidestone;
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
