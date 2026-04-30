// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reverse-engineered FlatBuffer schema for Akida `.fbz` model files.
//!
//! Observed across all 25 zoo artifacts:
//!
//! ```text
//! Root table at offset 0x10, VTable at 0x06:
//!   VTable size: 10 bytes, Table size: 12 bytes, 3 fields
//!   Field 0 (offset 4): offset → sub-table (model/layers structure)
//!   Field 1: absent in all observed models
//!   Field 2 (offset 8): offset → version string (e.g. "2.19.1")
//!
//! Layer entries follow as a nested FlatBuffer table tree. Each layer
//! contains key-value pairs: "layer_type", "weights_bits", "units",
//! "output_bits", "buffer_bits", "post_op_buffer_bits", "scales",
//! plus binary weight data at the end of the buffer.
//! ```
//!
//! This module provides a FlatBuffer serializer that constructs `.fbz`-
//! compatible binaries using the `flatbuffers` crate's builder API,
//! and a Snappy compressor that wraps the output.

use flatbuffers::FlatBufferBuilder;

/// Build a complete FlatBuffer payload matching the Akida `.fbz` structure.
///
/// The output is a raw FlatBuffer binary (NOT Snappy-compressed). Call
/// [`compress_to_fbz`] to produce a final `.fbz` file.
pub fn build_model_flatbuffer(desc: &ModelDescriptor) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(4096);

    // FlatBuffers requires ALL strings/vectors/sub-tables to be created
    // BEFORE starting their parent table. Build bottom-up.

    // Phase 1: Build all layer tables (which internally build KV pairs first)
    let layer_offsets: Vec<_> = desc
        .layers
        .iter()
        .map(|layer| build_layer_table(&mut fbb, layer))
        .collect();

    // Phase 2: Build the layers vector and version string
    let layers_vec = fbb.create_vector(&layer_offsets);
    let version = fbb.create_string(&desc.version);

    // Phase 3: Build model sub-table (field 0 of root)
    let model_table = {
        let start = fbb.start_table();
        fbb.push_slot_always(4, layers_vec);
        fbb.end_table(start)
    };

    // Phase 4: Build root table
    let root = {
        let start = fbb.start_table();
        fbb.push_slot_always(4, model_table);
        fbb.push_slot_always(8, version);
        fbb.end_table(start)
    };

    fbb.finish(root, None);
    fbb.finished_data().to_vec()
}

fn build_layer_table<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    layer: &LayerDescriptor,
) -> flatbuffers::WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    // Build ALL nested content before starting the layer table.

    // KV pair tables (each needs its strings pre-built)
    let kv_offsets: Vec<_> = layer
        .properties
        .iter()
        .map(|(k, v)| build_kv_pair(fbb, k, v))
        .collect();

    // Pre-build layer-level strings and vectors
    let kv_vec = fbb.create_vector(&kv_offsets);
    let name = fbb.create_string(&layer.name);
    let weight_blob = layer.weight_data.as_ref().map(|w| fbb.create_vector(w));

    // NOW start the table
    let start = fbb.start_table();
    fbb.push_slot_always(4, name);
    fbb.push_slot_always(6, kv_vec);
    if let Some(wb) = weight_blob {
        fbb.push_slot_always(8, wb);
    }
    fbb.end_table(start)
}

fn build_kv_pair<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    key: &str,
    value: &PropertyValue,
) -> flatbuffers::WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    // Pre-build ALL strings before starting the table
    let key_off = fbb.create_string(key);
    let str_off = match value {
        PropertyValue::Str(s) => Some(fbb.create_string(s)),
        PropertyValue::Int(_) => None,
    };

    let start = fbb.start_table();
    fbb.push_slot_always(4, key_off);
    match value {
        PropertyValue::Int(v) => fbb.push_slot::<i64>(6, *v, 0),
        PropertyValue::Str(_) => {
            if let Some(off) = str_off {
                fbb.push_slot_always(6, off);
            }
        }
    }
    fbb.end_table(start)
}

/// Snappy-compress a FlatBuffer payload into `.fbz` format.
///
/// Output format: `[varint: uncompressed_size][snappy_compressed_data]`
#[must_use]
pub fn compress_to_fbz(flatbuffer: &[u8]) -> Vec<u8> {
    snap::raw::Encoder::new()
        .compress_vec(flatbuffer)
        .expect("snappy compression should not fail on valid input")
}

/// Write a complete `.fbz` file from a model descriptor.
#[must_use]
pub fn build_fbz(desc: &ModelDescriptor) -> Vec<u8> {
    let fb = build_model_flatbuffer(desc);
    compress_to_fbz(&fb)
}

/// High-level description of a model to serialize.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    /// SDK version string (e.g. "2.19.1").
    pub version: String,
    /// Layers in order.
    pub layers: Vec<LayerDescriptor>,
}

/// Description of a single layer.
#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    /// Layer name (e.g. "conv_0", "dense_1").
    pub name: String,
    /// Key-value properties (layer_type, weights_bits, etc.).
    pub properties: Vec<(String, PropertyValue)>,
    /// Optional raw weight data.
    pub weight_data: Option<Vec<u8>>,
}

/// Value type for layer properties.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    /// Integer value (weights_bits, units, etc.).
    Int(i64),
    /// String value (layer_type name, etc.).
    Str(String),
}

impl ModelDescriptor {
    /// Create a minimal model descriptor.
    #[must_use]
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            layers: Vec::new(),
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: LayerDescriptor) {
        self.layers.push(layer);
    }
}

impl LayerDescriptor {
    /// Create a new layer descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            properties: Vec::new(),
            weight_data: None,
        }
    }

    /// Add a string property.
    pub fn with_str(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties
            .push((key.into(), PropertyValue::Str(value.into())));
        self
    }

    /// Add an integer property.
    pub fn with_int(mut self, key: impl Into<String>, value: i64) -> Self {
        self.properties
            .push((key.into(), PropertyValue::Int(value)));
        self
    }

    /// Attach raw weight data.
    pub fn with_weights(mut self, data: Vec<u8>) -> Self {
        self.weight_data = Some(data);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_minimal_flatbuffer() {
        let desc = ModelDescriptor::new("2.19.1");
        let fb = build_model_flatbuffer(&desc);
        assert!(!fb.is_empty());

        // Verify FlatBuffer root table offset
        let root_off = u32::from_le_bytes([fb[0], fb[1], fb[2], fb[3]]) as usize;
        assert!(root_off > 0 && root_off < fb.len());
    }

    #[test]
    fn build_fbz_round_trip_through_parser() {
        let mut desc = ModelDescriptor::new("2.19.1");
        desc.add_layer(
            LayerDescriptor::new("input_conv")
                .with_str("layer_type", "InputConv")
                .with_int("weights_bits", 4),
        );
        desc.add_layer(
            LayerDescriptor::new("fc_output")
                .with_str("layer_type", "FullyConnected")
                .with_int("units", 10),
        );

        let fbz = build_fbz(&desc);
        assert!(!fbz.is_empty());

        // Parse it back with our parser
        let model = crate::Model::from_bytes(&fbz).expect("round-trip parse");
        assert_eq!(model.version(), "2.19.1");
    }

    #[test]
    fn compress_decompress_round_trip() {
        let fb = build_model_flatbuffer(&ModelDescriptor::new("2.19.1"));
        let compressed = compress_to_fbz(&fb);

        let decompressed = snap::raw::Decoder::new()
            .decompress_vec(&compressed)
            .expect("decompress");
        assert_eq!(decompressed, fb);
    }

    #[test]
    fn layer_with_weights_serializes() {
        let weights = vec![0xAA_u8; 128];
        let mut desc = ModelDescriptor::new("2.19.1");
        desc.add_layer(
            LayerDescriptor::new("dense_0")
                .with_str("layer_type", "FullyConnected")
                .with_int("weights_bits", 8)
                .with_weights(weights),
        );

        let fbz = build_fbz(&desc);
        let model = crate::Model::from_bytes(&fbz).expect("parse with weights");
        assert_eq!(model.version(), "2.19.1");
    }

    #[test]
    fn model_descriptor_builder_api() {
        let desc = ModelDescriptor::new("2.19.1");
        assert_eq!(desc.version, "2.19.1");
        assert!(desc.layers.is_empty());

        let layer = LayerDescriptor::new("conv_0")
            .with_str("layer_type", "Conv2D")
            .with_int("units", 64);
        assert_eq!(layer.name, "conv_0");
        assert_eq!(layer.properties.len(), 2);
    }
}
