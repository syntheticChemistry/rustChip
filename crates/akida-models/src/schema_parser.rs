// SPDX-License-Identifier: AGPL-3.0-or-later

//! Schema-aware FlatBuffer parser for Akida `.fbz` model files.
//!
//! Replaces the heuristic parser (`parser.rs`) with vtable-navigated
//! field access based on the reverse-engineered schema (see `schema.rs`
//! and `schemas/akida_model.fbs`).
//!
//! ## Observed root structure (consistent across 25 zoo models)
//!
//! ```text
//! Root table at 0x10, VTable at 0x06:
//!   Field 0 (offset 4): model body sub-table (contains layers vector)
//!   Field 1: absent
//!   Field 2 (offset 8): version string
//! ```
//!
//! ## Layer entry structure
//!
//! Each layer entry is a table with key-value property pairs stored as
//! FlatBuffer strings. The version string, layer names, and property
//! keys are all length-prefixed FlatBuffer strings.

use crate::error::{AkidaModelError, Result};

/// Parsed model from schema-aware FlatBuffer navigation.
#[derive(Debug, Clone)]
pub struct ParsedModel {
    /// SDK version string.
    pub version: String,
    /// Layer entries with properties.
    pub layers: Vec<ParsedLayer>,
    /// Raw decompressed FlatBuffer bytes (for weight access).
    pub raw_flatbuffer: Vec<u8>,
}

/// A layer extracted via schema-aware parsing.
#[derive(Debug, Clone)]
pub struct ParsedLayer {
    /// Layer name (e.g. "conv_0", "dense_1").
    pub name: String,
    /// Key-value properties.
    pub properties: Vec<(String, String)>,
}

/// Parse a decompressed FlatBuffer payload using schema-aware navigation.
///
/// This replaces the heuristic scanning in `parser.rs` with proper
/// vtable traversal.
pub fn parse_flatbuffer(data: &[u8]) -> Result<ParsedModel> {
    if data.len() < 8 {
        return Err(AkidaModelError::parse_error("FlatBuffer too small"));
    }

    let root_off = read_u32(data, 0)? as usize;
    if root_off >= data.len() {
        return Err(AkidaModelError::parse_error("Root offset out of bounds"));
    }

    // Navigate vtable for root table
    let vtable = read_vtable(data, root_off)?;

    // Field 2: version string
    let version = if let Some(field_off) = vtable_field(&vtable, 2) {
        let abs = root_off + field_off;
        read_string_indirect(data, abs)?
    } else {
        // Fall back to scanning if vtable field is absent
        scan_version_string(data)
            .unwrap_or_else(|| "unknown".to_string())
    };

    // Field 0: model body sub-table (contains layers)
    let layers = if let Some(field_off) = vtable_field(&vtable, 0) {
        let body_ref = root_off + field_off;
        let body_off = body_ref + read_u32(data, body_ref)? as usize;
        parse_model_body(data, body_off)?
    } else {
        // No model body — try heuristic layer name extraction
        extract_layer_names_from_strings(data)
    };

    Ok(ParsedModel {
        version,
        layers,
        raw_flatbuffer: data.to_vec(),
    })
}

/// Parse a complete `.fbz` file (decompress + parse).
pub fn parse_fbz(data: &[u8]) -> Result<ParsedModel> {
    let (payload, _compressed) = crate::parser::decompress_fbz(data)?;
    parse_flatbuffer(&payload)
}

fn parse_model_body(data: &[u8], body_off: usize) -> Result<Vec<ParsedLayer>> {
    let vtable = read_vtable(data, body_off)?;

    // Field 0 of body: layers vector
    let Some(layers_field) = vtable_field(&vtable, 0) else {
        return Ok(Vec::new());
    };

    let vec_ref = body_off + layers_field;
    let vec_off = vec_ref + read_u32(data, vec_ref)? as usize;

    read_table_vector(data, vec_off)
}

fn read_table_vector(data: &[u8], vec_off: usize) -> Result<Vec<ParsedLayer>> {
    let count = read_u32(data, vec_off)? as usize;
    let mut layers = Vec::with_capacity(count);

    for i in 0..count {
        let entry_ref = vec_off + 4 + i * 4;
        if entry_ref + 4 > data.len() {
            break;
        }
        let entry_off = entry_ref + read_u32(data, entry_ref)? as usize;
        if entry_off >= data.len() {
            break;
        }

        match parse_layer_entry(data, entry_off) {
            Ok(layer) => layers.push(layer),
            Err(_) => break,
        }
    }

    Ok(layers)
}

fn parse_layer_entry(data: &[u8], entry_off: usize) -> Result<ParsedLayer> {
    let vtable = read_vtable(data, entry_off)?;

    // Field 0: layer name
    let name = if let Some(off) = vtable_field(&vtable, 0) {
        let abs = entry_off + off;
        read_string_indirect(data, abs).unwrap_or_default()
    } else {
        String::new()
    };

    // Field 1: properties vector
    let properties = if let Some(off) = vtable_field(&vtable, 1) {
        let vec_ref = entry_off + off;
        if vec_ref + 4 <= data.len() {
            let vec_off = vec_ref + read_u32(data, vec_ref)? as usize;
            read_kv_vector(data, vec_off).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Ok(ParsedLayer { name, properties })
}

fn read_kv_vector(data: &[u8], vec_off: usize) -> Result<Vec<(String, String)>> {
    let count = read_u32(data, vec_off)? as usize;
    let mut pairs = Vec::with_capacity(count);

    for i in 0..count {
        let entry_ref = vec_off + 4 + i * 4;
        if entry_ref + 4 > data.len() {
            break;
        }
        let entry_off = entry_ref + read_u32(data, entry_ref)? as usize;
        if entry_off >= data.len() {
            break;
        }

        let vtable = match read_vtable(data, entry_off) {
            Ok(v) => v,
            Err(_) => break,
        };

        let key = if let Some(off) = vtable_field(&vtable, 0) {
            let abs = entry_off + off;
            read_string_indirect(data, abs).unwrap_or_default()
        } else {
            continue;
        };

        let value = if let Some(off) = vtable_field(&vtable, 1) {
            let abs = entry_off + off;
            // Try reading as string first, then as integer
            if let Ok(s) = read_string_indirect(data, abs) {
                s
            } else if abs + 8 <= data.len() {
                let v = i64::from_le_bytes([
                    data[abs], data[abs + 1], data[abs + 2], data[abs + 3],
                    data[abs + 4], data[abs + 5], data[abs + 6], data[abs + 7],
                ]);
                v.to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        pairs.push((key, value));
    }

    Ok(pairs)
}

// ── FlatBuffer vtable helpers ────────────────────────────────────────────────

struct VTable {
    vtable_size: usize,
    _table_size: usize,
    field_offsets: Vec<u16>,
}

fn read_vtable(data: &[u8], table_off: usize) -> Result<VTable> {
    if table_off + 4 > data.len() {
        return Err(AkidaModelError::parse_error("vtable ref out of bounds"));
    }

    let soff = i32::from_le_bytes([
        data[table_off],
        data[table_off + 1],
        data[table_off + 2],
        data[table_off + 3],
    ]);

    let vtable_off = (table_off as i64 - soff as i64) as usize;
    if vtable_off + 4 > data.len() {
        return Err(AkidaModelError::parse_error("vtable out of bounds"));
    }

    let vtable_size = u16::from_le_bytes([data[vtable_off], data[vtable_off + 1]]) as usize;
    let table_size = u16::from_le_bytes([data[vtable_off + 2], data[vtable_off + 3]]) as usize;

    let n_fields = vtable_size.saturating_sub(4) / 2;
    let mut field_offsets = Vec::with_capacity(n_fields);

    for i in 0..n_fields {
        let pos = vtable_off + 4 + i * 2;
        if pos + 2 > data.len() {
            break;
        }
        field_offsets.push(u16::from_le_bytes([data[pos], data[pos + 1]]));
    }

    Ok(VTable {
        vtable_size,
        _table_size: table_size,
        field_offsets,
    })
}

fn vtable_field(vtable: &VTable, field_idx: usize) -> Option<usize> {
    vtable
        .field_offsets
        .get(field_idx)
        .copied()
        .filter(|&off| off != 0)
        .map(|off| off as usize)
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(AkidaModelError::parse_error(format!(
            "u32 read at {offset:#x} exceeds buffer ({} bytes)",
            data.len()
        )));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_string_indirect(data: &[u8], offset: usize) -> Result<String> {
    let str_off = offset + read_u32(data, offset)? as usize;
    if str_off + 4 > data.len() {
        return Err(AkidaModelError::parse_error("string offset out of bounds"));
    }

    let len = read_u32(data, str_off)? as usize;
    let str_start = str_off + 4;
    if str_start + len > data.len() {
        return Err(AkidaModelError::parse_error("string data out of bounds"));
    }

    std::str::from_utf8(&data[str_start..str_start + len])
        .map(|s| s.to_string())
        .map_err(|e| AkidaModelError::parse_error(format!("invalid UTF-8: {e}")))
}

// ── Fallback heuristics (for models that don't match observed schema) ────────

fn scan_version_string(data: &[u8]) -> Option<String> {
    let search_end = data.len().min(0x200);
    for offset in 0x08..search_end {
        if offset + 10 > data.len() {
            break;
        }
        let slice = &data[offset..];
        if slice[0].is_ascii_digit() && slice[1] == b'.' {
            if let Some(end) = slice.iter().position(|&b| b == 0) {
                if end > 3 && end < 20 {
                    if let Ok(s) = std::str::from_utf8(&slice[..end]) {
                        if s.chars().filter(|&c| c == '.').count() >= 1
                            && s.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-')
                        {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_layer_names_from_strings(data: &[u8]) -> Vec<ParsedLayer> {
    let names = crate::parser::extract_layer_names(data);
    names
        .into_iter()
        .map(|name| ParsedLayer {
            name,
            properties: Vec::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{LayerDescriptor, ModelDescriptor};

    fn make_test_fbz() -> Vec<u8> {
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
        crate::schema::build_fbz(&desc)
    }

    #[test]
    fn schema_parser_extracts_version() {
        let fbz = make_test_fbz();
        let parsed = parse_fbz(&fbz).expect("parse");
        assert_eq!(parsed.version, "2.19.1");
    }

    #[test]
    fn schema_parser_extracts_layers() {
        let fbz = make_test_fbz();
        let parsed = parse_fbz(&fbz).expect("parse");
        assert_eq!(parsed.layers.len(), 2);
        assert_eq!(parsed.layers[0].name, "input_conv");
        assert_eq!(parsed.layers[1].name, "fc_output");
    }

    #[test]
    fn schema_parser_extracts_layer_properties() {
        let fbz = make_test_fbz();
        let parsed = parse_fbz(&fbz).expect("parse");

        let layer0 = &parsed.layers[0];
        assert!(
            layer0
                .properties
                .iter()
                .any(|(k, _)| k == "layer_type"),
            "expected layer_type property"
        );
    }

    #[test]
    fn schema_parser_handles_empty_model() {
        let desc = ModelDescriptor::new("2.19.1");
        let fbz = crate::schema::build_fbz(&desc);
        let parsed = parse_fbz(&fbz).expect("parse empty");
        assert_eq!(parsed.version, "2.19.1");
        assert!(parsed.layers.is_empty());
    }

    #[test]
    fn schema_parser_rejects_tiny_buffer() {
        assert!(parse_flatbuffer(&[0u8; 4]).is_err());
    }

    #[test]
    fn round_trip_build_parse_verify() {
        let mut desc = ModelDescriptor::new("2.19.1");
        desc.add_layer(
            LayerDescriptor::new("conv_0")
                .with_str("layer_type", "InputConv")
                .with_int("channels", 50)
                .with_int("kernel_size", 1),
        );
        desc.add_layer(
            LayerDescriptor::new("dense_1")
                .with_str("layer_type", "FullyConnected")
                .with_int("units", 128),
        );
        desc.add_layer(
            LayerDescriptor::new("dense_2")
                .with_str("layer_type", "FullyConnected")
                .with_int("units", 1),
        );

        let fbz = crate::schema::build_fbz(&desc);

        // Parse with schema-aware parser
        let parsed = parse_fbz(&fbz).expect("schema parse");
        assert_eq!(parsed.version, "2.19.1");
        assert_eq!(parsed.layers.len(), 3);
        assert_eq!(parsed.layers[0].name, "conv_0");
        assert_eq!(parsed.layers[1].name, "dense_1");
        assert_eq!(parsed.layers[2].name, "dense_2");

        // Also verify the old parser still works
        let model = crate::Model::from_bytes(&fbz).expect("legacy parse");
        assert_eq!(model.version(), "2.19.1");
    }
}
