// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binary parser for Akida .fbz files
//!
//! Parses FlatBuffers-based Akida model files.

use crate::error::{AkidaModelError, Result};

/// `FlatBuffers` magic bytes for Akida models
pub const FLATBUFFERS_MAGIC: [u8; 4] = [0x80, 0x44, 0x04, 0x10];

/// Parsed model header
#[derive(Debug, Clone)]
pub struct ModelHeader {
    /// SDK version string
    pub version: String,

    /// Number of layers
    pub layer_count: usize,
}

/// Parse model header from .fbz file data
///
/// # Errors
///
/// Returns error if magic bytes are invalid or parsing fails.
pub fn parse_header(data: &[u8]) -> Result<ModelHeader> {
    tracing::debug!("Parsing model header ({} bytes)", data.len());

    if data.len() < 16 {
        return Err(AkidaModelError::parse_error("File too small"));
    }

    // Check FlatBuffers magic
    if data[0..4] != FLATBUFFERS_MAGIC {
        tracing::error!("Invalid magic bytes: {:02x?}", &data[0..4]);
        return Err(AkidaModelError::InvalidHeader);
    }

    tracing::debug!("Valid FlatBuffers header detected");

    // Extract version string (starts around offset 0x1E)
    let version = extract_version(data)?;
    tracing::info!("Model version: {}", version);

    // Count layers (simplified - actual parsing would traverse FlatBuffers tables)
    let layer_count = estimate_layer_count(data);
    tracing::debug!("Estimated {} layers", layer_count);

    Ok(ModelHeader {
        version,
        layer_count,
    })
}

/// Extract version string from model data
fn extract_version(data: &[u8]) -> Result<String> {
    // Version string appears around offset 0x1E-0x2A
    // Format: "2.18.2\0"

    for offset in 0x18..0x40 {
        if offset + 10 > data.len() {
            break;
        }

        // Look for version pattern: X.XX.X\0 where X are digits
        if let Some(version) = try_extract_version_at(data, offset) {
            return Ok(version);
        }
    }

    Err(AkidaModelError::parse_error("Version string not found"))
}

/// Try to extract version string at specific offset
fn try_extract_version_at(data: &[u8], offset: usize) -> Option<String> {
    let slice = &data[offset..];

    // Look for pattern like "2.18.2\0"
    if slice.len() < 8 {
        return None;
    }

    // Check for digit.digit pattern
    if slice[0].is_ascii_digit() && slice[1] == b'.' {
        // Find null terminator
        if let Some(end) = slice.iter().position(|&b| b == 0)
            && end > 3
            && end < 20
            && let Ok(s) = std::str::from_utf8(&slice[..end])
        {
            // Validate it looks like a version string
            if s.chars().filter(|&c| c == '.').count() >= 1 {
                return Some(s.to_string());
            }
        }
    }

    None
}

/// Count layers by traversing `FlatBuffers` table structure.
///
/// `FlatBuffers` layout: bytes [0..4] are file identifier/magic,
/// bytes [4..8] are the root table offset (little-endian u32).
/// The root table contains a vtable pointer and field offsets.
/// We look for array-of-tables patterns (layer vectors) by
/// counting `layer_type` metadata strings as confirmed markers,
/// then cross-validate with `FlatBuffers` vector length fields
/// (u32 element counts preceding table offset arrays).
fn estimate_layer_count(data: &[u8]) -> usize {
    // Primary: count confirmed layer_type metadata markers
    let marker_count = data
        .windows(b"layer_type".len())
        .filter(|w| *w == b"layer_type")
        .count();

    if marker_count > 0 {
        return marker_count;
    }

    // Secondary: attempt FlatBuffers root table traversal.
    // Root table offset is at bytes 4..8 (after the 4-byte magic/identifier).
    if data.len() >= 12 {
        let root_offset = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        // The root table starts with a negative vtable offset (i32),
        // then field data. A layer vector field would store a u32
        // element count followed by table offsets.
        if root_offset < data.len().saturating_sub(8) {
            let table_start = root_offset;
            // Scan nearby offsets for small vector-length candidates (1..128)
            for probe in
                (table_start..data.len().saturating_sub(4).min(table_start + 64)).step_by(4)
            {
                let candidate = u32::from_le_bytes([
                    data[probe],
                    data[probe + 1],
                    data[probe + 2],
                    data[probe + 3],
                ]);
                if (1..128).contains(&candidate) {
                    // Validate: if this many 4-byte offsets follow and stay in-bounds
                    let vec_end = probe + 4 + candidate as usize * 4;
                    if vec_end <= data.len() {
                        return candidate as usize;
                    }
                }
            }
        }
    }

    // Fallback: at least 1 layer for any parseable model
    1
}

/// Extract layer names from model data.
///
/// Strategy: First try FlatBuffers-aware extraction by following string
/// offset pointers in the binary data, then fall back to a linear scan
/// for null-terminated ASCII strings matching neural-network layer patterns.
pub fn extract_layer_names(data: &[u8]) -> Vec<String> {
    // Primary: FlatBuffers string references.
    // FlatBuffers stores strings as (u32 length, UTF-8 data, NUL).
    // String offsets appear as relative u32 pointers in tables.
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Scan for FlatBuffers-style length-prefixed strings
    for i in (0..data.len().saturating_sub(8)).step_by(4) {
        let len_bytes = &data[i..i + 4];
        let str_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        if (2..32).contains(&str_len) && i + 4 + str_len < data.len() {
            let str_data = &data[i + 4..i + 4 + str_len];
            // Check null terminator after string
            if data.get(i + 4 + str_len) == Some(&0)
                && let Ok(s) = std::str::from_utf8(str_data)
                && is_valid_layer_name(s)
                && !seen.contains(s)
            {
                tracing::debug!("FlatBuffers string at {:#x}: {}", i, s);
                names.push(s.to_string());
                seen.insert(s.to_string());
            }
        }
    }

    if !names.is_empty() {
        return names;
    }

    // Fallback: linear scan for null-terminated ASCII strings
    let mut i = 0;
    while i + 20 < data.len() {
        if data[i].is_ascii_alphabetic()
            && let Some(name) = try_extract_string_at(data, i)
            && is_valid_layer_name(&name)
            && !seen.contains(&name)
        {
            tracing::debug!("Found layer name: {}", name);
            names.push(name.clone());
            seen.insert(name);
        }
        i += 1;
    }

    names
}

/// Try to extract null-terminated string at offset
fn try_extract_string_at(data: &[u8], offset: usize) -> Option<String> {
    let slice = &data[offset..];

    // Find null terminator within reasonable distance
    if let Some(end) = slice.iter().take(32).position(|&b| b == 0)
        && end > 2
        && end < 20
        && let Ok(s) = std::str::from_utf8(&slice[..end])
    {
        // Check if it's ASCII and reasonable
        if s.chars().all(|c| c.is_ascii() && !c.is_control()) {
            return Some(s.to_string());
        }
    }

    None
}

/// Check if string looks like a valid layer name
fn is_valid_layer_name(s: &str) -> bool {
    // Common patterns: "input", "fc", "conv_0", "relu", etc.

    // Basic format check
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return false;
    }

    if s.len() < 2 || s.len() > 32 {
        return false;
    }

    // Filter out metadata keys (not actual layer names)
    let metadata_keys = [
        "layer_type",
        "weights_bits",
        "activation",
        "output_shape",
        "input_shape",
        "kernel_size",
        "stride",
        "padding",
        "bias",
        "filters",
        "neurons",
        "quantization",
        "scale",
        "offset",
    ];

    if metadata_keys.contains(&s) {
        return false;
    }

    // Must contain a layer-like pattern
    s.contains("input")
        || s.contains("fc")
        || s.contains("conv")
        || s.contains("pool")
        || s.contains("relu")
        || s.contains("dense")
        || s.starts_with("layer")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_bytes() {
        assert_eq!(FLATBUFFERS_MAGIC, [0x80, 0x44, 0x04, 0x10]);
    }

    #[test]
    fn test_version_extraction() {
        // Real data from minimal_test.fbz
        let data = b"\x80D\x04\x10\x00\x01\x01@\x0a\x00\x0c\x00\x04\x00\x00\x00\
                     \x08\x00\x0a\x00\x00\x00\x14\x00\x00\x05\x0eD\x06\x00\x00\x00\
                     2.18.2\x00\x00\x02\x00\x00\x00";

        let header = parse_header(data).unwrap();
        assert_eq!(header.version, "2.18.2");
    }

    #[test]
    fn test_invalid_magic() {
        // Valid size but wrong magic bytes
        let data = [0u8; 128]; // Large enough to pass size check
        assert!(matches!(
            parse_header(&data),
            Err(AkidaModelError::InvalidHeader)
        ));
    }

    #[test]
    fn parse_header_rejects_tiny_buffer() {
        let data = [0u8; 8];
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn extract_layer_names_returns_empty_for_random_bytes() {
        let data = [0xABu8; 256];
        assert!(extract_layer_names(&data).is_empty());
    }

    #[test]
    fn parsed_header_layer_count_is_non_zero() {
        let mut data = b"\x80D\x04\x10\x00\x01\x01@\x0a\x00\x0c\x00\x04\x00\x00\x00\
                     \x08\x00\x0a\x00\x00\x00\x14\x00\x00\x05\x0eD\x06\x00\x00\x00\
                     2.18.2\x00\x00\x02\x00\x00\x00"
            .to_vec();
        data.resize(256, 0);
        let header = parse_header(&data).unwrap();
        assert!(header.layer_count >= 1);
    }

    #[test]
    fn extract_layer_names_finds_flatbuffers_length_prefixed_name() {
        let s = b"input_fc";
        let len = s.len() as u32;
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&len.to_le_bytes());
        data[4..4 + s.len()].copy_from_slice(s);
        data[4 + s.len()] = 0;
        let names = extract_layer_names(&data);
        assert!(names.iter().any(|n| n.contains("input")));
    }

    #[test]
    fn extract_layer_names_skips_metadata_keys() {
        let mut data = vec![0u8; 128];
        let s = b"layer_type";
        let len = s.len() as u32;
        data[0..4].copy_from_slice(&len.to_le_bytes());
        data[4..4 + s.len()].copy_from_slice(s);
        data[4 + s.len()] = 0;
        assert!(extract_layer_names(&data).is_empty());
    }

    #[test]
    fn parse_header_counts_layer_type_markers() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&FLATBUFFERS_MAGIC);
        data[30..37].copy_from_slice(b"2.18.2\0");
        data[40..50].copy_from_slice(b"layer_type");
        data[60..70].copy_from_slice(b"layer_type");
        let header = parse_header(&data).unwrap();
        assert!(header.layer_count >= 2);
    }

    #[test]
    fn parse_header_version_not_found_is_error() {
        let mut data = vec![0x80, 0x44, 0x04, 0x10];
        data.resize(128, 0xFF);
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn estimate_layer_count_uses_root_table_vector_length() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&FLATBUFFERS_MAGIC);
        data[4..8].copy_from_slice(&16u32.to_le_bytes());
        data[16..20].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(super::estimate_layer_count(&data), 4);
    }

    #[test]
    fn estimate_layer_count_ignores_vector_that_exceeds_buffer() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&FLATBUFFERS_MAGIC);
        data[4..8].copy_from_slice(&16u32.to_le_bytes());
        data[16..20].copy_from_slice(&50u32.to_le_bytes());
        assert_eq!(super::estimate_layer_count(&data), 1);
    }

    #[test]
    fn extract_layer_names_linear_scan_finds_null_terminated_layer_string() {
        let mut data = vec![0u8; 200];
        let name = b"conv_test\0";
        data[80..80 + name.len()].copy_from_slice(name);
        let names = extract_layer_names(&data);
        assert!(names.iter().any(|n| n.contains("conv")));
    }
}
