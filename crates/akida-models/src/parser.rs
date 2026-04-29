// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binary parser for Akida `.fbz` files.
//!
//! `.fbz` files are **Snappy-compressed FlatBuffers**. The first bytes encode
//! the uncompressed payload size as a Snappy varint; the remainder is
//! Snappy-compressed data that decompresses to a standard FlatBuffer binary.
//!
//! The parser also accepts raw (uncompressed) FlatBuffer data for hand-built
//! test models produced by `ProgramBuilder`.

use crate::error::{AkidaModelError, Result};

/// Legacy magic bytes from early hand-built test models.
/// Real model-zoo `.fbz` files do NOT start with these bytes — they start
/// with a Snappy varint. Retained only for backward compatibility with
/// existing test fixtures.
pub const LEGACY_TEST_MAGIC: [u8; 4] = [0x80, 0x44, 0x04, 0x10];

/// Parsed model header.
#[derive(Debug, Clone)]
pub struct ModelHeader {
    /// SDK version string (e.g., "2.19.1").
    pub version: String,

    /// Number of layers.
    pub layer_count: usize,
}

/// Decompress a `.fbz` file if it is Snappy-compressed, otherwise return
/// the data as-is.
///
/// Returns `(payload, was_compressed)`.
pub fn decompress_fbz(data: &[u8]) -> Result<(Vec<u8>, bool)> {
    if data.is_empty() {
        return Err(AkidaModelError::parse_error("Empty file"));
    }

    // Try Snappy block decompression. Real `.fbz` files start with a
    // Snappy varint encoding the uncompressed size.
    match snap::raw::Decoder::new().decompress_vec(data) {
        Ok(decompressed) => {
            tracing::debug!(
                "Snappy decompression: {} -> {} bytes",
                data.len(),
                decompressed.len()
            );
            Ok((decompressed, true))
        }
        Err(_) => {
            tracing::debug!("Not Snappy-compressed, treating as raw FlatBuffer data");
            Ok((data.to_vec(), false))
        }
    }
}

/// Parse model header from `.fbz` file data.
///
/// Handles both Snappy-compressed `.fbz` files (model zoo) and raw
/// FlatBuffer data (hand-built test models).
///
/// # Errors
///
/// Returns error if decompression fails or the decompressed data cannot
/// be parsed.
pub fn parse_header(data: &[u8]) -> Result<ModelHeader> {
    tracing::debug!("Parsing model header ({} bytes)", data.len());

    if data.len() < 16 {
        return Err(AkidaModelError::parse_error("File too small"));
    }

    // Decompress if needed
    let (payload, compressed) = decompress_fbz(data)?;

    if payload.len() < 8 {
        return Err(AkidaModelError::parse_error(
            "Decompressed payload too small for FlatBuffer",
        ));
    }

    if compressed {
        tracing::debug!("Parsing decompressed FlatBuffer ({} bytes)", payload.len());
    }

    // FlatBuffer format: bytes [0..4] are the root table offset (u32 LE).
    // Validate that it points within the buffer.
    let root_offset =
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;

    if root_offset == 0 || root_offset >= payload.len() {
        // Fall back to legacy magic check for hand-built test models
        if data[0..4] == LEGACY_TEST_MAGIC {
            tracing::debug!("Legacy test-model magic detected");
            let version = extract_version(data)?;
            let layer_count = estimate_layer_count(data);
            return Ok(ModelHeader {
                version,
                layer_count,
            });
        }

        return Err(AkidaModelError::InvalidHeader);
    }

    tracing::debug!("FlatBuffer root table offset: {:#x}", root_offset);

    let version = extract_version(&payload)?;
    tracing::info!("Model version: {}", version);

    let layer_count = estimate_layer_count(&payload);
    tracing::debug!("Estimated {} layers", layer_count);

    Ok(ModelHeader {
        version,
        layer_count,
    })
}

/// Extract version string from model data.
///
/// Scans offsets 0x10..0x80 for a pattern like `"2.19.1\0"`.
/// The version string offset varies across models (observed: 33-35 in
/// raw `.fbz`, variable in decompressed FlatBuffer).
fn extract_version(data: &[u8]) -> Result<String> {
    // Widen search window to cover observed offsets in both raw and
    // decompressed data
    let search_end = data.len().min(0x200);

    for offset in 0x08..search_end {
        if offset + 10 > data.len() {
            break;
        }
        if let Some(version) = try_extract_version_at(data, offset) {
            return Ok(version);
        }
    }

    Err(AkidaModelError::parse_error("Version string not found"))
}

/// Try to extract version string at specific offset.
fn try_extract_version_at(data: &[u8], offset: usize) -> Option<String> {
    let slice = &data[offset..];

    if slice.len() < 8 {
        return None;
    }

    // Look for pattern: digit.digit(s).digit(s)\0
    if slice[0].is_ascii_digit() && slice[1] == b'.' {
        if let Some(end) = slice.iter().position(|&b| b == 0)
            && end > 3
            && end < 20
            && let Ok(s) = std::str::from_utf8(&slice[..end])
        {
            if s.chars().filter(|&c| c == '.').count() >= 1
                && s.chars()
                    .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
            {
                return Some(s.to_string());
            }
        }
    }

    None
}

/// Count layers by scanning for `layer_type` metadata markers, then
/// falling back to FlatBuffer vector-length probing.
fn estimate_layer_count(data: &[u8]) -> usize {
    let marker_count = data
        .windows(b"layer_type".len())
        .filter(|w| *w == b"layer_type")
        .count();

    if marker_count > 0 {
        return marker_count;
    }

    // FlatBuffer root table traversal: root offset at bytes [0..4].
    if data.len() >= 12 {
        let root_offset = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if root_offset > 0 && root_offset < data.len().saturating_sub(8) {
            for probe in
                (root_offset..data.len().saturating_sub(4).min(root_offset + 64)).step_by(4)
            {
                let candidate = u32::from_le_bytes([
                    data[probe],
                    data[probe + 1],
                    data[probe + 2],
                    data[probe + 3],
                ]);
                if (1..128).contains(&candidate) {
                    let vec_end = probe + 4 + candidate as usize * 4;
                    if vec_end <= data.len() {
                        return candidate as usize;
                    }
                }
            }
        }
    }

    1
}

/// Extract layer names from model data.
///
/// Strategy: FlatBuffers-style length-prefixed string scan, then linear
/// scan for null-terminated ASCII strings matching layer name patterns.
pub fn extract_layer_names(data: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for i in (0..data.len().saturating_sub(8)).step_by(4) {
        let len_bytes = &data[i..i + 4];
        let str_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        if (2..32).contains(&str_len) && i + 4 + str_len < data.len() {
            let str_data = &data[i + 4..i + 4 + str_len];
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

/// Try to extract null-terminated string at offset.
fn try_extract_string_at(data: &[u8], offset: usize) -> Option<String> {
    let slice = &data[offset..];

    if let Some(end) = slice.iter().take(32).position(|&b| b == 0)
        && end > 2
        && end < 20
        && let Ok(s) = std::str::from_utf8(&slice[..end])
    {
        if s.chars().all(|c| c.is_ascii() && !c.is_control()) {
            return Some(s.to_string());
        }
    }

    None
}

/// Check if string looks like a valid layer name.
fn is_valid_layer_name(s: &str) -> bool {
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return false;
    }

    if s.len() < 2 || s.len() > 32 {
        return false;
    }

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
    fn legacy_magic_bytes_constant_preserved() {
        assert_eq!(LEGACY_TEST_MAGIC, [0x80, 0x44, 0x04, 0x10]);
    }

    #[test]
    fn test_version_extraction_from_legacy_model() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&LEGACY_TEST_MAGIC);
        let ver = b"2.18.2\0";
        data[30..30 + ver.len()].copy_from_slice(ver);

        let header = parse_header(&data).unwrap();
        assert_eq!(header.version, "2.18.2");
    }

    #[test]
    fn parse_header_rejects_tiny_buffer() {
        let data = [0u8; 8];
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn decompress_fbz_passes_through_raw_flatbuffer() {
        let mut data = vec![0u8; 64];
        // Set a plausible root table offset
        data[0..4].copy_from_slice(&16u32.to_le_bytes());
        let (payload, compressed) = decompress_fbz(&data).unwrap();
        assert!(!compressed);
        assert_eq!(payload, data);
    }

    #[test]
    fn decompress_fbz_handles_snappy_data() {
        let original = b"Hello, this is test data for Snappy compression roundtrip!";
        let compressed = snap::raw::Encoder::new()
            .compress_vec(original)
            .expect("compress");

        let (decompressed, was_compressed) = decompress_fbz(&compressed).unwrap();
        assert!(was_compressed);
        assert_eq!(decompressed, original);
    }

    #[test]
    fn parse_header_with_snappy_compressed_flatbuffer() {
        // Build a minimal FlatBuffer-like payload with a version string
        let mut payload = vec![0u8; 256];
        // Root table offset at bytes [0..4]
        payload[0..4].copy_from_slice(&32u32.to_le_bytes());
        // Version string "2.19.1\0" at offset 0x21 (33)
        let ver = b"2.19.1\0";
        payload[33..33 + ver.len()].copy_from_slice(ver);

        // Snappy-compress it
        let compressed = snap::raw::Encoder::new()
            .compress_vec(&payload)
            .expect("compress");

        let header = parse_header(&compressed).unwrap();
        assert_eq!(header.version, "2.19.1");
        assert!(header.layer_count >= 1);
    }

    #[test]
    fn parse_header_rejects_invalid_data() {
        // Data that's neither valid Snappy nor a valid FlatBuffer
        let data = [0xFF; 128];
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn extract_layer_names_returns_empty_for_random_bytes() {
        let data = [0xABu8; 256];
        assert!(extract_layer_names(&data).is_empty());
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
    fn estimate_layer_count_uses_root_table_vector_length() {
        let mut data = vec![0u8; 256];
        // Root table offset
        data[0..4].copy_from_slice(&16u32.to_le_bytes());
        // Vector length at root table offset
        data[16..20].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(super::estimate_layer_count(&data), 4);
    }

    #[test]
    fn estimate_layer_count_ignores_vector_that_exceeds_buffer() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&16u32.to_le_bytes());
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

    #[test]
    fn parse_header_counts_layer_type_markers_in_decompressed() {
        let mut payload = vec![0u8; 128];
        payload[0..4].copy_from_slice(&32u32.to_le_bytes());
        payload[20..27].copy_from_slice(b"2.18.2\0");
        payload[40..50].copy_from_slice(b"layer_type");
        payload[60..70].copy_from_slice(b"layer_type");

        let compressed = snap::raw::Encoder::new()
            .compress_vec(&payload)
            .expect("compress");

        let header = parse_header(&compressed).unwrap();
        assert!(header.layer_count >= 2);
    }
}
