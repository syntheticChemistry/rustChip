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
        if let Some(end) = slice.iter().position(|&b| b == 0) {
            if end > 3 && end < 20 {
                if let Ok(s) = std::str::from_utf8(&slice[..end]) {
                    // Validate it looks like a version string
                    if s.chars().filter(|&c| c == '.').count() >= 1 {
                        return Some(s.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Estimate number of layers (simplified heuristic)
fn estimate_layer_count(data: &[u8]) -> usize {
    // Count occurrences of "layer_type" string as a proxy
    let pattern = b"layer_type";

    data.windows(pattern.len())
        .filter(|window| *window == pattern)
        .count()
        .max(1) // At least 1 layer
}

/// Extract layer names from model data
pub fn extract_layer_names(data: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Common layer name patterns: "input", "fc", "conv", etc.
    // They appear as null-terminated strings in the metadata section

    let mut i = 0;
    while i + 20 < data.len() {
        // Look for potential string markers
        if data[i].is_ascii_alphabetic() {
            if let Some(name) = try_extract_string_at(data, i) {
                if is_valid_layer_name(&name) && !seen.contains(&name) {
                    tracing::debug!("Found layer name: {}", name);
                    names.push(name.clone());
                    seen.insert(name);
                }
            }
        }
        i += 1;
    }

    names
}

/// Try to extract null-terminated string at offset
fn try_extract_string_at(data: &[u8], offset: usize) -> Option<String> {
    let slice = &data[offset..];

    // Find null terminator within reasonable distance
    if let Some(end) = slice.iter().take(32).position(|&b| b == 0) {
        if end > 2 && end < 20 {
            if let Ok(s) = std::str::from_utf8(&slice[..end]) {
                // Check if it's ASCII and reasonable
                if s.chars().all(|c| c.is_ascii() && !c.is_control()) {
                    return Some(s.to_string());
                }
            }
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
}
