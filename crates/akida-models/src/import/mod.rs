// SPDX-License-Identifier: AGPL-3.0-or-later

//! Weight import from non-Python sources.
//!
//! Reads weight tensors from `.npy`, `.safetensors`, and raw `f32` slices,
//! producing `ImportedWeights` ready for quantization via [`crate::quantize`]
//! and serialization via [`crate::builder::ProgramBuilder`].
//!
//! The `.npy` parser is hand-rolled (~60 lines of header parsing) to avoid
//! an external dependency. The safetensors path uses the `safetensors` crate.
//!
//! ## Lineage
//!
//! Pattern absorbed from `neuralSpring/src/weight_loader.rs` (safetensors
//! deserialization, dtype upcast, tensor extraction).

pub mod onnx;

use crate::error::{AkidaModelError, Result};
use std::path::Path;

/// A weight tensor imported from an external source.
#[derive(Debug, Clone)]
pub struct ImportedWeights {
    /// Tensor name (from safetensors key or npy filename).
    pub name: String,
    /// Flattened weight values as f32.
    pub data: Vec<f32>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
}

impl ImportedWeights {
    /// Construct from a raw f32 slice (the trivial import path).
    #[must_use]
    pub fn from_raw(name: impl Into<String>, data: &[f32], shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            data: data.to_vec(),
            shape,
        }
    }

    /// Total number of weight values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── .npy import ──────────────────────────────────────────────────────────────

/// Load a single tensor from a `.npy` file.
///
/// Supports `float32` and `float64` dtypes in little-endian byte order,
/// which covers the vast majority of weight files exported from PyTorch
/// and NumPy.
///
/// # Errors
///
/// Returns error if the file cannot be read, the magic is wrong, or the
/// dtype is unsupported.
pub fn load_npy(path: &Path) -> Result<ImportedWeights> {
    let data = std::fs::read(path).map_err(|e| {
        AkidaModelError::parse_error(format!("read {}: {e}", path.display()))
    })?;
    parse_npy(&data, path.file_stem().and_then(|s| s.to_str()).unwrap_or("npy"))
}

/// Parse `.npy` bytes in memory.
fn parse_npy(data: &[u8], name: &str) -> Result<ImportedWeights> {
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(AkidaModelError::parse_error("not a .npy file (bad magic)"));
    }

    let major = data[6];
    let _minor = data[7];

    let (header_len, header_start) = if major == 1 {
        let len = u16::from_le_bytes([data[8], data[9]]) as usize;
        (len, 10)
    } else if major >= 2 {
        if data.len() < 12 {
            return Err(AkidaModelError::parse_error("npy v2 header truncated"));
        }
        let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        (len, 12)
    } else {
        return Err(AkidaModelError::parse_error(format!(
            "unsupported npy version {major}"
        )));
    };

    let header_end = header_start + header_len;
    if header_end > data.len() {
        return Err(AkidaModelError::parse_error("npy header exceeds file size"));
    }

    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| AkidaModelError::parse_error(format!("npy header not UTF-8: {e}")))?;

    let dtype = extract_npy_descr(header)?;
    let shape = extract_npy_shape(header)?;
    let payload = &data[header_end..];

    let weights = match dtype.as_str() {
        "<f4" | "float32" => read_f32_le(payload),
        "<f8" | "float64" => read_f64_le_as_f32(payload),
        _ => {
            return Err(AkidaModelError::parse_error(format!(
                "unsupported npy dtype: {dtype}"
            )));
        }
    };

    Ok(ImportedWeights {
        name: name.to_string(),
        data: weights,
        shape,
    })
}

fn extract_npy_descr(header: &str) -> Result<String> {
    // header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
    let descr_start = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| AkidaModelError::parse_error("npy header missing 'descr'"))?;

    let after_key = &header[descr_start + 7..];
    let val_start = after_key.find(['\'', '"'])
        .ok_or_else(|| AkidaModelError::parse_error("npy descr: no opening quote"))?;
    let quote_char = after_key.as_bytes()[val_start];
    let val_body = &after_key[val_start + 1..];
    let val_end = val_body
        .find(quote_char as char)
        .ok_or_else(|| AkidaModelError::parse_error("npy descr: no closing quote"))?;

    Ok(val_body[..val_end].to_string())
}

fn extract_npy_shape(header: &str) -> Result<Vec<usize>> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| AkidaModelError::parse_error("npy header missing 'shape'"))?;

    let after_key = &header[shape_start + 7..];
    let paren_open = after_key
        .find('(')
        .ok_or_else(|| AkidaModelError::parse_error("npy shape: no opening paren"))?;
    let paren_close = after_key
        .find(')')
        .ok_or_else(|| AkidaModelError::parse_error("npy shape: no closing paren"))?;

    let shape_str = &after_key[paren_open + 1..paren_close];
    if shape_str.trim().is_empty() {
        return Ok(vec![]); // scalar
    }

    shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| AkidaModelError::parse_error(format!("npy shape parse: {e}")))
        })
        .collect()
}

fn read_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f64_le_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(8)
        .map(|c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        })
        .collect()
}

// ── .safetensors import ──────────────────────────────────────────────────────

/// Load all tensors from a `.safetensors` file as f32.
///
/// # Errors
///
/// Returns error if the file cannot be read, is not valid safetensors,
/// or contains unsupported dtypes.
pub fn load_safetensors(path: &Path) -> Result<Vec<ImportedWeights>> {
    let raw = std::fs::read(path).map_err(|e| {
        AkidaModelError::parse_error(format!("read {}: {e}", path.display()))
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| AkidaModelError::parse_error(format!("safetensors parse: {e}")))?;

    let mut result = Vec::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = upcast_to_f32(&view).map_err(|e| {
            AkidaModelError::parse_error(format!("tensor '{name}': {e}"))
        })?;

        result.push(ImportedWeights {
            name,
            data,
            shape,
        });
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

/// Load a single named tensor from a `.safetensors` file.
///
/// # Errors
///
/// Returns error if the file or tensor is not found.
pub fn load_safetensors_tensor(path: &Path, tensor_name: &str) -> Result<ImportedWeights> {
    let raw = std::fs::read(path).map_err(|e| {
        AkidaModelError::parse_error(format!("read {}: {e}", path.display()))
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| AkidaModelError::parse_error(format!("safetensors parse: {e}")))?;

    let view = tensors
        .tensor(tensor_name)
        .map_err(|e| AkidaModelError::parse_error(format!("tensor '{tensor_name}': {e}")))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let data = upcast_to_f32(&view).map_err(|e| {
        AkidaModelError::parse_error(format!("tensor '{tensor_name}': {e}"))
    })?;

    Ok(ImportedWeights {
        name: tensor_name.to_string(),
        data,
        shape,
    })
}

/// List tensor names and shapes in a `.safetensors` file.
///
/// # Errors
///
/// Returns error if the file cannot be parsed.
pub fn list_safetensors(path: &Path) -> Result<Vec<(String, Vec<usize>, String)>> {
    let raw = std::fs::read(path).map_err(|e| {
        AkidaModelError::parse_error(format!("read {}: {e}", path.display()))
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| AkidaModelError::parse_error(format!("safetensors parse: {e}")))?;

    let mut result: Vec<(String, Vec<usize>, String)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let shape = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            (name, shape, dtype)
        })
        .collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

fn upcast_to_f32(view: &safetensors::tensor::TensorView<'_>) -> std::result::Result<Vec<f32>, String> {
    use safetensors::Dtype;
    let bytes = view.data();

    match view.dtype() {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err("F32 data length not aligned".into());
            }
            Ok(read_f32_le(bytes))
        }
        Dtype::F64 => {
            if bytes.len() % 8 != 0 {
                return Err("F64 data length not aligned".into());
            }
            Ok(read_f64_le_as_f32(bytes))
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err("F16 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err("BF16 data length not aligned".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        other => Err(format!("unsupported dtype: {other:?}")),
    }
}

/// IEEE 754 half-precision (binary16) to single-precision.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut shift = 0_u32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            shift += 1;
        }
        f &= 0x3FF;
        let exp32 = 127 - 15 + 1 - shift;
        f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
        }
    } else {
        let exp32 = exp + (127 - 15);
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

/// bfloat16 to single-precision.
const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ── Build .npy bytes (for testing) ───────────────────────────────────────────

/// Build a minimal `.npy` file for f32 data (used in tests and model export).
#[must_use]
pub fn build_npy_f32(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", inner.join(", "))
    };

    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    );

    // Pad header to 64-byte alignment (including magic + version + header_len)
    let preamble_len = 10; // magic(6) + major(1) + minor(1) + header_len(2)
    let total_unpadded = preamble_len + header_dict.len() + 1; // +1 for \n
    let padding = (64 - (total_unpadded % 64)) % 64;
    let header_len = header_dict.len() + padding + 1; // include \n

    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1); // major
    buf.push(0); // minor
    buf.extend_from_slice(&(header_len as u16).to_le_bytes());
    buf.extend_from_slice(header_dict.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');

    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Raw import ──────────────────────────────────────────────────────

    #[test]
    fn from_raw_preserves_data() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let w = ImportedWeights::from_raw("test", &data, vec![3]);
        assert_eq!(w.len(), 3);
        assert!(!w.is_empty());
        assert_eq!(w.name, "test");
        assert_eq!(w.shape, vec![3]);
    }

    // ── .npy round-trip ─────────────────────────────────────────────────

    #[test]
    fn npy_f32_round_trip() {
        let original = vec![1.0_f32, -2.5, 3.14, 0.0];
        let npy_bytes = build_npy_f32(&original, &[2, 2]);

        let parsed = parse_npy(&npy_bytes, "test").expect("parse npy");
        assert_eq!(parsed.shape, vec![2, 2]);
        assert_eq!(parsed.data, original);
    }

    #[test]
    fn npy_1d_round_trip() {
        let original = vec![42.0_f32];
        let npy_bytes = build_npy_f32(&original, &[1]);

        let parsed = parse_npy(&npy_bytes, "scalar").expect("parse npy 1d");
        assert_eq!(parsed.shape, vec![1]);
        assert_eq!(parsed.data, original);
    }

    #[test]
    fn npy_rejects_bad_magic() {
        let data = vec![0u8; 64];
        assert!(parse_npy(&data, "bad").is_err());
    }

    #[test]
    fn npy_rejects_truncated() {
        let data = b"\x93NUMPY";
        assert!(parse_npy(data, "trunc").is_err());
    }

    // ── .safetensors ────────────────────────────────────────────────────

    #[test]
    fn safetensors_file_not_found() {
        let err = load_safetensors(Path::new("/nonexistent/model.safetensors"));
        assert!(err.is_err());
    }

    #[test]
    fn safetensors_tensor_not_found() {
        let err = load_safetensors_tensor(
            Path::new("/nonexistent/model.safetensors"),
            "layer0",
        );
        assert!(err.is_err());
    }

    #[test]
    fn list_safetensors_not_found() {
        let err = list_safetensors(Path::new("/nonexistent/model.safetensors"));
        assert!(err.is_err());
    }

    // ── f16/bf16 conversion ─────────────────────────────────────────────

    #[test]
    fn f16_one() {
        let one = f16_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f16_infinity() {
        assert!(f16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn f16_subnormal() {
        let tiny = f16_to_f32(0x0001);
        assert!(tiny > 0.0 && tiny < 1e-4);
    }

    #[test]
    fn bf16_one() {
        let one = bf16_to_f32(0x3F80);
        assert!((one - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    // ── build_npy_f32 ───────────────────────────────────────────────────

    #[test]
    fn build_npy_header_alignment() {
        let npy = build_npy_f32(&[1.0], &[1]);
        // Header must end at a 64-byte boundary
        let header_len = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        let total_header = 10 + header_len;
        assert_eq!(total_header % 64, 0);
    }
}
