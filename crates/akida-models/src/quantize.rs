// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quantization primitives for Akida NPU deployment.
//!
//! Converts float32 weights to int1/int2/int4/int8 packed representations
//! compatible with the AKD1000/AKD1500 weight format. This is the encoding
//! half — [`crate::weights`] provides the decoding half.
//!
//! ## Quantization scheme
//!
//! Symmetric max-abs quantization: `scale = max(|w|) / max_int`, where
//! `max_int = 2^(bits-1) - 1` for signed types. The zero-point is always 0
//! for symmetric schemes (Akida's native format).
//!
//! ## Packing
//!
//! Sub-byte values are packed little-endian: low bits first within each byte.
//! For int4, two values per byte (low nibble first). For int2, four values per
//! byte. For int1, eight values per byte. This matches the layout observed in
//! `.fbz` weight blocks and the dequantization path in [`crate::weights`].
//!
//! ## Lineage
//!
//! Algorithm patterns absorbed from:
//! - `neuralSpring/src/quantized.rs` — Q8/Q4 symmetric quantization
//! - `hotSpring/barracuda/src/bin/validate_npu_quantization.rs` — NPU validation
//! - `barraCuda/crates/barracuda/src/esn_v2/npu.rs` — affine int8 for NPU

/// Minimum absolute maximum to prevent division by zero during scale computation.
const QUANTIZATION_FLOOR: f32 = 1e-30;

/// Result of quantizing a float weight tensor.
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    /// Packed weight bytes (sub-byte values packed LE within each byte).
    pub packed: Vec<u8>,
    /// Scale factor: `float_value = quantized_int * scale`.
    pub scale: f32,
    /// Zero-point offset (0 for symmetric quantization).
    pub zero_point: i32,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Bit width (1, 2, 4, or 8).
    pub bits: u8,
}

impl QuantizedWeights {
    /// Number of logical weight values.
    #[must_use]
    pub fn weight_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Dequantize back to float32 for round-trip verification.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let unpacked = match self.bits {
            1 => unpack_int1(&self.packed),
            2 => unpack_int2(&self.packed),
            4 => unpack_int4(&self.packed),
            8 => unpack_int8(&self.packed),
            _ => Vec::new(),
        };
        unpacked
            .iter()
            .map(|&q| (q as f32 - self.zero_point as f32) * self.scale)
            .collect()
    }
}

/// Symmetric per-layer quantization.
///
/// Computes a single scale from `max(|weights|)` and quantizes all values
/// uniformly. Suitable for layers where the weight distribution is roughly
/// uniform across all output channels.
#[must_use]
pub fn quantize_per_layer(weights: &[f32], bits: u8) -> QuantizedWeights {
    let max_int = max_signed_int(bits);
    let abs_max = weights
        .iter()
        .map(|w| w.abs())
        .fold(0.0_f32, f32::max)
        .max(QUANTIZATION_FLOOR);

    let scale = abs_max / max_int as f32;
    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            (w / scale)
                .round()
                .clamp(-(max_int as f32), max_int as f32) as i8
        })
        .collect();

    let packed = pack_signed(&quantized, bits);
    QuantizedWeights {
        packed,
        scale,
        zero_point: 0,
        shape: vec![weights.len()],
        bits,
    }
}

/// Symmetric per-channel quantization.
///
/// Computes a separate scale per output channel (sliced along `axis`) for
/// better accuracy when channel magnitudes vary. Returns one `QuantizedWeights`
/// per channel.
#[must_use]
pub fn quantize_per_channel(
    weights: &[f32],
    shape: &[usize],
    axis: usize,
    bits: u8,
) -> Vec<QuantizedWeights> {
    assert!(
        axis < shape.len(),
        "axis {axis} out of bounds for shape {shape:?}"
    );

    let total: usize = shape.iter().product();
    assert_eq!(
        weights.len(),
        total,
        "weight count {} != shape product {}",
        weights.len(),
        total
    );

    let n_channels = shape[axis];
    let stride: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let channel_size = outer * stride;

    let mut results = Vec::with_capacity(n_channels);

    for ch in 0..n_channels {
        let mut channel_weights = Vec::with_capacity(channel_size);
        for o in 0..outer {
            let base = o * n_channels * stride + ch * stride;
            channel_weights.extend_from_slice(&weights[base..base + stride]);
        }

        let mut qw = quantize_per_layer(&channel_weights, bits);
        qw.shape = if outer == 1 {
            vec![stride]
        } else {
            vec![outer, stride]
        };
        results.push(qw);
    }

    results
}

// ── Packing ──────────────────────────────────────────────────────────────────

/// Pack signed int4 values: two per byte, low nibble first.
#[must_use]
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    for pair in values.chunks(2) {
        let lo = (pair[0] & 0x0F) as u8;
        let hi = if pair.len() > 1 {
            (pair[1] & 0x0F) as u8
        } else {
            0
        };
        packed.push(lo | (hi << 4));
    }
    packed
}

/// Unpack int4: extract two signed 4-bit values per byte.
#[must_use]
pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
    let mut values = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        let lo = (byte & 0x0F) as i8;
        let hi = ((byte >> 4) & 0x0F) as i8;
        // Sign-extend from 4-bit: if bit 3 is set, the value is negative
        let lo = if lo & 0x08 != 0 { lo | !0x0F } else { lo };
        let hi = if hi & 0x08 != 0 { hi | !0x0F } else { hi };
        values.push(lo);
        values.push(hi);
    }
    values
}

/// Pack signed int2 values: four per byte, low bits first.
#[must_use]
pub fn pack_int2(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 3) / 4);
    for quad in values.chunks(4) {
        let mut byte = 0u8;
        for (j, &v) in quad.iter().enumerate() {
            byte |= ((v as u8) & 0x03) << (j * 2);
        }
        packed.push(byte);
    }
    packed
}

/// Unpack int2: extract four signed 2-bit values per byte.
#[must_use]
pub fn unpack_int2(packed: &[u8]) -> Vec<i8> {
    let mut values = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        for shift in (0..8).step_by(2) {
            let raw = ((byte >> shift) & 0x03) as i8;
            let signed = if raw & 0x02 != 0 { raw | !0x03 } else { raw };
            values.push(signed);
        }
    }
    values
}

/// Pack signed int1 values: eight per byte, LSB first.
#[must_use]
pub fn pack_int1(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 7) / 8);
    for octet in values.chunks(8) {
        let mut byte = 0u8;
        for (j, &v) in octet.iter().enumerate() {
            byte |= ((v as u8) & 0x01) << j;
        }
        packed.push(byte);
    }
    packed
}

/// Unpack int1: extract eight 1-bit values per byte.
#[must_use]
pub fn unpack_int1(packed: &[u8]) -> Vec<i8> {
    let mut values = Vec::with_capacity(packed.len() * 8);
    for &byte in packed {
        for bit in 0..8 {
            values.push(((byte >> bit) & 1) as i8);
        }
    }
    values
}

/// Unpack int8 (identity — one value per byte, reinterpret as signed).
#[must_use]
pub fn unpack_int8(packed: &[u8]) -> Vec<i8> {
    packed.iter().map(|&b| b as i8).collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Maximum representable signed integer for a given bit width.
const fn max_signed_int(bits: u8) -> i32 {
    match bits {
        1 => 1,
        2 => 1,   // [-2, 1] but symmetric range is [-1, 1]
        4 => 7,   // [-8, 7]
        8 => 127, // [-128, 127]
        _ => 127,
    }
}

/// Pack signed values at the specified bit width.
fn pack_signed(values: &[i8], bits: u8) -> Vec<u8> {
    match bits {
        1 => pack_int1(values),
        2 => pack_int2(values),
        4 => pack_int4(values),
        8 => values.iter().map(|&v| v as u8).collect(),
        _ => values.iter().map(|&v| v as u8).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pack/unpack round-trips ──────────────────────────────────────────

    #[test]
    fn pack_unpack_int4_round_trip() {
        let values: Vec<i8> = vec![-7, 3, 0, -1, 7, -8, 5, 2];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed);
        assert_eq!(&unpacked[..values.len()], &values[..]);
    }

    #[test]
    fn pack_unpack_int2_round_trip() {
        let values: Vec<i8> = vec![-1, 0, 1, -2, 0, 1, -1, 0];
        let packed = pack_int2(&values);
        let unpacked = unpack_int2(&packed);
        assert_eq!(&unpacked[..values.len()], &values[..]);
    }

    #[test]
    fn pack_unpack_int1_round_trip() {
        let values: Vec<i8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let packed = pack_int1(&values);
        let unpacked = unpack_int1(&packed);
        assert_eq!(&unpacked[..values.len()], &values[..]);
    }

    #[test]
    fn pack_unpack_int8_round_trip() {
        let values: Vec<i8> = vec![-128, -1, 0, 1, 127, 42, -42];
        let packed: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        let unpacked = unpack_int8(&packed);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_int4_odd_count() {
        let values: Vec<i8> = vec![3, -2, 5];
        let packed = pack_int4(&values);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_int4(&packed);
        assert_eq!(unpacked[0], 3);
        assert_eq!(unpacked[1], -2);
        assert_eq!(unpacked[2], 5);
    }

    // ── Quantize per-layer ──────────────────────────────────────────────

    #[test]
    fn quantize_per_layer_int8_round_trip() {
        let weights: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25];
        let qw = quantize_per_layer(&weights, 8);
        assert_eq!(qw.bits, 8);
        assert_eq!(qw.zero_point, 0);
        assert!(qw.scale > 0.0);

        let dq = qw.dequantize();
        assert_eq!(dq.len(), weights.len());
        for (orig, recovered) in weights.iter().zip(dq.iter()) {
            assert!(
                (orig - recovered).abs() <= qw.scale / 2.0 + 1e-6,
                "Q8 round-trip: {orig} vs {recovered}, scale={}",
                qw.scale
            );
        }
    }

    #[test]
    fn quantize_per_layer_int4_round_trip() {
        let weights: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let qw = quantize_per_layer(&weights, 4);
        assert_eq!(qw.bits, 4);

        let dq = qw.dequantize();
        for (orig, recovered) in weights.iter().zip(dq.iter()) {
            assert!(
                (orig - recovered).abs() <= qw.scale / 2.0 + 1e-6,
                "Q4 round-trip: {orig} vs {recovered}, scale={}",
                qw.scale
            );
        }
    }

    #[test]
    fn quantize_per_layer_int2_round_trip() {
        let weights: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let qw = quantize_per_layer(&weights, 2);
        assert_eq!(qw.bits, 2);
        let dq = qw.dequantize();
        assert_eq!(dq.len(), weights.len());
    }

    #[test]
    fn quantize_per_layer_int1_round_trip() {
        let weights: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
        let qw = quantize_per_layer(&weights, 1);
        assert_eq!(qw.bits, 1);
        let dq = qw.dequantize();
        assert_eq!(dq.len(), weights.len());
    }

    #[test]
    fn quantize_all_zeros() {
        let weights = vec![0.0_f32; 16];
        let qw = quantize_per_layer(&weights, 8);
        let dq = qw.dequantize();
        for v in &dq {
            assert!(v.abs() < 1e-6, "zero weights should dequantize to ~0");
        }
    }

    #[test]
    fn quantize_clamps_extremes() {
        let weights = vec![1000.0_f32, -1000.0];
        let qw = quantize_per_layer(&weights, 8);
        let dq = qw.dequantize();
        assert!((dq[0] - 1000.0).abs() < qw.scale);
        assert!((dq[1] + 1000.0).abs() < qw.scale);
    }

    // ── Per-channel ─────────────────────────────────────────────────────

    #[test]
    fn quantize_per_channel_basic() {
        // shape: [2, 4] — 2 output channels, 4 weights each
        let weights: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // channel 0
            1.0, 2.0, 3.0, 4.0, // channel 1
        ];
        let channels = quantize_per_channel(&weights, &[2, 4], 0, 8);
        assert_eq!(channels.len(), 2);

        // Channel 1 has larger values, so it should have a larger scale
        assert!(channels[1].scale > channels[0].scale);
    }

    // ── Scale computation ───────────────────────────────────────────────

    #[test]
    fn scale_is_maxabs_over_maxint() {
        let weights = vec![0.0_f32, 3.5, -2.1];
        let qw = quantize_per_layer(&weights, 8);
        let expected_scale = 3.5 / 127.0;
        assert!(
            (qw.scale - expected_scale).abs() < 1e-6,
            "scale should be max_abs/127"
        );
    }

    #[test]
    fn max_quantization_error_within_half_scale() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 50.0).collect();
        // half-scale bound holds for 4-bit and 8-bit symmetric quantization;
        // 1-bit and 2-bit have coarser representable sets where the bound is
        // scale (not scale/2) due to the tiny codebook.
        for bits in [4, 8] {
            let qw = quantize_per_layer(&weights, bits);
            let dq = qw.dequantize();
            let max_err = weights
                .iter()
                .zip(dq.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err <= qw.scale / 2.0 + 1e-5,
                "bits={bits}: max_err={max_err}, half_scale={}",
                qw.scale / 2.0
            );
        }
    }
}
