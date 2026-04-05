// SPDX-License-Identifier: AGPL-3.0-or-later

//! Physical Unclonable Function (PUF) via int4 quantization noise.
//!
//! Each Akida chip produces slightly different int4 quantization when
//! the same floating-point weights are loaded. This analog manufacturing
//! variation creates a unique, reproducible device fingerprint.
//!
//! # Method
//!
//! 1. Load a known weight pattern (e.g., ramp from 0.0 to 1.0)
//! 2. Read back the int4-quantized values from SRAM
//! 3. The quantization delta (expected - actual) is the PUF signature
//! 4. Shannon entropy of the delta measures fingerprint uniqueness
//! 5. Hamming distance between two devices measures distinctiveness
//!
//! # Measured results
//!
//! Temporal PUF entropy: **6.34 bits** (see `baseCamp/systems/temporal_puf.md`).
//! This was measured using the ESN readout weights as the probe pattern.
//!
//! # Status
//!
//! Scaffolded — the measurement API and entropy calculations are defined.
//! Production use requires an `SramAccessor` and a loaded model.

use crate::error::{AkidaError, Result};
use crate::sram::SramAccessor;

/// A device fingerprint derived from int4 quantization noise.
///
/// The signature is the raw byte-level delta between expected and
/// actual SRAM contents after loading known weights.
#[derive(Debug, Clone)]
pub struct PufSignature {
    /// Raw quantization deltas (`expected_byte` - `actual_byte`).
    pub deltas: Vec<i8>,
    /// Number of SRAM probes taken.
    pub probe_count: usize,
    /// Which NPs were probed.
    pub np_indices: Vec<u32>,
    /// Shannon entropy of the signature in bits.
    pub entropy: f64,
}

impl PufSignature {
    /// Shannon entropy of this signature.
    #[must_use]
    pub const fn entropy(&self) -> f64 {
        self.entropy
    }

    /// Length of the signature in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.deltas.len()
    }

    /// Whether the signature is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }
}

/// Configuration for PUF measurement.
#[derive(Debug, Clone)]
pub struct PufConfig {
    /// Number of probe reads per NP (averaging reduces noise).
    pub probes_per_np: usize,
    /// Which NPs to probe (empty = probe all available).
    pub np_selection: Vec<u32>,
    /// Number of bytes to read from each NP.
    pub bytes_per_np: usize,
    /// Known weight pattern to load before readback.
    pub weight_pattern: WeightPattern,
}

impl Default for PufConfig {
    fn default() -> Self {
        Self {
            probes_per_np: 3,
            np_selection: Vec::new(),
            bytes_per_np: 256,
            weight_pattern: WeightPattern::Ramp,
        }
    }
}

/// The known weight pattern loaded for PUF measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPattern {
    /// Linear ramp from 0 to max (exercises all quantization bins).
    Ramp,
    /// Alternating 0x55/0xAA (maximum bit transitions).
    Checkerboard,
    /// All-ones pattern (tests saturation behavior).
    AllOnes,
    /// Pseudorandom pattern seeded from device ID.
    Pseudorandom,
}

/// Measure a PUF signature from a device.
///
/// Loads a known weight pattern into SRAM, reads it back, and
/// computes the quantization delta as the device fingerprint.
///
/// # Errors
///
/// Returns error if SRAM access fails or NP indices are invalid.
pub fn measure_puf(sram: &mut SramAccessor, config: &PufConfig) -> Result<PufSignature> {
    let layout = sram.layout().clone();
    let np_count = layout.np_count;

    let np_indices: Vec<u32> = if config.np_selection.is_empty() {
        (0..np_count.min(8)).collect()
    } else {
        config
            .np_selection
            .iter()
            .copied()
            .filter(|&np| np < np_count)
            .collect()
    };

    if np_indices.is_empty() {
        return Err(AkidaError::invalid_state(
            "no valid NPs for PUF measurement",
        ));
    }

    let expected = generate_pattern(config.weight_pattern, config.bytes_per_np);

    let mut all_deltas = Vec::new();

    for &np in &np_indices {
        let base_offset = layout
            .np_base_offset(np)
            .ok_or_else(|| AkidaError::invalid_state(format!("NP {np} out of range")))?;

        // Write known pattern
        #[expect(
            clippy::cast_possible_truncation,
            reason = "PUF word index fits u32 register write"
        )]
        sram.write_bar1(base_offset as usize, &expected)?;

        // Read back (average over multiple probes for noise reduction)
        let mut accumulated = vec![0i32; config.bytes_per_np];
        for _ in 0..config.probes_per_np {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "PUF word index fits u32 register write"
            )]
            let readback = sram.read_bar1(base_offset as usize, config.bytes_per_np)?;
            for (acc, &byte) in accumulated.iter_mut().zip(readback.iter()) {
                *acc += i32::from(byte);
            }
        }

        // Compute deltas
        let probe_count = config.probes_per_np as i32;
        for (i, acc) in accumulated.iter().enumerate() {
            let avg = (acc / probe_count) as i8;
            let exp = expected[i] as i8;
            all_deltas.push(exp.wrapping_sub(avg));
        }
    }

    let entropy = puf_entropy_from_deltas(&all_deltas);

    Ok(PufSignature {
        deltas: all_deltas,
        probe_count: config.probes_per_np * np_indices.len(),
        np_indices,
        entropy,
    })
}

/// Compute Shannon entropy of a PUF signature.
///
/// Higher entropy = more unique / harder to predict.
/// Maximum for 8-bit symbols: 8.0 bits.
#[must_use]
pub const fn puf_entropy(signature: &PufSignature) -> f64 {
    signature.entropy
}

/// Compute normalized Hamming distance between two PUF signatures.
///
/// Returns a value in [0.0, 1.0]:
/// - 0.0 = identical signatures (same device)
/// - 0.5 = random (independent devices, ideal PUF)
/// - 1.0 = perfectly anti-correlated (unlikely)
///
/// Signatures must be the same length.
#[must_use]
pub fn puf_hamming_distance(a: &PufSignature, b: &PufSignature) -> f64 {
    if a.deltas.len() != b.deltas.len() || a.deltas.is_empty() {
        return 1.0;
    }

    let total_bits = a.deltas.len() * 8;
    let differing_bits: usize = a
        .deltas
        .iter()
        .zip(b.deltas.iter())
        .map(|(&da, &db)| (da as u8 ^ db as u8).count_ones() as usize)
        .sum();

    differing_bits as f64 / total_bits as f64
}

/// Generate a known weight pattern of the given size.
fn generate_pattern(pattern: WeightPattern, size: usize) -> Vec<u8> {
    match pattern {
        WeightPattern::Ramp => (0..size).map(|i| (i % 256) as u8).collect(),
        WeightPattern::Checkerboard => (0..size)
            .map(|i| if i % 2 == 0 { 0x55 } else { 0xAA })
            .collect(),
        WeightPattern::AllOnes => vec![0xFF; size],
        WeightPattern::Pseudorandom => {
            let mut buf = vec![0u8; size];
            let mut state: u64 = 0x1234_5678_9ABC_DEF0;
            for byte in &mut buf {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = (state & 0xFF) as u8;
            }
            buf
        }
    }
}

/// Compute Shannon entropy from a delta array.
fn puf_entropy_from_deltas(deltas: &[i8]) -> f64 {
    if deltas.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &d in deltas {
        counts[d as u8 as usize] += 1;
    }

    let total = deltas.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = f64::from(count) / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_pattern() {
        let pattern = generate_pattern(WeightPattern::Ramp, 256);
        assert_eq!(pattern.len(), 256);
        assert_eq!(pattern[0], 0);
        assert_eq!(pattern[255], 255);
    }

    #[test]
    fn checkerboard_pattern() {
        let pattern = generate_pattern(WeightPattern::Checkerboard, 4);
        assert_eq!(pattern, vec![0x55, 0xAA, 0x55, 0xAA]);
    }

    #[test]
    fn entropy_uniform() {
        // Uniform distribution over 256 values = 8.0 bits
        let deltas: Vec<i8> = (0..=255).map(|i| i as i8).collect();
        let e = puf_entropy_from_deltas(&deltas);
        assert!((e - 8.0).abs() < 0.01);
    }

    #[test]
    fn entropy_constant() {
        // All same value = 0.0 bits
        let deltas = vec![0i8; 100];
        let e = puf_entropy_from_deltas(&deltas);
        assert!((e - 0.0).abs() < 0.001);
    }

    #[test]
    fn hamming_identical() {
        let sig = PufSignature {
            deltas: vec![1, 2, 3, 4],
            probe_count: 1,
            np_indices: vec![0],
            entropy: 2.0,
        };
        assert!((puf_hamming_distance(&sig, &sig) - 0.0).abs() < 0.001);
    }

    #[test]
    fn hamming_opposite() {
        let a = PufSignature {
            deltas: vec![0, 0, 0, 0],
            probe_count: 1,
            np_indices: vec![0],
            entropy: 0.0,
        };
        let b = PufSignature {
            deltas: vec![-1, -1, -1, -1], // 0xFF
            probe_count: 1,
            np_indices: vec![0],
            entropy: 0.0,
        };
        let dist = puf_hamming_distance(&a, &b);
        assert!(dist > 0.9);
    }

    #[test]
    fn default_config() {
        let config = PufConfig::default();
        assert_eq!(config.probes_per_np, 3);
        assert_eq!(config.weight_pattern, WeightPattern::Ramp);
    }

    #[test]
    fn all_ones_and_pseudorandom_patterns_have_expected_len() {
        let n = 64;
        let ones = generate_pattern(WeightPattern::AllOnes, n);
        assert!(ones.iter().all(|&b| b == 0xFF));
        let pr = generate_pattern(WeightPattern::Pseudorandom, n);
        assert_eq!(pr.len(), n);
        assert_ne!(pr, vec![0u8; n]);
    }

    #[test]
    fn puf_signature_accessors() {
        let sig = PufSignature {
            deltas: vec![1i8, 2, 3],
            probe_count: 9,
            np_indices: vec![0, 1],
            entropy: 1.25,
        };
        assert_eq!(sig.len(), 3);
        assert!(!sig.is_empty());
        assert!((sig.entropy() - 1.25).abs() < f64::EPSILON);
        assert!((puf_entropy(&sig) - 1.25).abs() < f64::EPSILON);
    }

    #[test]
    fn hamming_empty_or_mismatched_length_returns_one() {
        let a = PufSignature {
            deltas: vec![1i8],
            probe_count: 1,
            np_indices: vec![0],
            entropy: 0.0,
        };
        let b = PufSignature {
            deltas: vec![1i8, 2i8],
            probe_count: 1,
            np_indices: vec![0],
            entropy: 0.0,
        };
        assert!((puf_hamming_distance(&a, &b) - 1.0).abs() < f64::EPSILON);
        let empty = PufSignature {
            deltas: vec![],
            probe_count: 0,
            np_indices: vec![],
            entropy: 0.0,
        };
        assert!((puf_hamming_distance(&empty, &empty) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn entropy_empty_deltas_is_zero() {
        assert!((puf_entropy_from_deltas(&[]) - 0.0).abs() < f64::EPSILON);
    }
}
