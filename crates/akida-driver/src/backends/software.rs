// SPDX-License-Identifier: AGPL-3.0-or-later

//! Software (Virtual NPU) backend
//!
//! Implements the `NpuBackend` trait using pure f32 CPU arithmetic — the same
//! precision and computation path the AKD1000 hardware uses. This enables:
//!
//! 1. **Side-by-side comparison**: Run the same ESN on CPU-f64, SoftwareBackend-f32,
//!    and real AKD1000 to quantify the int4 quantization cost.
//!
//! 2. **CI without hardware**: `SoftwareBackend` produces results that are
//!    numerically close to the hardware (f32 vs int4 quantization is ~1–3%
//!    relative error). All tests pass without a physical AKD1000.
//!
//! 3. **Baseline for hardware validation**: The hardware validation binary
//!    (`validate_all`) compares real AKD1000 output to `SoftwareBackend` output
//!    as the ground truth.
//!
//! ## Origin
//!
//! Ported from `hotSpring::barracuda::md::reservoir::NpuSimulator` and
//! `MultiHeadNpu` (ecosystem context — not a runtime dependency). The hotSpring version uses dynamic allocation; this version
//! is restructured to implement `NpuBackend` and matches `ExportedWeights` format.
//!
//! ## Precision model
//!
//! ```text
//! CPU-f64 ESN training  →  f64 weights
//!    ↓ export_weights()
//! SoftwareBackend       →  f32 weights, f32 arithmetic  (this module)
//!    ↓ int4 quantization
//! AKD1000 hardware      →  int4 weights, hardware NP arithmetic
//! ```
//!
//! `SoftwareBackend` ≈ hardware in f32. The f32→int4 gap is the remaining
//! quantization error (typically 1–3% relative for well-conditioned weights).

use crate::backend::{BackendType, ModelHandle, NpuBackend};
use crate::capabilities::{Capabilities, ChipVersion, PcieConfig, WeightMutationSupport};
use crate::error::{AkidaError, Result};
use tracing::{debug, info};

/// Software (virtual NPU) backend.
///
/// Mirrors the AKD1000's f32 computation path in pure Rust CPU code.
/// Ideal for cross-substrate comparison, CI, and algorithm validation
/// before committing to hardware deployment.
#[derive(Debug)]
pub struct SoftwareBackend {
    /// Reservoir size (number of NPs simulated)
    reservoir_size: usize,
    /// Input dimensionality
    input_size: usize,
    /// Output dimensionality
    output_size: usize,
    /// Leak rate for ESN dynamics (same as hardware uses)
    leak_rate: f32,

    // Weight matrices (row-major, f32)
    w_in: Vec<f32>,  // [reservoir_size × input_size]
    w_res: Vec<f32>, // [reservoir_size × reservoir_size]
    w_out: Vec<f32>, // [output_size × reservoir_size]

    /// Current reservoir state (f32, matches hardware precision)
    state: Vec<f32>,

    /// Synthetic capabilities (represents the software simulation)
    caps: Capabilities,

    /// Whether reservoir weights have been loaded
    reservoir_loaded: bool,
    /// Model handle counter
    next_handle: u32,
}

impl SoftwareBackend {
    /// Create a software backend with explicit architecture.
    ///
    /// `reservoir_size` corresponds to the number of NPs in the Akida model.
    /// Common sizes from ecoPrimals models:
    /// - 64 NPs (Anderson regime, phase classifier)
    /// - 128 NPs (ESN QCD thermalization, transport predictor)
    /// - 256 NPs (ESN MSLP chaotic)
    pub fn new(reservoir_size: usize, input_size: usize, output_size: usize) -> Self {
        let caps = Capabilities {
            chip_version: ChipVersion::Akd1000,
            npu_count: reservoir_size as u32,
            memory_mb: 0,
            pcie: PcieConfig {
                generation: 0,
                lanes: 0,
                bandwidth_gbps: 0.0,
                speed_gts: 0.0,
            },
            power_mw: None,
            temperature_c: None,
            weight_mutation: WeightMutationSupport::Full, // SW supports instant swap
            mesh: None,
            clock_mode: None,
            batch: None,
        };
        Self {
            reservoir_size,
            input_size,
            output_size,
            leak_rate: 0.3,
            w_in: vec![0.0f32; reservoir_size * input_size],
            w_res: vec![0.0f32; reservoir_size * reservoir_size],
            w_out: vec![0.0f32; output_size * reservoir_size],
            state: vec![0.0f32; reservoir_size],
            caps,
            reservoir_loaded: false,
            next_handle: 1,
        }
    }

    /// Create with the default hotSpring ESN architecture (50 NPs, 8-dim input).
    ///
    /// Matches `hotSpring::barracuda::md::reservoir::EsnConfig::default()`.
    pub fn default_hotspring() -> Self {
        Self::new(50, 8, 1)
    }

    /// Set the leak rate (default 0.3, matches hotSpring default; ecosystem context — not a runtime dependency).
    #[must_use]
    pub const fn with_leak_rate(mut self, alpha: f32) -> Self {
        self.leak_rate = alpha;
        self
    }

    /// Load weights directly from flat f32 slices (same format as `ExportedWeights`).
    ///
    /// Layout (row-major):
    /// - `w_in`:  [RS × IS] — input → reservoir
    /// - `w_res`: [RS × RS] — reservoir → reservoir
    /// - `w_out`: [OS × RS] — reservoir → output
    ///
    /// # Errors
    ///
    /// Returns error if weight dimensions don't match configured architecture.
    pub fn load_weights(&mut self, w_in: &[f32], w_res: &[f32], w_out: &[f32]) -> Result<()> {
        let expected_in = self.reservoir_size * self.input_size;
        let expected_res = self.reservoir_size * self.reservoir_size;
        let expected_out = self.output_size * self.reservoir_size;

        if w_in.len() != expected_in {
            return Err(AkidaError::capability_query_failed(format!(
                "w_in size mismatch: got {}, expected {}×{}={}",
                w_in.len(),
                self.reservoir_size,
                self.input_size,
                expected_in
            )));
        }
        if w_res.len() != expected_res {
            return Err(AkidaError::capability_query_failed(format!(
                "w_res size mismatch: got {}, expected {}²={}",
                w_res.len(),
                self.reservoir_size,
                expected_res
            )));
        }
        if w_out.len() != expected_out {
            return Err(AkidaError::capability_query_failed(format!(
                "w_out size mismatch: got {}, expected {}×{}={}",
                w_out.len(),
                self.output_size,
                self.reservoir_size,
                expected_out
            )));
        }

        self.w_in.copy_from_slice(w_in);
        self.w_res.copy_from_slice(w_res);
        self.w_out.copy_from_slice(w_out);
        self.state.fill(0.0);
        self.reservoir_loaded = true;
        info!(
            "SoftwareBackend: loaded weights RS={} IS={} OS={}",
            self.reservoir_size, self.input_size, self.output_size
        );
        Ok(())
    }

    /// Swap readout weights only (equivalent to AKD1000 `set_variable()`, Discovery 6).
    ///
    /// The reservoir weights stay fixed; only `w_out` changes. This enables
    /// regime-specific readout heads — 86 µs on hardware, ~0 ns on software.
    ///
    /// # Errors
    ///
    /// Returns error if new `w_out` size doesn't match `output_size × reservoir_size`.
    pub fn swap_readout(&mut self, w_out: &[f32]) -> Result<()> {
        let expected = self.output_size * self.reservoir_size;
        if w_out.len() != expected {
            return Err(AkidaError::capability_query_failed(format!(
                "w_out swap size mismatch: got {}, expected {}",
                w_out.len(),
                expected
            )));
        }
        self.w_out.copy_from_slice(w_out);
        Ok(())
    }

    /// Swap to a different output dimensionality (e.g., 1→3 for multi-output).
    ///
    /// Reconfigures the backend for a different number of output heads.
    ///
    /// # Errors
    ///
    /// Returns error if `w_out` length does not match `new_output_size × reservoir_size`.
    pub fn reconfigure_output(&mut self, new_output_size: usize, w_out: &[f32]) -> Result<()> {
        let expected = new_output_size * self.reservoir_size;
        if w_out.len() != expected {
            return Err(AkidaError::capability_query_failed(format!(
                "w_out reconfigure size mismatch: got {}, expected {}×{}={}",
                w_out.len(),
                new_output_size,
                self.reservoir_size,
                expected
            )));
        }
        self.output_size = new_output_size;
        self.w_out = w_out.to_vec();
        Ok(())
    }

    /// Run the reservoir step on a single input frame (f32).
    ///
    /// Matches AKD1000 NP dynamics: leaky integrator with tanh activation.
    /// `state[t+1] = (1-α)·state[t] + α·tanh(W_in·input + W_res·state[t])`
    fn step(&mut self, input: &[f32]) {
        let rs = self.reservoir_size;
        let is = self.input_size;
        let alpha = self.leak_rate;
        let mut pre = vec![0.0f32; rs];

        for (i, pre_slot) in pre.iter_mut().enumerate() {
            let mut val = 0.0f32;
            // W_in contribution
            for (j, &inp) in input.iter().enumerate().take(is.min(input.len())) {
                val += self.w_in[i * is + j] * inp;
            }
            // W_res contribution
            for j in 0..rs {
                val += self.w_res[i * rs + j] * self.state[j];
            }
            *pre_slot = val;
        }

        for (i, pre_v) in pre.iter().enumerate() {
            self.state[i] = (1.0 - alpha).mul_add(self.state[i], alpha * pre_v.tanh());
        }
    }

    /// Apply readout layer to current reservoir state.
    ///
    /// Returns `output_size` output values.
    fn readout(&self) -> Vec<f32> {
        let rs = self.reservoir_size;
        let os = self.output_size;
        (0..os)
            .map(|i| {
                self.w_out[i * rs..(i + 1) * rs]
                    .iter()
                    .zip(self.state.iter())
                    .map(|(w, s)| w * s)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Process a full input sequence (multiple timesteps) and return readout.
    ///
    /// State is **not** reset between calls — the caller controls reset
    /// via `reset_state()`. This matches AKD1000 temporal streaming behavior.
    pub fn run_sequence(&mut self, inputs: &[f32]) -> Vec<f32> {
        let is = self.input_size;
        let frames = inputs.len() / is;
        for t in 0..frames {
            self.step(&inputs[t * is..(t + 1) * is]);
        }
        self.readout()
    }

    /// Reset reservoir state to zero (equivalent to power-cycle on hardware).
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Return a copy of the current reservoir state (for cross-substrate comparison).
    pub fn reservoir_state(&self) -> Vec<f32> {
        self.state.clone()
    }

    /// Return output dimensionality.
    pub const fn output_size(&self) -> usize {
        self.output_size
    }

    /// Return reservoir size (equivalent to NP count on hardware).
    pub const fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }
}

impl NpuBackend for SoftwareBackend {
    fn init(_device_id: &str) -> Result<Self> {
        // Default architecture — caller can reconfigure via load_weights()
        Ok(Self::default_hotspring())
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle> {
        // The software backend accepts a compact model blob:
        // [4 bytes RS][4 bytes IS][4 bytes OS][4 bytes leak_rate_bits]
        // [RS×IS × 4 bytes w_in][RS×RS × 4 bytes w_res][OS×RS × 4 bytes w_out]
        if model.len() < 16 {
            return Err(AkidaError::capability_query_failed(
                "SoftwareBackend model blob too short (need header + weights)".to_string(),
            ));
        }
        let to_u32 = |s: &[u8]| -> Result<u32> {
            let arr: [u8; 4] = s
                .try_into()
                .map_err(|_| AkidaError::capability_query_failed("invalid model header"))?;
            Ok(u32::from_le_bytes(arr))
        };
        let rs = to_u32(&model[0..4])? as usize;
        let is = to_u32(&model[4..8])? as usize;
        let os = to_u32(&model[8..12])? as usize;
        let lr = f32::from_le_bytes(
            model[12..16]
                .try_into()
                .map_err(|_| AkidaError::capability_query_failed("invalid leak rate bytes"))?,
        );

        let n_in = rs * is;
        let n_res = rs * rs;
        let n_out = os * rs;
        let expected = 16 + (n_in + n_res + n_out) * 4;

        if model.len() < expected {
            return Err(AkidaError::capability_query_failed(format!(
                "SoftwareBackend model blob too short: {} < {expected}",
                model.len()
            )));
        }

        self.reservoir_size = rs;
        self.input_size = is;
        self.output_size = os;
        self.leak_rate = lr;
        self.w_in = vec![0.0; n_in];
        self.w_res = vec![0.0; n_res];
        self.w_out = vec![0.0; n_out];
        self.state = vec![0.0; rs];
        self.caps.npu_count = rs as u32;

        let parse_f32_weights = |data: &[u8], out: &mut [f32]| {
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                // chunks_exact(4) guarantees exactly 4 bytes per chunk
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        };
        let base = 16usize;
        parse_f32_weights(&model[base..base + n_in * 4], &mut self.w_in);
        let base = base + n_in * 4;
        parse_f32_weights(&model[base..base + n_res * 4], &mut self.w_res);
        let base = base + n_res * 4;
        parse_f32_weights(&model[base..base + n_out * 4], &mut self.w_out);

        self.reservoir_loaded = true;
        let h = ModelHandle::new(self.next_handle);
        self.next_handle += 1;
        debug!("SoftwareBackend: loaded model RS={rs} IS={is} OS={os} leak={lr}");
        Ok(h)
    }

    fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()> {
        // Infer RS and IS from slice sizes
        let rs = self.reservoir_size;
        let is = if rs > 0 { w_in.len() / rs } else { 0 };
        if is == 0 || w_in.len() != rs * is || w_res.len() != rs * rs {
            return Err(AkidaError::capability_query_failed(format!(
                "load_reservoir: w_in={} w_res={} for RS={rs}",
                w_in.len(),
                w_res.len()
            )));
        }
        self.input_size = is;
        self.w_in = w_in.to_vec();
        self.w_res = w_res.to_vec();
        self.state.fill(0.0);
        self.reservoir_loaded = true;
        Ok(())
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if !self.reservoir_loaded {
            return Err(AkidaError::capability_query_failed(
                "SoftwareBackend: no weights loaded; call load_weights() first".to_string(),
            ));
        }
        // Single-step inference: treat input as one timestep
        self.step(input);
        Ok(self.readout())
    }

    fn measure_power(&self) -> Result<f32> {
        // CPU power not measured; return 0.0 (benchmark layer handles this)
        Ok(0.0)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Software
    }

    fn is_ready(&self) -> bool {
        self.reservoir_loaded
    }
}

/// Serialize ESN weights into a compact blob for `SoftwareBackend::load_model()`.
///
/// Format: [RS u32 LE][IS u32 LE][OS u32 LE][leak_rate f32 LE][`w_in` f32...][w_res f32...][`w_out` f32...]
#[must_use]
pub fn pack_software_model(
    reservoir_size: usize,
    input_size: usize,
    output_size: usize,
    leak_rate: f32,
    w_in: &[f32],
    w_res: &[f32],
    w_out: &[f32],
) -> Vec<u8> {
    let mut blob = Vec::with_capacity(16 + (w_in.len() + w_res.len() + w_out.len()) * 4);
    blob.extend_from_slice(&(reservoir_size as u32).to_le_bytes());
    blob.extend_from_slice(&(input_size as u32).to_le_bytes());
    blob.extend_from_slice(&(output_size as u32).to_le_bytes());
    blob.extend_from_slice(&leak_rate.to_le_bytes());
    for &w in w_in {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    for &w in w_res {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    for &w in w_out {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_esn(rs: usize, is: usize, os: usize) -> SoftwareBackend {
        let mut b = SoftwareBackend::new(rs, is, os);
        // w_in = zeros (no input effect)
        // w_res = zeros (no recurrence)
        // w_out = ones (sum of state)
        let w_in = vec![0.0f32; rs * is];
        let w_res = vec![0.0f32; rs * rs];
        let w_out = vec![1.0f32 / rs as f32; os * rs]; // mean readout
        b.load_weights(&w_in, &w_res, &w_out).unwrap();
        b
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let mut b = make_identity_esn(10, 3, 1);
        let input = vec![0.0f32; 3];
        let out = b.infer(&input).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0]).abs() < 1e-6, "zero input → zero output");
    }

    #[test]
    fn deterministic_inference() {
        let mut b1 = make_identity_esn(20, 4, 2);
        let mut b2 = make_identity_esn(20, 4, 2);
        let input = vec![1.0, -0.5, 0.3, 0.7];
        let o1 = b1.infer(&input).unwrap();
        let o2 = b2.infer(&input).unwrap();
        for (a, b) in o1.iter().zip(o2.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn state_resets_between_calls() {
        let mut b = make_identity_esn(10, 2, 1);
        let input = vec![1.0f32, 1.0];
        let _ = b.infer(&input).unwrap();
        b.reset_state();
        let state = b.reservoir_state();
        assert!(state.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn readout_swap_doubles_output() {
        let rs = 5;
        let mut b = SoftwareBackend::new(rs, 2, 1);
        let w_in = vec![1.0f32; rs * 2];
        let w_res = vec![0.0f32; rs * rs];
        let w_out_1 = vec![0.1f32; rs];
        let w_out_2 = vec![0.2f32; rs]; // double
        b.load_weights(&w_in, &w_res, &w_out_1).unwrap();
        let input = vec![0.5f32, 0.5];
        let out1 = b.infer(&input).unwrap()[0];
        b.swap_readout(&w_out_2).unwrap();
        b.reset_state();
        let out2 = b.infer(&input).unwrap()[0];
        assert!(
            (out2 / out1 - 2.0).abs() < 0.01,
            "doubled w_out should double output: {out1} → {out2}"
        );
    }

    #[test]
    fn pack_and_load_model_roundtrip() {
        let rs = 8;
        let is = 3;
        let os = 2;
        let w_in: Vec<f32> = (0..rs * is).map(|i| i as f32 * 0.01).collect();
        let w_res: Vec<f32> = (0..rs * rs).map(|i| i as f32 * 0.001).collect();
        let w_out: Vec<f32> = (0..os * rs).map(|i| i as f32 * 0.1).collect();
        let blob = pack_software_model(rs, is, os, 0.3, &w_in, &w_res, &w_out);

        let mut b = SoftwareBackend::new(rs, is, os);
        b.load_model(&blob).unwrap();
        assert_eq!(b.reservoir_size(), rs);
        assert_eq!(b.output_size(), os);
        assert!((b.leak_rate - 0.3).abs() < 1e-6);
    }

    #[test]
    fn backend_type_is_software() {
        let b = SoftwareBackend::new(10, 3, 1);
        assert_eq!(b.backend_type(), BackendType::Software);
    }

    #[test]
    fn not_ready_before_weights() {
        let b = SoftwareBackend::new(10, 3, 1);
        assert!(!b.is_ready());
    }

    #[test]
    fn ready_after_weights() {
        let b = make_identity_esn(10, 3, 1);
        assert!(b.is_ready());
    }

    #[test]
    fn load_model_rejects_short_header() {
        let mut b = SoftwareBackend::new(4, 2, 1);
        assert!(b.load_model(&[0u8; 8]).is_err());
    }

    #[test]
    fn load_model_rejects_truncated_weights() {
        let mut b = SoftwareBackend::new(4, 2, 1);
        let mut blob = vec![0u8; 16];
        blob[0..4].copy_from_slice(&4u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2u32.to_le_bytes());
        blob[8..12].copy_from_slice(&1u32.to_le_bytes());
        blob[12..16].copy_from_slice(&0.3f32.to_le_bytes());
        assert!(b.load_model(&blob).is_err());
    }

    #[test]
    fn load_reservoir_rejects_bad_dimensions() {
        let mut b = SoftwareBackend::new(4, 2, 1);
        assert!(b.load_reservoir(&[0.0f32; 7], &[0.0f32; 16]).is_err());
    }

    #[test]
    fn run_sequence_accumulates_state() {
        let mut b = SoftwareBackend::new(3, 2, 1);
        let w_in = vec![0.0f32; 6];
        let w_res = vec![0.0f32; 9];
        let w_out = vec![1.0f32, 0.0, 0.0];
        b.load_weights(&w_in, &w_res, &w_out).unwrap();
        let seq = vec![1.0f32, 0.0, 0.0, 1.0];
        let out = b.run_sequence(&seq);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn reconfigure_output_changes_dim() {
        let mut b = SoftwareBackend::new(4, 2, 1);
        let w_in = vec![0.0f32; 8];
        let w_res = vec![0.0f32; 16];
        let w_out = vec![1.0f32; 4];
        b.load_weights(&w_in, &w_res, &w_out).unwrap();
        let new_out = vec![0.5f32; 8];
        b.reconfigure_output(2, &new_out).unwrap();
        assert_eq!(b.output_size(), 2);
        let o = b.infer(&[0.0f32, 0.0]).unwrap();
        assert_eq!(o.len(), 2);
    }

    #[test]
    fn infer_without_weights_returns_error() {
        let mut b = SoftwareBackend::new(3, 2, 1);
        assert!(b.infer(&[0.0f32, 0.0]).is_err());
    }

    #[test]
    fn load_weights_rejects_each_weight_slot() {
        let mut b = SoftwareBackend::new(2, 2, 1);
        assert!(
            b.load_weights(&[0.0f32; 3], &[0.0f32; 4], &[0.0f32; 2])
                .is_err()
        );
        assert!(
            b.load_weights(&[0.0f32; 4], &[0.0f32; 3], &[0.0f32; 2])
                .is_err()
        );
        assert!(
            b.load_weights(&[0.0f32; 4], &[0.0f32; 4], &[0.0f32; 1])
                .is_err()
        );
    }

    #[test]
    fn swap_readout_rejects_bad_len() {
        let mut b = make_identity_esn(4, 2, 1);
        assert!(b.swap_readout(&[0.0f32; 3]).is_err());
    }

    #[test]
    fn reconfigure_output_rejects_bad_len() {
        let mut b = SoftwareBackend::new(4, 2, 1);
        assert!(b.reconfigure_output(2, &[0.0f32; 7]).is_err());
    }

    #[test]
    fn default_hotspring_and_with_leak_rate() {
        let b = SoftwareBackend::default_hotspring().with_leak_rate(0.5f32);
        assert_eq!(b.reservoir_size(), 50);
        let b2 = SoftwareBackend::new(4, 2, 1).with_leak_rate(0.2f32);
        assert_eq!(b2.reservoir_size(), 4);
    }

    #[test]
    fn npu_backend_init_default_architecture() {
        let b = SoftwareBackend::init("unused").expect("software init");
        assert_eq!(b.reservoir_size(), 50);
        assert_eq!(b.output_size(), 1);
    }

    #[test]
    fn measure_power_returns_zero() {
        let b = SoftwareBackend::new(4, 2, 1);
        assert!((b.measure_power().unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn load_reservoir_rs_zero_is_rejected() {
        let mut b = SoftwareBackend::new(0, 2, 1);
        assert!(b.load_reservoir(&[], &[]).is_err());
    }

    #[test]
    fn load_model_header_parse_errors() {
        let mut b = SoftwareBackend::new(2, 2, 1);
        assert!(b.load_model(&[0u8; 15]).is_err());
        let mut blob = vec![0xffu8; 20];
        blob[0..4].copy_from_slice(&2u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2u32.to_le_bytes());
        blob[8..12].copy_from_slice(&1u32.to_le_bytes());
        blob[12..16].copy_from_slice(&0.3f32.to_le_bytes());
        assert!(b.load_model(&blob).is_err());
    }
}
