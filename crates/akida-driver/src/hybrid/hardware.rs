// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware-backed ESN executor and scale-trick configuration.
//!
//! Implements Approach B (scale trick) for running tanh-trained ESN weights on
//! AKD1000's bounded `ReLU` activation path. See `metalForge/experiments/004_HYBRID_TANH`.

use super::{EsnWeights, SubstrateMode};
use crate::error::{AkidaError, Result};
use tracing::info;

// ── Approach B: scale trick parameters ───────────────────────────────────────

/// Scale trick configuration for Approach B hybrid execution.
///
/// Scales reservoir weights by epsilon so all pre-activation values remain in the
/// linear region of bounded `ReLU` (approximately linear for |x| < 0.1). The host then
/// recovers tanh by applying `tanh(hw_output / epsilon)`.
///
/// `metalForge/experiments/004_HYBRID_TANH` (Phase 1) validates this approach
/// live on the AKD1000. This struct is the software simulation — mathematically
/// identical to the hardware path, differing only in compute location.
#[derive(Debug, Clone)]
pub(super) struct ScaleTrickConfig {
    /// Scale factor (weights multiplied, activations stay linear).
    /// Default 0.01 puts activations in [0, 0.01 * `max_weight`], well within
    /// the bounded `ReLU` linear region.
    pub(super) epsilon: f32,
    /// Inverse: applied to `hw_output` before tanh recovery.
    pub(super) inv_epsilon: f32,
}

impl ScaleTrickConfig {
    /// Choose epsilon automatically using the 3-sigma statistical bound.
    ///
    /// Target: RMS pre-activation <= 0.05, so that activations remain in the
    /// approximately linear region of bounded `ReLU` (before the upper clamp).
    ///
    /// Expected max pre-activation approximately equals
    /// `epsilon * max_w * 3 * sqrt(is + rs)` (3-sigma bound).
    /// Solving: `epsilon <= 0.05 / (max_w * 3 * sqrt(is + rs))`.
    ///
    /// **Limitation**: bounded `ReLU`'s LOWER clamp (clip negatives to 0) is NOT
    /// eliminated by epsilon scaling — only the upper clamp is irrelevant here.
    /// Approach B is a partial fix. Approach A (`FlatBuffer` threshold override)
    /// eliminates the lower clamp entirely and achieves full tanh parity.
    pub(super) fn from_weights(w_in: &[f32], w_res: &[f32]) -> Self {
        let max_win = w_in.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_wres = w_res.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_weight = max_win.max(max_wres).max(1e-8);
        let rs = (w_res.len() as f64).sqrt().round() as usize;
        let is = w_in.len() / rs.max(1);
        let dof = ((is + rs) as f32).sqrt().max(1.0);
        let epsilon = (0.05 / (max_weight * 3.0 * dof)).clamp(1e-6, 1.0);
        Self {
            epsilon,
            inv_epsilon: 1.0 / epsilon,
        }
    }
}

// ── Internal: hardware executor ───────────────────────────────────────────────

/// Hardware-backed ESN executor.
///
/// **`HardwareLinear` (Approach B — active today)**
/// Uses the scale trick: weights * epsilon -> hardware matrix multiply -> host tanh recovery.
/// The hardware computes `bounded_relu(eps * W * x)`. Since epsilon is small,
/// `bounded_relu ~ identity` in that range, and the host recovers
/// `tanh((eps * W * x) / eps) = tanh(W * x)`.
///
/// Current implementation: software simulation of the hardware path. When
/// `metalForge/experiments/004_HYBRID_TANH` Phase 2 (`FlatBuffer` injection) validates
/// the actual hardware linear pass-through, replace `step_linear_emulated()` with
/// a real `device.infer()` call. The math — and the API — stay identical.
///
/// **`HardwareNative` (bounded `ReLU`)**
/// Requires MetaTF-compiled weights. For hotSpring/toadStool use `HardwareLinear` (ecosystem context — not a runtime dependency).
pub(super) struct HardwareEsnExecutor {
    pub(super) reservoir_dim: usize,
    pub(super) input_dim: usize,
    pub(super) output_dim: usize,
    pub(super) leak: f32,
    pub(super) mode: SubstrateMode,
    pub(super) state: Vec<f32>,
    pub(super) w_out: Vec<f32>,
    pub(super) w_in: Vec<f32>,
    pub(super) w_res: Vec<f32>,
    pub(super) w_in_scaled: Vec<f32>,
    pub(super) w_res_scaled: Vec<f32>,
    pub(super) scale: ScaleTrickConfig,
    pub(super) _device: Box<dyn std::any::Any + Send + Sync>,
}

impl std::fmt::Debug for HardwareEsnExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HardwareEsnExecutor")
            .field("reservoir_dim", &self.reservoir_dim)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("leak", &self.leak)
            .field("mode", &self.mode)
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}

impl HardwareEsnExecutor {
    pub(super) fn new_linear(device: crate::device::AkidaDevice, w: &EsnWeights) -> Self {
        let scale = ScaleTrickConfig::from_weights(&w.w_in, &w.w_res);
        let eps = scale.epsilon;
        let w_in_scaled: Vec<f32> = w.w_in.iter().map(|x| x * eps).collect();
        let w_res_scaled: Vec<f32> = w.w_res.iter().map(|x| x * eps).collect();

        info!(
            "HardwareLinear (Approach B): eps={:.4}, scaled max_w_in={:.4}, \
             max_w_res={:.4}. Emulating scale trick — hardware dispatch pending Exp 004 Phase 2.",
            eps,
            w_in_scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
            w_res_scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
        );

        Self {
            reservoir_dim: w.reservoir_dim,
            input_dim: w.input_dim,
            output_dim: w.output_dim,
            leak: w.leak_rate,
            mode: SubstrateMode::HardwareLinear,
            state: vec![0.0f32; w.reservoir_dim],
            w_out: w.w_out.clone(),
            w_in: w.w_in.clone(),
            w_res: w.w_res.clone(),
            w_in_scaled,
            w_res_scaled,
            scale,
            _device: Box::new(device),
        }
    }

    pub(super) fn new_native(device: crate::device::AkidaDevice, w: &EsnWeights) -> Self {
        let scale = ScaleTrickConfig::from_weights(&w.w_in, &w.w_res);
        info!("HardwareNative: device acquired, bounded ReLU emulation active");
        Self {
            reservoir_dim: w.reservoir_dim,
            input_dim: w.input_dim,
            output_dim: w.output_dim,
            leak: w.leak_rate,
            mode: SubstrateMode::HardwareNative,
            state: vec![0.0f32; w.reservoir_dim],
            w_out: w.w_out.clone(),
            w_in: w.w_in.clone(),
            w_res: w.w_res.clone(),
            w_in_scaled: w.w_in.clone(),
            w_res_scaled: w.w_res.clone(),
            scale,
            _device: Box::new(device),
        }
    }

    pub(super) fn step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let rs = self.reservoir_dim;
        let is = self.input_dim;
        if input.len() != is {
            return Err(AkidaError::capability_query_failed(format!(
                "hw step: input len {} != input_dim {}",
                input.len(),
                is
            )));
        }
        match self.mode {
            SubstrateMode::HardwareLinear => Ok(self.step_linear_emulated(input, rs, is)),
            SubstrateMode::HardwareNative => Ok(self.step_native_emulated(input, rs, is)),
            SubstrateMode::PureSoftware => unreachable!("HardwareEsnExecutor never in SW mode"),
        }
    }

    /// Approach B: scaled weights -> "hardware" linear multiply -> host tanh recovery.
    fn step_linear_emulated(&mut self, input: &[f32], rs: usize, is: usize) -> Vec<f32> {
        let alpha = self.leak;
        let inv_eps = self.scale.inv_epsilon;

        let mut hw_out = vec![0.0f32; rs];
        for (i, hw_slot) in hw_out.iter_mut().enumerate() {
            for (j, &inp) in input.iter().enumerate().take(is) {
                *hw_slot += self.w_in_scaled[i * is + j] * inp;
            }
            for j in 0..rs {
                *hw_slot += self.w_res_scaled[i * rs + j] * self.state[j];
            }
            *hw_slot = (*hw_slot).max(0.0);
        }

        let mut new_state = vec![0.0f32; rs];
        for (i, hw_v) in hw_out.iter().enumerate() {
            let pre_activation = hw_v * inv_eps;
            new_state[i] = (1.0 - alpha).mul_add(self.state[i], alpha * pre_activation.tanh());
        }
        self.state = new_state;

        let os = self.output_dim;
        (0..os)
            .map(|i| {
                self.w_out[i * rs..(i + 1) * rs]
                    .iter()
                    .zip(self.state.iter())
                    .map(|(w, s)| w * s)
                    .sum()
            })
            .collect()
    }

    /// `HardwareNative` emulation: bounded `ReLU` activation (SDK default behavior).
    fn step_native_emulated(&mut self, input: &[f32], rs: usize, is: usize) -> Vec<f32> {
        let alpha = self.leak;
        let mut pre = vec![0.0f32; rs];
        for (i, pre_slot) in pre.iter_mut().enumerate() {
            for (j, &inp) in input.iter().enumerate().take(is) {
                *pre_slot += self.w_in[i * is + j] * inp;
            }
            for j in 0..rs {
                *pre_slot += self.w_res[i * rs + j] * self.state[j];
            }
        }
        for (i, pre_v) in pre.iter().enumerate() {
            let relu = pre_v.max(0.0);
            self.state[i] = (1.0 - alpha).mul_add(self.state[i], alpha * relu);
        }
        let os = self.output_dim;
        (0..os)
            .map(|i| {
                self.w_out[i * rs..(i + 1) * rs]
                    .iter()
                    .zip(self.state.iter())
                    .map(|(w, s)| w * s)
                    .sum()
            })
            .collect()
    }

    pub(super) fn reset(&mut self) {
        self.state.fill(0.0);
    }
}
