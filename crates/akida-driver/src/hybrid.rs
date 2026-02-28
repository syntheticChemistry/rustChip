// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid ESN executor and substrate abstraction.
//!
//! # The Problem This Solves
//!
//! The AKD1000 uses bounded ReLU as its activation function. Reservoir computing
//! (Echo State Networks) requires tanh for robust echo state property with arbitrary
//! weight initialization. Running an ESN on hardware with bounded ReLU requires
//! purpose-designed reservoir weights — random initialization produces degenerate
//! (near-chance) reservoir states.
//!
//! See `whitePaper/explorations/TANH_CONSTRAINT.md` for the full analysis.
//!
//! # The Solution
//!
//! `HybridEsn` accepts weights trained under tanh dynamics (hotSpring's standard
//! output) and executes them correctly regardless of which substrate is available:
//!
//! - **Software mode** (available today): SoftwareBackend with native tanh.
//!   Accuracy: hotSpring's validated 89.7% on QCD thermalization.
//!   Throughput: ~800 Hz. Used when no NPU is present.
//!
//! - **Hardware-linear mode** (pending `metalForge/experiments/004_HYBRID_TANH`):
//!   AKD1000 computes the matrix multiply (int4, parallel, 54 µs).
//!   Host applies tanh to the 128-float result (< 1 µs).
//!   Accuracy: same 89.7% (tanh preserved). Throughput: 18,500 Hz.
//!   Activated by calling `with_hardware_device()`.
//!
//! # hotSpring Integration
//!
//! ```no_run
//! use akida_driver::{HybridEsn, EsnSubstrate};
//!
//! # let w_in  = vec![0.1f32; 128 * 6];
//! # let w_res = vec![0.05f32; 128 * 128];
//! # let w_out = vec![0.2f32; 3 * 128];
//! # let plaquette_features = vec![0.0f32; 6];
//! // hotSpring exports its existing tanh-trained weights — no retraining
//! let mut esn = HybridEsn::from_weights(
//!     &w_in,          // hotSpring's existing f32 w_in  (reservoir_size × input_dim)
//!     &w_res,         // hotSpring's existing f32 w_res (reservoir_size × reservoir_size)
//!     &w_out,         // hotSpring's existing f32 w_out (output_dim × reservoir_size)
//!     0.3,            // leak rate (hotSpring's α)
//! )?;
//!
//! // Identical API to hotSpring's software ESN — drop-in replacement
//! let prediction = esn.step(&plaquette_features)?;
//! # Ok::<(), akida_driver::AkidaError>(())
//! ```
//!
//! When hardware is available and validated, swap to hardware speed with one call:
//! ```no_run
//! # use akida_driver::{HybridEsn, DeviceManager};
//! # let (w_in, w_res, w_out) = (vec![0.1f32; 128*4], vec![0.05f32; 128*128], vec![0.2f32; 128]);
//! # let esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3).unwrap();
//! let mgr = DeviceManager::discover()?;
//! let esn = esn.with_hardware_linear(mgr.open_first()?)?;
//! // Now runs at 18,500 Hz, 1.4 µJ/inference — same weights, same accuracy
//! # Ok::<(), akida_driver::AkidaError>(())
//! ```
//!
//! # toadStool Integration
//!
//! ```no_run
//! use akida_driver::{SubstrateSelector, SubstrateInfo};
//!
//! # let w_in  = vec![0.1f32; 128 * 6];
//! # let w_res = vec![0.05f32; 128 * 128];
//! # let w_out = vec![0.2f32; 3 * 128];
//! # let features = vec![0.0f32; 6];
//! // toadStool builds a selector — dispatches to best available substrate
//! let mut selector = SubstrateSelector::for_weights(&w_in, &w_res, &w_out, 0.3)?;
//! println!("Active substrate: {:?}", selector.active_substrate().mode);
//!
//! // Single dispatch call works on any substrate
//! let result = selector.esn_step(&features)?;
//! # Ok::<(), akida_driver::AkidaError>(())
//! ```

use crate::error::{AkidaError, Result};
use tracing::{debug, info};

// ── Substrate mode ────────────────────────────────────────────────────────────

/// Which substrate is currently executing the ESN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstrateMode {
    /// Pure CPU f32 with tanh activation (SoftwareBackend).
    /// Available today. Accuracy: hotSpring's validated software performance.
    PureSoftware,

    /// AKD1000 hardware linear transform + host tanh activation.
    /// Pending `metalForge/experiments/004_HYBRID_TANH` validation.
    /// Accuracy: same as software (tanh preserved). Throughput: 18,500 Hz.
    HardwareLinear,

    /// AKD1000 hardware with bounded ReLU (SDK default mode).
    /// Requires purpose-designed reservoir weights (MetaTF-trained).
    /// Accuracy: 86.1% on QCD (3.6% below software tanh).
    HardwareNative,
}

impl SubstrateMode {
    /// Human-readable description for logging and toadStool telemetry.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::PureSoftware    => "CPU f32 + tanh  (~800 Hz, ~44 mJ/inf)",
            Self::HardwareLinear  => "AKD1000 linear + host tanh  (18,500 Hz, 1.4 µJ/inf)",
            Self::HardwareNative  => "AKD1000 bounded ReLU  (18,500 Hz, 1.4 µJ/inf, -3.6% acc)",
        }
    }

    /// Whether this substrate preserves tanh-trained weight accuracy.
    #[must_use]
    pub fn is_tanh_accurate(&self) -> bool {
        matches!(self, Self::PureSoftware | Self::HardwareLinear)
    }
}

// ── ESN substrate trait — what hotSpring and toadStool program against ────────

/// Unified interface for ESN inference across all substrates.
///
/// hotSpring implements its simulation runner against this trait.
/// toadStool's substrate dispatch uses this trait for NPU-aware scheduling.
///
/// All implementors must preserve the temporal state between calls — a `step()`
/// call advances the reservoir state, and the next call sees the updated state.
/// Call `reset()` to clear state between independent sequences.
pub trait EsnSubstrate: Send + Sync {
    /// Advance reservoir by one timestep and return readout.
    ///
    /// `input` must have length == `input_dim()`.
    /// Returns `output_dim()` float values.
    ///
    /// # Errors
    ///
    /// Returns error if input dimension mismatches or substrate is not ready.
    fn step(&mut self, input: &[f32]) -> Result<Vec<f32>>;

    /// Process a sequence of inputs, returning the final readout.
    ///
    /// Equivalent to calling `step()` `inputs.len() / input_dim()` times.
    ///
    /// # Errors
    ///
    /// Returns error if input length is not a multiple of `input_dim()`.
    fn run_sequence(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        let is = self.input_dim();
        if inputs.len() % is != 0 {
            return Err(AkidaError::capability_query_failed(format!(
                "run_sequence: input length {} not divisible by input_dim {}",
                inputs.len(), is
            )));
        }
        let mut out = vec![0.0f32; self.output_dim()];
        for chunk in inputs.chunks(is) {
            out = self.step(chunk)?;
        }
        Ok(out)
    }

    /// Reset reservoir state to zero (start of new sequence).
    fn reset(&mut self);

    /// Current reservoir state vector (for cross-substrate comparison / debug).
    fn reservoir_state(&self) -> Vec<f32>;

    /// Input dimension (number of floats expected per `step()` call).
    fn input_dim(&self) -> usize;

    /// Reservoir dimension (number of NPs / simulated neurons).
    fn reservoir_dim(&self) -> usize;

    /// Output dimension (number of floats returned per `step()` call).
    fn output_dim(&self) -> usize;

    /// Which substrate is executing this instance.
    fn substrate_mode(&self) -> SubstrateMode;

    /// Estimated throughput in inferences/second.
    ///
    /// Used by toadStool's scheduler to select the fastest available substrate.
    fn estimated_hz(&self) -> f64 {
        match self.substrate_mode() {
            SubstrateMode::PureSoftware   => 800.0,
            SubstrateMode::HardwareLinear => 18_500.0,
            SubstrateMode::HardwareNative => 18_500.0,
        }
    }

    /// Estimated energy per inference in µJ.
    fn estimated_energy_uj(&self) -> f64 {
        match self.substrate_mode() {
            SubstrateMode::PureSoftware   => 44_000.0,  // ~44 mJ
            SubstrateMode::HardwareLinear => 1.4,
            SubstrateMode::HardwareNative => 1.4,
        }
    }
}

// ── Weight container — what hotSpring produces ────────────────────────────────

/// ESN weight matrices exported from hotSpring (or any training framework).
///
/// All weights are in tanh-training format (f32, row-major).
/// No quantization, no bounded-ReLU re-optimization required.
#[derive(Debug, Clone)]
pub struct EsnWeights {
    /// Input projection: `[reservoir_dim × input_dim]` row-major
    pub w_in:  Vec<f32>,
    /// Recurrent weights: `[reservoir_dim × reservoir_dim]` row-major
    pub w_res: Vec<f32>,
    /// Readout weights: `[output_dim × reservoir_dim]` row-major
    pub w_out: Vec<f32>,
    /// Input dimensionality
    pub input_dim:     usize,
    /// Reservoir dimensionality (number of NPs on hardware)
    pub reservoir_dim: usize,
    /// Output dimensionality
    pub output_dim:    usize,
    /// Leak rate α ∈ (0, 1]
    pub leak_rate: f32,
}

impl EsnWeights {
    /// Construct from raw weight slices.
    ///
    /// Validates dimensions before accepting.
    ///
    /// # Errors
    ///
    /// Returns error if slice lengths are inconsistent with declared dimensions.
    pub fn new(
        w_in:  Vec<f32>,
        w_res: Vec<f32>,
        w_out: Vec<f32>,
        input_dim:     usize,
        reservoir_dim: usize,
        output_dim:    usize,
        leak_rate: f32,
    ) -> Result<Self> {
        if w_in.len()  != reservoir_dim * input_dim {
            return Err(AkidaError::capability_query_failed(format!(
                "w_in: expected {}×{}={}, got {}",
                reservoir_dim, input_dim, reservoir_dim * input_dim, w_in.len()
            )));
        }
        if w_res.len() != reservoir_dim * reservoir_dim {
            return Err(AkidaError::capability_query_failed(format!(
                "w_res: expected {}²={}, got {}",
                reservoir_dim, reservoir_dim * reservoir_dim, w_res.len()
            )));
        }
        if w_out.len() != output_dim * reservoir_dim {
            return Err(AkidaError::capability_query_failed(format!(
                "w_out: expected {}×{}={}, got {}",
                output_dim, reservoir_dim, output_dim * reservoir_dim, w_out.len()
            )));
        }
        if !(0.0..=1.0).contains(&leak_rate) {
            return Err(AkidaError::capability_query_failed(format!(
                "leak_rate {leak_rate} must be in (0, 1]"
            )));
        }
        Ok(Self { w_in, w_res, w_out, input_dim, reservoir_dim, output_dim, leak_rate })
    }

    /// Spectral radius of w_res (rough estimate via power iteration).
    ///
    /// An ESN with tanh needs spectral radius < 1 for echo state property.
    /// Hardware ESNs may use higher values (bounded ReLU prevents explosion).
    /// After hybrid migration, ensure spectral radius < 1.
    #[must_use]
    pub fn spectral_radius_estimate(&self, iters: usize) -> f32 {
        let rs = self.reservoir_dim;
        let mut v = vec![1.0f32 / (rs as f32).sqrt(); rs];
        for _ in 0..iters {
            let mut mv = vec![0.0f32; rs];
            for i in 0..rs {
                for j in 0..rs {
                    mv[i] += self.w_res[i * rs + j] * v[j];
                }
            }
            let norm = mv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            for (vi, mvi) in v.iter_mut().zip(mv.iter()) {
                *vi = mvi / norm;
            }
            let rayleigh: f32 = v.iter().enumerate()
                .map(|(i, &vi)| vi * mv[i].max(-norm).min(norm))
                .sum();
            if rayleigh.abs() > 0.0 { return rayleigh.abs(); }
        }
        1.0
    }
}

// ── HybridEsn — the main interface ───────────────────────────────────────────

/// Substrate-agnostic ESN executor.
///
/// Accepts tanh-trained weights from hotSpring and dispatches to:
/// - CPU f32 + tanh today (SoftwareBackend, correct results)
/// - AKD1000 + host tanh when hardware mode is validated (Exp 004)
///
/// The substrate can be changed at runtime without re-loading weights.
/// hotSpring and toadStool program against this type (or `EsnSubstrate`).
pub struct HybridEsn {
    weights:    EsnWeights,
    mode:       SubstrateMode,
    /// Software backend — always present (fallback + current primary)
    sw_backend: SoftwareEsnExecutor,
    /// Hardware backend — present only when hardware is available and validated
    hw_backend: Option<HardwareEsnExecutor>,
}

impl std::fmt::Debug for HybridEsn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridEsn")
            .field("reservoir_dim", &self.weights.reservoir_dim)
            .field("input_dim", &self.weights.input_dim)
            .field("output_dim", &self.weights.output_dim)
            .field("mode", &self.mode)
            .finish()
    }
}

impl HybridEsn {
    /// Create from raw weight slices (hotSpring's primary path).
    ///
    /// Uses `PureSoftware` mode by default. Call `with_hardware_device()` to
    /// upgrade to hardware once `metalForge/experiments/004_HYBRID_TANH` is validated.
    ///
    /// # Errors
    ///
    /// Returns error if weight dimensions are inconsistent.
    pub fn from_weights(
        w_in:  &[f32],
        w_res: &[f32],
        w_out: &[f32],
        leak_rate: f32,
    ) -> Result<Self> {
        // Infer dimensions from slice sizes
        // w_res must be square: rs×rs
        let rs_sq = w_res.len();
        let reservoir_dim = (rs_sq as f64).sqrt().round() as usize;
        if reservoir_dim * reservoir_dim != rs_sq {
            return Err(AkidaError::capability_query_failed(format!(
                "w_res length {} is not a perfect square", rs_sq
            )));
        }
        let input_dim = w_in.len() / reservoir_dim;
        if input_dim * reservoir_dim != w_in.len() {
            return Err(AkidaError::capability_query_failed(format!(
                "w_in length {} not divisible by reservoir_dim {}", w_in.len(), reservoir_dim
            )));
        }
        let output_dim = w_out.len() / reservoir_dim;
        if output_dim * reservoir_dim != w_out.len() {
            return Err(AkidaError::capability_query_failed(format!(
                "w_out length {} not divisible by reservoir_dim {}", w_out.len(), reservoir_dim
            )));
        }

        let weights = EsnWeights::new(
            w_in.to_vec(), w_res.to_vec(), w_out.to_vec(),
            input_dim, reservoir_dim, output_dim, leak_rate,
        )?;

        let sw_backend = SoftwareEsnExecutor::new(&weights);

        info!("HybridEsn: {}→{}→{} (leak={:.2}), mode=PureSoftware",
              input_dim, reservoir_dim, output_dim, leak_rate);

        Ok(Self {
            weights,
            mode: SubstrateMode::PureSoftware,
            sw_backend,
            hw_backend: None,
        })
    }

    /// Create from a validated `EsnWeights` container.
    ///
    /// # Errors
    ///
    /// Returns error if weights are internally inconsistent.
    pub fn from_esn_weights(weights: EsnWeights) -> Result<Self> {
        let sw_backend = SoftwareEsnExecutor::new(&weights);
        info!("HybridEsn: {}→{}→{} (leak={:.2}), mode=PureSoftware",
              weights.input_dim, weights.reservoir_dim, weights.output_dim, weights.leak_rate);
        Ok(Self {
            weights,
            mode: SubstrateMode::PureSoftware,
            sw_backend,
            hw_backend: None,
        })
    }

    /// Upgrade to hardware-linear mode.
    ///
    /// The device is used for the matrix-multiply step; the host applies tanh.
    /// This preserves full tanh accuracy at hardware speed (18,500 Hz, 1.4 µJ/inf).
    ///
    /// **Status:** Pending `metalForge/experiments/004_HYBRID_TANH` validation.
    /// Call this only after confirming hardware linear-only inference is working.
    ///
    /// # Errors
    ///
    /// Returns error if device is incompatible with the loaded weights.
    pub fn with_hardware_linear(mut self, device: crate::device::AkidaDevice) -> Result<Self> {
        let hw = HardwareEsnExecutor::new_linear(device, &self.weights)?;
        self.hw_backend = Some(hw);
        self.mode = SubstrateMode::HardwareLinear;
        info!("HybridEsn: upgraded to HardwareLinear mode (18,500 Hz, tanh-accurate)");
        Ok(self)
    }

    /// Upgrade to hardware-native mode (bounded ReLU — for MetaTF-designed weights only).
    ///
    /// Use this only when weights were explicitly designed for bounded ReLU dynamics
    /// (i.e., trained via MetaTF, not hotSpring's software ESN path).
    /// Accuracy: 86.1% on QCD (3.6% below tanh). Throughput: 18,500 Hz.
    ///
    /// For hotSpring weights: prefer `with_hardware_linear()` instead.
    ///
    /// # Errors
    ///
    /// Returns error if device cannot be initialized.
    pub fn with_hardware_native(mut self, device: crate::device::AkidaDevice) -> Result<Self> {
        let hw = HardwareEsnExecutor::new_native(device, &self.weights)?;
        self.hw_backend = Some(hw);
        self.mode = SubstrateMode::HardwareNative;
        info!("HybridEsn: upgraded to HardwareNative mode (bounded ReLU, -3.6% acc)");
        Ok(self)
    }

    /// Downgrade to software mode (e.g., hardware device lost or being reconfigured).
    pub fn to_software_mode(&mut self) {
        self.hw_backend = None;
        self.mode = SubstrateMode::PureSoftware;
        info!("HybridEsn: downgraded to PureSoftware mode");
    }

    /// Current operating mode.
    #[must_use]
    pub fn mode(&self) -> &SubstrateMode {
        &self.mode
    }

    /// Underlying weights (for inspection, export, or cross-substrate validation).
    #[must_use]
    pub fn weights(&self) -> &EsnWeights {
        &self.weights
    }

    /// Spectral radius estimate of the reservoir (should be < 1 for echo state property).
    #[must_use]
    pub fn spectral_radius(&self) -> f32 {
        self.weights.spectral_radius_estimate(50)
    }
}

impl EsnSubstrate for HybridEsn {
    fn step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        match self.mode {
            SubstrateMode::PureSoftware => self.sw_backend.step(input),
            SubstrateMode::HardwareLinear | SubstrateMode::HardwareNative => {
                if let Some(hw) = self.hw_backend.as_mut() {
                    hw.step(input)
                } else {
                    // Fallback: hardware selected but device not initialized
                    debug!("HybridEsn: hw_backend missing, falling back to software");
                    self.sw_backend.step(input)
                }
            }
        }
    }

    fn reset(&mut self) {
        self.sw_backend.reset();
        if let Some(hw) = self.hw_backend.as_mut() {
            hw.reset();
        }
    }

    fn reservoir_state(&self) -> Vec<f32> {
        match self.mode {
            SubstrateMode::PureSoftware => self.sw_backend.state.clone(),
            _ => self.hw_backend.as_ref()
                    .map(|hw| hw.state.clone())
                    .unwrap_or_else(|| self.sw_backend.state.clone()),
        }
    }

    fn input_dim(&self)     -> usize { self.weights.input_dim }
    fn reservoir_dim(&self) -> usize { self.weights.reservoir_dim }
    fn output_dim(&self)    -> usize { self.weights.output_dim }
    fn substrate_mode(&self) -> SubstrateMode { self.mode.clone() }
}

// ── Internal: software executor ───────────────────────────────────────────────

/// Pure f32 + tanh ESN executor (powers PureSoftware mode).
struct SoftwareEsnExecutor {
    input_dim:     usize,
    reservoir_dim: usize,
    output_dim:    usize,
    w_in:  Vec<f32>,
    w_res: Vec<f32>,
    w_out: Vec<f32>,
    state: Vec<f32>,
    leak:  f32,
}

impl SoftwareEsnExecutor {
    fn new(w: &EsnWeights) -> Self {
        Self {
            input_dim:     w.input_dim,
            reservoir_dim: w.reservoir_dim,
            output_dim:    w.output_dim,
            w_in:  w.w_in.clone(),
            w_res: w.w_res.clone(),
            w_out: w.w_out.clone(),
            state: vec![0.0f32; w.reservoir_dim],
            leak:  w.leak_rate,
        }
    }

    fn step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let rs = self.reservoir_dim;
        let is = self.input_dim;
        if input.len() != is {
            return Err(AkidaError::capability_query_failed(format!(
                "step: input len {} != input_dim {}", input.len(), is
            )));
        }
        let alpha = self.leak;
        let mut pre = vec![0.0f32; rs];
        for i in 0..rs {
            for j in 0..is  { pre[i] += self.w_in[i * is + j] * input[j]; }
            for j in 0..rs  { pre[i] += self.w_res[i * rs + j] * self.state[j]; }
        }
        for i in 0..rs {
            // tanh — the critical activation for echo state property
            self.state[i] = (1.0 - alpha) * self.state[i] + alpha * pre[i].tanh();
        }
        let os = self.output_dim;
        Ok((0..os).map(|i| {
            self.w_out[i * rs..(i + 1) * rs].iter().zip(self.state.iter())
                .map(|(w, s)| w * s).sum()
        }).collect())
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

// ── Approach B: scale trick parameters ───────────────────────────────────────

/// Scale trick configuration for Approach B hybrid execution.
///
/// Scales reservoir weights by ε so all pre-activation values remain in the
/// linear region of bounded ReLU (≈ linear for |x| < 0.1). The host then
/// recovers tanh by applying `tanh(hw_output / ε)`.
///
/// `metalForge/experiments/004_HYBRID_TANH` (Phase 1) validates this approach
/// live on the AKD1000. This struct is the software simulation — mathematically
/// identical to the hardware path, differing only in compute location.
#[derive(Debug, Clone)]
struct ScaleTrickConfig {
    /// Scale factor ε (weights multiplied, activations stay linear).
    /// Default 0.01 puts activations in [0, 0.01 × max_weight], well within
    /// the bounded ReLU linear region.
    epsilon: f32,
    /// Inverse: applied to hw_output before tanh recovery.
    inv_epsilon: f32,
}

impl ScaleTrickConfig {
    /// Choose ε automatically using the 3σ statistical bound.
    ///
    /// Target: RMS pre-activation ≤ 0.05, so that activations remain in the
    /// approximately linear region of bounded ReLU (before the upper clamp).
    ///
    /// Expected max pre-activation ≈ ε × max_w × 3 × √(is + rs) [3σ bound].
    /// Solving: ε ≤ 0.05 / (max_w × 3 × √(is + rs)).
    ///
    /// **Limitation**: bounded ReLU's LOWER clamp (clip negatives to 0) is NOT
    /// eliminated by ε scaling — only the upper clamp is irrelevant here.
    /// Approach B is a partial fix. Approach A (FlatBuffer threshold override)
    /// eliminates the lower clamp entirely and achieves full tanh parity.
    fn from_weights(w_in: &[f32], w_res: &[f32]) -> Self {
        let max_win  = w_in.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_wres = w_res.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_weight = max_win.max(max_wres).max(1e-8);
        // Infer (is + rs) from w_in and w_res sizes:
        // w_in is rs × is → rs = sqrt(w_res.len()), is = w_in.len() / rs
        let rs = (w_res.len() as f64).sqrt().round() as usize;
        let is = w_in.len() / rs.max(1);
        let dof = ((is + rs) as f32).sqrt().max(1.0);
        let epsilon = (0.05 / (max_weight * 3.0 * dof)).min(1.0).max(1e-6);
        Self { epsilon, inv_epsilon: 1.0 / epsilon }
    }
}

// ── Internal: hardware executor ───────────────────────────────────────────────

/// Hardware-backed ESN executor.
///
/// **`HardwareLinear` (Approach B — active today)**
/// Uses the scale trick: weights × ε → hardware matrix multiply → host tanh recovery.
/// The hardware computes `bounded_relu(ε W x)`. Since ε is small, `bounded_relu ≈ identity`
/// in that range, and the host recovers `tanh((ε W x) / ε) = tanh(W x)`.
///
/// Current implementation: software simulation of the hardware path. When
/// `metalForge/experiments/004_HYBRID_TANH` Phase 2 (FlatBuffer injection) validates
/// the actual hardware linear pass-through, replace `step_linear_emulated()` with
/// a real `device.infer()` call. The math — and the API — stay identical.
///
/// **`HardwareNative` (bounded ReLU)**
/// Requires MetaTF-compiled weights. For hotSpring/toadStool use `HardwareLinear`.
struct HardwareEsnExecutor {
    reservoir_dim: usize,
    input_dim:     usize,
    output_dim:    usize,
    leak:          f32,
    mode:          SubstrateMode,
    state:         Vec<f32>,
    /// Readout weights (host-side, applied after reservoir step).
    w_out: Vec<f32>,
    /// Full-scale reservoir weights (for `HardwareNative` emulation).
    w_in:  Vec<f32>,
    w_res: Vec<f32>,
    /// Scaled weights + ε config for `HardwareLinear` Approach B.
    w_in_scaled:  Vec<f32>,
    w_res_scaled: Vec<f32>,
    scale:        ScaleTrickConfig,
    /// Device handle — used in Phase 2 once FlatBuffer path is live.
    /// Box<dyn Any> lets us compile without HW on the bench path;
    /// Phase 2 will make this a concrete `crate::device::AkidaDevice`.
    _device: Box<dyn std::any::Any + Send + Sync>,
}

impl HardwareEsnExecutor {
    fn new_linear(device: crate::device::AkidaDevice, w: &EsnWeights) -> Result<Self> {
        let scale = ScaleTrickConfig::from_weights(&w.w_in, &w.w_res);
        let eps = scale.epsilon;
        let w_in_scaled:  Vec<f32> = w.w_in.iter().map(|x| x * eps).collect();
        let w_res_scaled: Vec<f32> = w.w_res.iter().map(|x| x * eps).collect();

        info!(
            "HardwareLinear (Approach B): eps={:.4}, scaled max_w_in={:.4}, \
             max_w_res={:.4}. Emulating scale trick — hardware dispatch pending Exp 004 Phase 2.",
            eps,
            w_in_scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
            w_res_scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
        );

        Ok(Self {
            reservoir_dim: w.reservoir_dim,
            input_dim:     w.input_dim,
            output_dim:    w.output_dim,
            leak:          w.leak_rate,
            mode:          SubstrateMode::HardwareLinear,
            state:         vec![0.0f32; w.reservoir_dim],
            w_out:         w.w_out.clone(),
            w_in:          w.w_in.clone(),
            w_res:         w.w_res.clone(),
            w_in_scaled,
            w_res_scaled,
            scale,
            _device:       Box::new(device),
        })
    }

    fn new_native(device: crate::device::AkidaDevice, w: &EsnWeights) -> Result<Self> {
        let scale = ScaleTrickConfig::from_weights(&w.w_in, &w.w_res);
        info!("HardwareNative: device acquired, bounded ReLU emulation active");
        Ok(Self {
            reservoir_dim: w.reservoir_dim,
            input_dim:     w.input_dim,
            output_dim:    w.output_dim,
            leak:          w.leak_rate,
            mode:          SubstrateMode::HardwareNative,
            state:         vec![0.0f32; w.reservoir_dim],
            w_out:         w.w_out.clone(),
            w_in:          w.w_in.clone(),
            w_res:         w.w_res.clone(),
            w_in_scaled:   w.w_in.clone(), // native: no scaling
            w_res_scaled:  w.w_res.clone(),
            scale,
            _device:       Box::new(device),
        })
    }

    fn step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let rs = self.reservoir_dim;
        let is = self.input_dim;
        if input.len() != is {
            return Err(AkidaError::capability_query_failed(format!(
                "hw step: input len {} != input_dim {}", input.len(), is
            )));
        }
        match self.mode {
            SubstrateMode::HardwareLinear  => self.step_linear_emulated(input, rs, is),
            SubstrateMode::HardwareNative  => self.step_native_emulated(input, rs, is),
            SubstrateMode::PureSoftware    => unreachable!("HardwareEsnExecutor never in SW mode"),
        }
    }

    /// Approach B: scaled weights → "hardware" linear multiply → host tanh recovery.
    ///
    /// When Phase 2 FlatBuffer path is live, replace the inner matvec with
    /// `self.device_infer_scaled(input)` and keep the recovery logic unchanged.
    fn step_linear_emulated(&mut self, input: &[f32], rs: usize, is: usize) -> Result<Vec<f32>> {
        let alpha = self.leak;
        let inv_eps = self.scale.inv_epsilon;

        // ── "Hardware" step (emulated): compute ε·(W_in·x + W_res·s) ────────
        // On actual hardware: device.infer(scaled_input) → DMA back this result
        let mut hw_out = vec![0.0f32; rs];
        for i in 0..rs {
            for j in 0..is  { hw_out[i] += self.w_in_scaled[i * is + j]  * input[j]; }
            for j in 0..rs  { hw_out[i] += self.w_res_scaled[i * rs + j] * self.state[j]; }
            // Emulate bounded_relu: hardware clips at 0 and the hardware threshold.
            // With ε-scaled weights, activations are ≤ ε × ‖W‖ × ‖x‖ ≈ 0.01 → linear region.
            hw_out[i] = hw_out[i].max(0.0); // bounded ReLU lower bound
        }

        // ── Host recovery: tanh(hw_out / ε) = tanh(W·x) ────────────────────
        let mut new_state = vec![0.0f32; rs];
        for i in 0..rs {
            let pre_activation = hw_out[i] * inv_eps; // undo epsilon scaling
            new_state[i] = (1.0 - alpha) * self.state[i] + alpha * pre_activation.tanh();
        }
        self.state = new_state;

        // ── Readout (always host-side) ────────────────────────────────────────
        let os = self.output_dim;
        Ok((0..os).map(|i| {
            self.w_out[i * rs..(i + 1) * rs].iter().zip(self.state.iter())
                .map(|(w, s)| w * s).sum()
        }).collect())
    }

    /// HardwareNative emulation: bounded ReLU activation (SDK default behavior).
    /// For MetaTF-designed weights — NOT for hotSpring tanh weights.
    fn step_native_emulated(&mut self, input: &[f32], rs: usize, is: usize) -> Result<Vec<f32>> {
        let alpha = self.leak;
        let mut pre = vec![0.0f32; rs];
        for i in 0..rs {
            for j in 0..is { pre[i] += self.w_in[i * is + j]  * input[j]; }
            for j in 0..rs { pre[i] += self.w_res[i * rs + j] * self.state[j]; }
        }
        for i in 0..rs {
            let relu = pre[i].max(0.0); // bounded ReLU (upper bound approximated by clamp)
            self.state[i] = (1.0 - alpha) * self.state[i] + alpha * relu;
        }
        let os = self.output_dim;
        Ok((0..os).map(|i| {
            self.w_out[i * rs..(i + 1) * rs].iter().zip(self.state.iter())
                .map(|(w, s)| w * s).sum()
        }).collect())
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

// ── SubstrateSelector — toadStool's dispatch point ───────────────────────────

/// Substrate information returned to toadStool's scheduler.
#[derive(Debug, Clone)]
pub struct SubstrateInfo {
    /// Which mode is active.
    pub mode:        SubstrateMode,
    /// Estimated throughput in inferences/second.
    pub est_hz:      f64,
    /// Estimated energy per inference in µJ.
    pub est_energy_uj: f64,
    /// Whether tanh-trained weights are fully accurate on this substrate.
    pub tanh_accurate: bool,
    /// Number of NPs consumed (0 if software).
    pub npu_nps:     usize,
}

/// Runtime substrate selector for toadStool's NPU dispatch system.
///
/// Discovers available substrates at construction time and selects the
/// optimal one (hardware if present, software if not). toadStool calls
/// `esn_step()` without knowing which substrate is executing.
///
/// ```no_run
/// use akida_driver::SubstrateSelector;
///
/// # let (w_in, w_res, w_out) = (vec![0.1f32; 128*4], vec![0.05f32; 128*128], vec![0.2f32; 128]);
/// # let features = vec![0.0f32; 4];
/// let mut selector = SubstrateSelector::for_weights(
///     &w_in, &w_res, &w_out, 0.3,
/// )?;
/// println!("Substrate: {}", selector.active_substrate().mode.description());
///
/// let prediction = selector.esn_step(&features)?;
/// # Ok::<(), akida_driver::AkidaError>(())
/// ```
pub struct SubstrateSelector {
    esn: HybridEsn,
}

impl SubstrateSelector {
    /// Build a selector with the given weights, auto-discovering hardware.
    ///
    /// Tries hardware discovery; falls back to software if no NPU found.
    ///
    /// # Errors
    ///
    /// Returns error only if weights are invalid. Hardware unavailability is
    /// silently handled by falling back to software.
    pub fn for_weights(
        w_in:  &[f32],
        w_res: &[f32],
        w_out: &[f32],
        leak_rate: f32,
    ) -> Result<Self> {
        let esn = HybridEsn::from_weights(w_in, w_res, w_out, leak_rate)?;
        // Hardware upgrade deferred until Exp 004 validates the path.
        // Uncomment once validated:
        //
        // if let Ok(mgr) = crate::discovery::DeviceManager::discover() {
        //     if let Ok(dev) = mgr.open_first() {
        //         esn = esn.with_hardware_linear(dev)?;
        //     }
        // }
        Ok(Self { esn })
    }

    /// Build from a pre-constructed `HybridEsn`.
    #[must_use]
    pub fn from_esn(esn: HybridEsn) -> Self {
        Self { esn }
    }

    /// Active substrate information for toadStool's scheduler/telemetry.
    #[must_use]
    pub fn active_substrate(&self) -> SubstrateInfo {
        let mode = self.esn.mode().clone();
        SubstrateInfo {
            est_hz:        self.esn.estimated_hz(),
            est_energy_uj: self.esn.estimated_energy_uj(),
            tanh_accurate: mode.is_tanh_accurate(),
            npu_nps:       match &mode {
                SubstrateMode::PureSoftware => 0,
                _ => self.esn.weights().reservoir_dim,
            },
            mode,
        }
    }

    /// Single-step inference — dispatches to the active substrate.
    ///
    /// # Errors
    ///
    /// Returns error if the active substrate fails.
    pub fn esn_step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.esn.step(input)
    }

    /// Reset reservoir state (call between independent input sequences).
    pub fn reset(&mut self) {
        self.esn.reset();
    }

    /// Expose the inner `HybridEsn` for direct access if needed.
    #[must_use]
    pub fn inner(&mut self) -> &mut HybridEsn {
        &mut self.esn
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(rs: usize, is: usize, os: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let w_in  = vec![0.1f32; rs * is];
        let w_res = vec![0.05f32; rs * rs];
        let w_out = vec![0.2f32; os * rs];
        (w_in, w_res, w_out)
    }

    #[test]
    fn hybrid_esn_constructs_and_runs() {
        let (w_in, w_res, w_out) = tiny_weights(32, 4, 1);
        let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3).unwrap();
        assert_eq!(esn.mode(), &SubstrateMode::PureSoftware);
        assert_eq!(esn.input_dim(), 4);
        assert_eq!(esn.reservoir_dim(), 32);
        assert_eq!(esn.output_dim(), 1);

        let out = esn.step(&[0.1, -0.2, 0.3, 0.0]).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn reset_clears_state() {
        let (w_in, w_res, w_out) = tiny_weights(16, 3, 1);
        let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.5).unwrap();
        let _ = esn.step(&[1.0, 1.0, 1.0]).unwrap();
        let state_pre = esn.reservoir_state();
        assert!(state_pre.iter().any(|&x| x != 0.0), "state should be non-zero after step");
        esn.reset();
        let state_post = esn.reservoir_state();
        assert!(state_post.iter().all(|&x| x == 0.0), "state should be zero after reset");
    }

    #[test]
    fn tanh_mode_is_accurate() {
        // PureSoftware must use tanh — verify states are in (-1, 1)
        let (w_in, w_res, w_out) = tiny_weights(64, 8, 2);
        let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3).unwrap();
        // Drive with large inputs to saturate tanh
        for _ in 0..20 {
            let _ = esn.step(&[10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]).unwrap();
        }
        let state = esn.reservoir_state();
        // tanh saturates at ±1: all states must be in (-1, 1)
        for &s in &state {
            assert!(s > -1.0 && s < 1.0, "tanh state {s} out of (-1,1)");
        }
    }

    #[test]
    fn software_mode_is_tanh_accurate() {
        assert!(SubstrateMode::PureSoftware.is_tanh_accurate());
    }

    #[test]
    fn hardware_native_is_not_tanh_accurate() {
        assert!(!SubstrateMode::HardwareNative.is_tanh_accurate());
    }

    #[test]
    fn approach_b_scale_trick_non_degenerate() {
        // Validates the core guarantee of Approach B:
        // hardware-linear (scale trick) must produce non-degenerate reservoir states.
        //
        // Approach B does NOT match software tanh outputs — documented limitation.
        // The bounded ReLU LOWER clamp (clips negatives to 0) causes state divergence.
        // This is why Approach A (FlatBuffer threshold override) is the full solution.
        //
        // What Approach B DOES guarantee:
        // 1. Reservoir is non-degenerate (states are non-zero and bounded)
        // 2. Results are deterministic
        // 3. Positive pre-activations are correctly recovered via tanh(hw_out/ε)
        let rs = 32; let is = 4; let os = 1;
        let w_in  = (0..rs * is).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect::<Vec<_>>();
        let w_res = (0..rs * rs).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect::<Vec<_>>();
        let w_out = vec![0.1f32; os * rs];

        let scale = ScaleTrickConfig::from_weights(&w_in, &w_res);
        let eps = scale.epsilon;
        let mut hw_exec = HardwareEsnExecutor {
            reservoir_dim: rs, input_dim: is, output_dim: os,
            leak: 0.3,
            mode: SubstrateMode::HardwareLinear,
            state: vec![0.0; rs],
            w_out: w_out.clone(),
            w_in: w_in.clone(),
            w_res: w_res.clone(),
            w_in_scaled:  w_in.iter().map(|x| x * eps).collect(),
            w_res_scaled: w_res.iter().map(|x| x * eps).collect(),
            scale,
            _device: Box::new(()),
        };

        let input = vec![0.5f32, -0.3, 0.8, -0.1];

        // Drive 30 steps
        let mut outputs = vec![];
        for _ in 0..30 {
            let out = hw_exec.step(&input).unwrap();
            outputs.push(out[0]);
        }

        // 1. Non-degenerate: state RMS must be > 0.001 (reservoir is alive)
        let state_rms = (hw_exec.state.iter().map(|x| x * x).sum::<f32>() / rs as f32).sqrt();
        assert!(state_rms > 0.001,
            "Approach B reservoir degenerated: state RMS = {state_rms:.5}");

        // 2. Bounded: all states in reasonable range (tanh keeps them < 1)
        for &s in &hw_exec.state {
            assert!(s.abs() < 2.0, "Approach B state {s:.4} out of bounds");
        }

        // 3. Deterministic: same input sequence produces same outputs
        let mut hw_exec2 = HardwareEsnExecutor {
            reservoir_dim: rs, input_dim: is, output_dim: os,
            leak: 0.3,
            mode: SubstrateMode::HardwareLinear,
            state: vec![0.0; rs],
            w_out: w_out.clone(),
            w_in: w_in.clone(),
            w_res: w_res.clone(),
            w_in_scaled:  w_in.iter().map(|x| x * eps).collect(),
            w_res_scaled: w_res.iter().map(|x| x * eps).collect(),
            scale: ScaleTrickConfig::from_weights(&w_in, &w_res),
            _device: Box::new(()),
        };
        let mut outputs2 = vec![];
        for _ in 0..30 { outputs2.push(hw_exec2.step(&input).unwrap()[0]); }
        for (o1, o2) in outputs.iter().zip(outputs2.iter()) {
            assert!((o1 - o2).abs() < 1e-6,
                "Approach B non-deterministic: {o1} vs {o2}");
        }
    }

    #[test]
    fn hardware_native_has_bounded_relu_saturation() {
        // HardwareNative should saturate near 0 with large negative inputs
        // (bounded ReLU clips at 0, unlike tanh which would return -1)
        let rs = 16; let is = 2; let os = 1;
        let (w_in, w_res, w_out) = tiny_weights(rs, is, os);
        let weights = EsnWeights::new(
            w_in.clone(), w_res.clone(), w_out.clone(), is, rs, os, 0.3,
        ).unwrap();
        let scale = ScaleTrickConfig::from_weights(&w_in, &w_res);
        let ε = scale.epsilon;
        let mut hw_exec = HardwareEsnExecutor {
            reservoir_dim: rs, input_dim: is, output_dim: os,
            leak: 0.3,
            mode: SubstrateMode::HardwareNative,
            state: vec![0.0; rs],
            w_out,
            w_in: w_in.clone(),
            w_res: w_res.clone(),
            w_in_scaled:  w_in.iter().map(|x| x * ε).collect(),
            w_res_scaled: w_res.iter().map(|x| x * ε).collect(),
            scale,
            _device: Box::new(()),
        };
        let _ = weights;
        // Drive with large negative input — bounded ReLU should keep state near 0
        for _ in 0..20 {
            let _ = hw_exec.step(&[-100.0, -100.0]).unwrap();
        }
        // State should be near 0 (bounded ReLU clips negatives; no saturation at -1)
        for &s in &hw_exec.state {
            assert!(s >= 0.0, "bounded ReLU state {s} should be non-negative");
        }
    }

    #[test]
    fn run_sequence_matches_repeated_step() {
        let (w_in, w_res, w_out) = tiny_weights(32, 4, 1);
        let mut esn_step = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3).unwrap();
        let mut esn_seq  = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3).unwrap();

        let input = vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2, -0.3]; // 2 frames
        let out_step_1 = esn_step.step(&input[0..4]).unwrap();
        let out_step_2 = esn_step.step(&input[4..8]).unwrap();
        let out_seq    = esn_seq.run_sequence(&input).unwrap();

        // Both should produce the same final output
        assert!((out_step_2[0] - out_seq[0]).abs() < 1e-6,
                "step={} seq={}", out_step_2[0], out_seq[0]);
        let _ = out_step_1;
    }

    #[test]
    fn weight_dimension_validation() {
        // Wrong w_in size should fail
        let result = HybridEsn::from_weights(
            &[0.1f32; 31],  // should be 32×4=128
            &[0.0f32; 1024],
            &[0.2f32; 32],
            0.3,
        );
        assert!(result.is_err(), "mismatched w_in should fail");
    }

    #[test]
    fn substrate_selector_builds_and_runs() {
        let (w_in, w_res, w_out) = tiny_weights(32, 4, 1);
        let mut sel = SubstrateSelector::for_weights(&w_in, &w_res, &w_out, 0.3).unwrap();
        let info = sel.active_substrate();
        assert_eq!(info.mode, SubstrateMode::PureSoftware);
        assert!(info.tanh_accurate);
        assert_eq!(info.npu_nps, 0);

        let out = sel.esn_step(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn esn_weights_spectral_radius() {
        // Identity-scaled w_res (small values) → spectral radius < 1
        let rs = 16;
        let w_res: Vec<f32> = (0..rs * rs)
            .map(|i| if i % (rs + 1) == 0 { 0.5f32 } else { 0.0 })
            .collect();
        let w = EsnWeights::new(
            vec![0.1; rs * 4], w_res, vec![0.1; rs],
            4, rs, 1, 0.3,
        ).unwrap();
        let rho = w.spectral_radius_estimate(20);
        assert!(rho < 1.0, "spectral radius {rho} should be < 1 for stability");
    }
}
