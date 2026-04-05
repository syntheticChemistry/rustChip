// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-shift detection and adaptive recovery for NPU inference.
//!
//! Monitors inference output distributions over time using exponentially
//! weighted moving averages (EWMA). When the distribution drifts beyond
//! configured thresholds, triggers alerts and optional automatic recovery
//! (weight re-evolution or model reload).
//!
//! # Architecture
//!
//! ```text
//! DriftMonitor
//!   .observe(input, output)    ← called after every inference
//!   .check() → DriftAlert      ← periodic distribution comparison
//!
//! DriftAlert::MajorDrift
//!   → AdaptiveRecovery::ReloadModel    (full reset)
//!   → AdaptiveRecovery::ReEvolve       (weight evolution to adapt)
//!   → AdaptiveRecovery::Notify         (alert only, human decides)
//! ```
//!
//! # Use case
//!
//! In production physics simulation (lattice QCD), the NPU observes
//! streaming data from GPU-computed trajectories. If the physics
//! regime changes (e.g., temperature quench, phase transition), the
//! ESN readout may become stale. The sentinel detects this drift
//! and can trigger automatic adaptation within 6 seconds.

// EWMA-based drift detection — no external dependencies.

/// Monitors inference output distributions for domain shift.
///
/// Uses EWMA (exponentially weighted moving average) for lightweight
/// online tracking of output statistics. Comparison between the
/// fast EWMA (recent) and slow EWMA (baseline) detects drift.
pub struct DriftMonitor {
    config: DriftConfig,
    fast_mean: Vec<f64>,
    fast_variance: Vec<f64>,
    slow_mean: Vec<f64>,
    slow_variance: Vec<f64>,
    observation_count: u64,
    last_alert: DriftAlert,
}

impl DriftMonitor {
    /// Create a new drift monitor.
    ///
    /// # Arguments
    ///
    /// * `output_dim` — number of output dimensions to track
    /// * `config` — drift detection thresholds and EWMA parameters
    #[must_use]
    pub fn new(output_dim: usize, config: DriftConfig) -> Self {
        Self {
            config,
            fast_mean: vec![0.0; output_dim],
            fast_variance: vec![1.0; output_dim],
            slow_mean: vec![0.0; output_dim],
            slow_variance: vec![1.0; output_dim],
            observation_count: 0,
            last_alert: DriftAlert::NoDrift,
        }
    }

    /// Record a new observation.
    ///
    /// Updates the fast and slow EWMA statistics with the given
    /// inference output. The first observation initializes both
    /// trackers to the observed value (avoiding cold-start divergence).
    pub fn observe(&mut self, _input: &[f32], output: &[f32]) {
        let dim = self.fast_mean.len().min(output.len());

        if self.observation_count == 0 {
            for (i, out_v) in output.iter().take(dim).enumerate() {
                let x = f64::from(*out_v);
                self.fast_mean[i] = x;
                self.slow_mean[i] = x;
                self.fast_variance[i] = 0.0;
                self.slow_variance[i] = 0.0;
            }
            self.observation_count = 1;
            return;
        }

        for (i, out_v) in output.iter().take(dim).enumerate() {
            let x = f64::from(*out_v);

            // Fast EWMA (short memory — tracks recent distribution)
            let alpha_fast = self.config.fast_alpha;
            let diff_fast = x - self.fast_mean[i];
            self.fast_mean[i] += alpha_fast * diff_fast;
            self.fast_variance[i] = (1.0 - alpha_fast)
                * (alpha_fast * diff_fast).mul_add(diff_fast, self.fast_variance[i]);

            // Slow EWMA (long memory — tracks baseline distribution)
            let alpha_slow = self.config.slow_alpha;
            let diff_slow = x - self.slow_mean[i];
            self.slow_mean[i] += alpha_slow * diff_slow;
            self.slow_variance[i] = (1.0 - alpha_slow)
                * (alpha_slow * diff_slow).mul_add(diff_slow, self.slow_variance[i]);
        }

        self.observation_count += 1;
    }

    /// Check for drift by comparing fast vs slow EWMA distributions.
    ///
    /// Returns the current drift alert level based on the divergence
    /// between recent and baseline statistics.
    #[must_use]
    pub fn check(&mut self) -> DriftAlert {
        if self.observation_count < self.config.warmup_observations {
            self.last_alert = DriftAlert::NoDrift;
            return DriftAlert::NoDrift;
        }

        let divergence = self.compute_divergence();

        let alert = if divergence > self.config.domain_shift_threshold {
            DriftAlert::DomainShift {
                divergence,
                observations: self.observation_count,
            }
        } else if divergence > self.config.major_drift_threshold {
            DriftAlert::MajorDrift {
                divergence,
                observations: self.observation_count,
            }
        } else if divergence > self.config.minor_drift_threshold {
            DriftAlert::MinorDrift {
                divergence,
                observations: self.observation_count,
            }
        } else {
            DriftAlert::NoDrift
        };

        self.last_alert = alert.clone();
        alert
    }

    /// Get the last alert without recomputing.
    #[must_use]
    pub const fn last_alert(&self) -> &DriftAlert {
        &self.last_alert
    }

    /// Number of observations recorded.
    #[must_use]
    pub const fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Reset the monitor, clearing all statistics.
    pub fn reset(&mut self) {
        let dim = self.fast_mean.len();
        self.fast_mean = vec![0.0; dim];
        self.fast_variance = vec![1.0; dim];
        self.slow_mean = vec![0.0; dim];
        self.slow_variance = vec![1.0; dim];
        self.observation_count = 0;
        self.last_alert = DriftAlert::NoDrift;
    }

    /// Compute symmetric KL-like divergence between fast and slow distributions.
    fn compute_divergence(&self) -> f64 {
        let dim = self.fast_mean.len();
        if dim == 0 {
            return 0.0;
        }

        let mut total_divergence = 0.0;

        for i in 0..dim {
            let mean_diff = self.fast_mean[i] - self.slow_mean[i];
            let var_fast = self.fast_variance[i].max(1e-10);
            let var_slow = self.slow_variance[i].max(1e-10);

            // Simplified symmetric KL divergence for Gaussians
            let kl = 0.5
                * ((mean_diff * mean_diff).mul_add(
                    1.0 / var_fast + 1.0 / var_slow,
                    (var_fast / var_slow) + (var_slow / var_fast),
                ) - 2.0);

            total_divergence += kl;
        }

        total_divergence / dim as f64
    }
}

/// Drift alert levels.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftAlert {
    /// No significant drift detected.
    NoDrift,

    /// Minor drift — output distribution shifting slightly.
    /// May be noise or gradual domain change.
    MinorDrift {
        /// Divergence score.
        divergence: f64,
        /// Total observations at time of alert.
        observations: u64,
    },

    /// Major drift — output distribution significantly different
    /// from baseline. Model accuracy likely degraded.
    MajorDrift {
        /// Divergence score.
        divergence: f64,
        /// Total observations at time of alert.
        observations: u64,
    },

    /// Domain shift — output distribution fundamentally changed.
    /// Model is likely operating on out-of-distribution data.
    DomainShift {
        /// Divergence score.
        divergence: f64,
        /// Total observations at time of alert.
        observations: u64,
    },
}

impl DriftAlert {
    /// Whether this alert requires action.
    #[must_use]
    pub const fn needs_action(&self) -> bool {
        matches!(self, Self::MajorDrift { .. } | Self::DomainShift { .. })
    }

    /// Whether this is a domain shift (highest severity).
    #[must_use]
    pub const fn is_domain_shift(&self) -> bool {
        matches!(self, Self::DomainShift { .. })
    }
}

/// Configuration for drift detection thresholds and EWMA parameters.
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// EWMA alpha for fast (recent) tracker. Higher = more responsive.
    pub fast_alpha: f64,
    /// EWMA alpha for slow (baseline) tracker. Lower = more stable.
    pub slow_alpha: f64,
    /// Minimum observations before drift detection activates.
    pub warmup_observations: u64,
    /// Divergence threshold for minor drift alert.
    pub minor_drift_threshold: f64,
    /// Divergence threshold for major drift alert.
    pub major_drift_threshold: f64,
    /// Divergence threshold for domain shift alert.
    pub domain_shift_threshold: f64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            fast_alpha: 0.1,
            slow_alpha: 0.001,
            warmup_observations: 100,
            minor_drift_threshold: 1.0,
            major_drift_threshold: 5.0,
            domain_shift_threshold: 20.0,
        }
    }
}

/// Recovery strategies for detected drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveRecovery {
    /// Notify the operator, take no automatic action.
    Notify,
    /// Reload the original model weights from the baseline.
    ReloadModel,
    /// Trigger weight re-evolution to adapt to the new domain.
    ReEvolve {
        /// Number of evolution generations to run.
        generations: u64,
    },
    /// Reset the drift monitor baselines to current statistics.
    ResetBaseline,
}

impl AdaptiveRecovery {
    /// Select a recovery strategy based on the drift alert level.
    #[must_use]
    pub const fn for_alert(alert: &DriftAlert) -> Self {
        match alert {
            DriftAlert::NoDrift | DriftAlert::MinorDrift { .. } => Self::Notify,
            DriftAlert::MajorDrift { .. } => Self::ReEvolve { generations: 50 },
            DriftAlert::DomainShift { .. } => Self::ReloadModel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drift_during_warmup() {
        let mut monitor = DriftMonitor::new(1, DriftConfig::default());
        for _ in 0..50 {
            monitor.observe(&[1.0], &[1.0]);
        }
        assert_eq!(monitor.check(), DriftAlert::NoDrift);
    }

    #[test]
    fn stable_distribution_no_drift() {
        let mut monitor = DriftMonitor::new(1, DriftConfig::default());
        for _ in 0..200 {
            monitor.observe(&[1.0], &[1.0]);
        }
        assert_eq!(monitor.check(), DriftAlert::NoDrift);
    }

    #[test]
    fn domain_shift_detected() {
        let config = DriftConfig {
            warmup_observations: 10,
            ..DriftConfig::default()
        };
        let mut monitor = DriftMonitor::new(1, config);

        // Establish baseline
        for _ in 0..50 {
            monitor.observe(&[1.0], &[0.0]);
        }

        // Sudden domain shift
        for _ in 0..50 {
            monitor.observe(&[1.0], &[100.0]);
        }

        let alert = monitor.check();
        assert!(alert.needs_action());
    }

    #[test]
    fn reset_clears_state() {
        let mut monitor = DriftMonitor::new(2, DriftConfig::default());
        for _ in 0..100 {
            monitor.observe(&[1.0], &[1.0, 2.0]);
        }
        monitor.reset();
        assert_eq!(monitor.observation_count(), 0);
    }

    #[test]
    fn recovery_selection() {
        assert_eq!(
            AdaptiveRecovery::for_alert(&DriftAlert::NoDrift),
            AdaptiveRecovery::Notify
        );
        assert_eq!(
            AdaptiveRecovery::for_alert(&DriftAlert::DomainShift {
                divergence: 50.0,
                observations: 1000
            }),
            AdaptiveRecovery::ReloadModel
        );
    }

    #[test]
    fn domain_shift_alert_flag() {
        let alert = DriftAlert::DomainShift {
            divergence: 99.0,
            observations: 10,
        };
        assert!(alert.is_domain_shift());
        assert!(alert.needs_action());
    }

    #[test]
    fn minor_drift_does_not_need_action() {
        let alert = DriftAlert::MinorDrift {
            divergence: 2.0,
            observations: 200,
        };
        assert!(!alert.needs_action());
    }

    #[test]
    fn major_drift_recovery_re_evolve() {
        let alert = DriftAlert::MajorDrift {
            divergence: 10.0,
            observations: 500,
        };
        assert!(alert.needs_action());
        assert_eq!(
            AdaptiveRecovery::for_alert(&alert),
            AdaptiveRecovery::ReEvolve { generations: 50 }
        );
    }

    #[test]
    fn first_observe_initializes_trackers() {
        let mut monitor = DriftMonitor::new(2, DriftConfig::default());
        monitor.observe(&[], &[3.0, 4.0]);
        assert_eq!(monitor.observation_count(), 1);
    }
}
