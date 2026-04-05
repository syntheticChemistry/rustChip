// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared utilities for Akida benchmark binaries.
//!
//! Provides capability-based hardware detection and common benchmark helpers.
//! No hardcoded device paths — all hardware access goes through runtime discovery.

#![forbid(unsafe_code)]
#![warn(clippy::expect_used, clippy::unwrap_used)]

use akida_driver::DeviceManager;
use std::time::Instant;

/// Detected hardware environment, discovered at runtime.
pub struct HardwareProbe {
    manager: Option<DeviceManager>,
}

impl HardwareProbe {
    /// Probe for available Akida hardware via runtime discovery.
    /// Never fails — returns a probe with `is_available() == false` if no hardware.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            manager: DeviceManager::discover().ok(),
        }
    }

    /// Whether any Akida hardware is present.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.manager.is_some()
    }

    /// Number of discovered devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.manager.as_ref().map_or(0, DeviceManager::device_count)
    }

    /// Borrow the device manager (only available when hardware is present).
    #[must_use]
    pub const fn manager(&self) -> Option<&DeviceManager> {
        self.manager.as_ref()
    }

    /// Human-readable status string for benchmark output.
    #[must_use]
    pub fn status_line(&self) -> String {
        self.manager.as_ref().map_or_else(
            || "Hardware: not detected (software mode)".to_string(),
            |mgr| {
                let count = mgr.device_count();
                mgr.devices().first().map_or_else(
                    || format!("Hardware: {count} device(s)"),
                    |dev| {
                        format!(
                            "Hardware: {} device(s), {:?} @ {} ({} NPUs, {} MB SRAM)",
                            count,
                            dev.capabilities.chip_version,
                            dev.pcie_address,
                            dev.capabilities.npu_count,
                            dev.capabilities.memory_mb,
                        )
                    },
                )
            },
        )
    }
}

/// Simple PRNG (xoshiro256++) for reproducible benchmarks.
/// Avoids pulling in `rand` as a dependency.
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    /// Create with a seed. Deterministic — same seed, same sequence.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut s = [seed, seed.wrapping_mul(6_364_136_223_846_793_005), 0, 0];
        s[2] = s[0] ^ s[1];
        s[3] = s[1] ^ s[0].rotate_left(17);
        let mut x = Self { s };
        for _ in 0..8 {
            let _ = x.next_u64();
        }
        x
    }

    /// Next u64 value.
    pub const fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Next f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "Integer stats to f64 for benchmark output"
        )]
        let v = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        v
    }

    /// Fill a slice with random f32 values in [-scale, scale].
    pub fn fill_f32(&mut self, buf: &mut [f32], scale: f32) {
        for x in buf.iter_mut() {
            *x = self.next_f32().mul_add(2.0, -1.0) * scale;
        }
    }
}

/// Benchmark timer that collects multiple iterations and reports statistics.
pub struct BenchTimer {
    label: String,
    times: Vec<std::time::Duration>,
}

impl BenchTimer {
    /// Create a new timer with a label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            times: Vec::new(),
        }
    }

    /// Time a single iteration of a closure.
    pub fn time<F: FnMut()>(&mut self, mut f: F) {
        let start = Instant::now();
        f();
        self.times.push(start.elapsed());
    }

    /// Median duration.
    #[must_use]
    pub fn median(&self) -> std::time::Duration {
        let mut sorted = self.times.clone();
        sorted.sort();
        sorted.get(sorted.len() / 2).copied().unwrap_or_default()
    }

    /// Print summary statistics.
    pub fn report(&self) {
        if self.times.is_empty() {
            println!("  {}: no measurements", self.label);
            return;
        }
        let median = self.median();
        let min = self.times.iter().min().copied().unwrap_or_default();
        let max = self.times.iter().max().copied().unwrap_or_default();
        println!(
            "  {}: median={:?}, min={:?}, max={:?} (n={})",
            self.label,
            median,
            min,
            max,
            self.times.len()
        );
    }
}
