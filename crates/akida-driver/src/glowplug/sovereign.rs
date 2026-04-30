// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign NPU boot — firmware init via kernel driver, then VFIO takeover.
//!
//! Absorbed from coralReef's `coral-glowplug/src/sovereign.rs`. This is
//! the NPU-specific version of the sovereign boot pattern:
//!
//! 1. Detect current driver state
//! 2. If cold on vfio-pci → warm via native driver (akida_pcie)
//! 3. Swap back to vfio-pci (firmware state may survive)
//! 4. Verify firmware is alive by probing BAR0 values
//!
//! For the full GPU sovereign boot with BAR0 PMC probing, nouveau warm
//! cycles, golden state recipes, and falcon boot, see:
//! `primals/coralReef/crates/coral-glowplug/src/sovereign.rs`

use super::lifecycle::{NpuLifecycle, detect_lifecycle};
use super::swap;
use super::sysfs;
use crate::vfio::VfioBackend;
use crate::NpuBackend;
use std::time::Instant;

/// Result of the full sovereign boot sequence.
#[derive(Debug)]
pub struct BootResult {
    /// PCI BDF address.
    pub bdf: String,
    /// Driver bound when we started.
    pub initial_driver: Option<String>,
    /// Whether a warm cycle was performed (native driver bound/unbound).
    pub warm_cycle_performed: bool,
    /// Driver bound after boot (should be "vfio-pci").
    pub final_driver: Option<String>,
    /// Whether firmware appears alive on BAR0.
    pub firmware_alive: bool,
    /// Per-step log.
    pub steps: Vec<BootStep>,
    /// Overall success.
    pub success: bool,
    /// Human-readable summary.
    pub summary: String,
}

/// A single step in the boot sequence.
#[derive(Debug)]
pub struct BootStep {
    /// Step name (e.g. "detect_driver", "warm_cycle").
    pub name: String,
    /// Status.
    pub status: StepStatus,
    /// Human-readable detail.
    pub detail: Option<String>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Status of a boot step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// Completed successfully.
    Ok,
    /// Not needed, skipped.
    Skipped,
    /// Failed (see detail).
    Failed,
}

/// Orchestrate the full sovereign boot for an NPU.
///
/// This is the main entry point. It detects the current state, performs
/// a warm cycle if needed, and verifies firmware is alive.
pub fn sovereign_boot(bdf: &str) -> BootResult {
    let lifecycle = detect_lifecycle(bdf);
    sovereign_boot_with_lifecycle(bdf, lifecycle.as_ref())
}

/// Sovereign boot with a specific lifecycle (for testing or custom NPUs).
pub fn sovereign_boot_with_lifecycle(bdf: &str, lifecycle: &dyn NpuLifecycle) -> BootResult {
    let mut steps = Vec::new();
    let start = Instant::now();
    let mut warm_cycle_performed = false;

    let fail = |steps: Vec<BootStep>, summary: String| BootResult {
        bdf: bdf.to_string(),
        initial_driver: sysfs::read_current_driver(bdf),
        warm_cycle_performed: false,
        final_driver: sysfs::read_current_driver(bdf),
        firmware_alive: false,
        steps,
        success: false,
        summary,
    };

    // ── Step 1: Detect current driver ─────────────────────────────────
    let step_start = Instant::now();
    let initial_driver = sysfs::read_current_driver(bdf);
    steps.push(BootStep {
        name: "detect_driver".into(),
        status: StepStatus::Ok,
        detail: Some(format!(
            "driver={}",
            initial_driver.as_deref().unwrap_or("none")
        )),
        duration_ms: step_start.elapsed().as_millis() as u64,
    });

    let driver_name = initial_driver.as_deref().unwrap_or("none");
    let is_vfio = driver_name == "vfio-pci";
    let native_sysfs = lifecycle.native_driver_sysfs();
    let native_module = lifecycle.native_driver_module();
    let is_native = driver_name == native_sysfs || driver_name == native_module;

    // ── Step 2: If on VFIO, probe firmware ────────────────────────────
    if is_vfio {
        let step_start = Instant::now();
        match probe_firmware(bdf) {
            Some(true) => {
                steps.push(BootStep {
                    name: "firmware_probe".into(),
                    status: StepStatus::Ok,
                    detail: Some("firmware already alive — skipping warm cycle".into()),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });

                return BootResult {
                    bdf: bdf.to_string(),
                    initial_driver,
                    warm_cycle_performed: false,
                    final_driver: sysfs::read_current_driver(bdf),
                    firmware_alive: true,
                    steps,
                    success: true,
                    summary: format!(
                        "firmware alive on VFIO — no warm cycle needed ({}ms)",
                        start.elapsed().as_millis()
                    ),
                };
            }
            Some(false) => {
                steps.push(BootStep {
                    name: "firmware_probe".into(),
                    status: StepStatus::Ok,
                    detail: Some("BAR0 shows raw SRAM — firmware not running".into()),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });
            }
            None => {
                steps.push(BootStep {
                    name: "firmware_probe".into(),
                    status: StepStatus::Failed,
                    detail: Some("could not open VFIO device for probing".into()),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });
            }
        }
    }

    // ── Step 3: Check native driver availability ──────────────────────
    let step_start = Instant::now();
    let module_avail = sysfs::module_available(native_module);
    if !module_avail {
        steps.push(BootStep {
            name: "check_native_driver".into(),
            status: StepStatus::Failed,
            detail: Some(format!(
                "{native_module} kernel module not available — cannot warm boot"
            )),
            duration_ms: step_start.elapsed().as_millis() as u64,
        });
        return fail(
            steps,
            format!("{native_module} not available — install BrainChip SDK kernel module"),
        );
    }
    steps.push(BootStep {
        name: "check_native_driver".into(),
        status: StepStatus::Ok,
        detail: Some(format!("{native_module} available")),
        duration_ms: step_start.elapsed().as_millis() as u64,
    });

    // ── Step 4: Warm cycle ────────────────────────────────────────────
    if !is_native {
        // Swap to native driver for firmware init
        let step_start = Instant::now();
        match swap::swap_to_driver(bdf, native_sysfs, lifecycle) {
            Ok(outcome) if outcome.success => {
                warm_cycle_performed = true;
                steps.push(BootStep {
                    name: "swap_to_native".into(),
                    status: StepStatus::Ok,
                    detail: Some(format!(
                        "from={} duration={}ms",
                        outcome.from_driver.as_deref().unwrap_or("none"),
                        outcome.duration.as_millis()
                    )),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });
            }
            Ok(outcome) => {
                steps.push(BootStep {
                    name: "swap_to_native".into(),
                    status: StepStatus::Failed,
                    detail: Some(format!(
                        "swap returned but driver not bound (actual={})",
                        outcome.to_driver.as_deref().unwrap_or("none")
                    )),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });
                return fail(steps, format!("swap to {native_sysfs} failed (not bound)"));
            }
            Err(e) => {
                steps.push(BootStep {
                    name: "swap_to_native".into(),
                    status: StepStatus::Failed,
                    detail: Some(format!("swap error: {e}")),
                    duration_ms: step_start.elapsed().as_millis() as u64,
                });
                return fail(steps, format!("swap to {native_sysfs} failed: {e}"));
            }
        }
    } else {
        steps.push(BootStep {
            name: "swap_to_native".into(),
            status: StepStatus::Skipped,
            detail: Some("already on native driver".into()),
            duration_ms: 0,
        });
    }

    // ── Step 5: Swap back to vfio-pci ─────────────────────────────────
    let step_start = Instant::now();
    match swap::swap_to_driver(bdf, "vfio-pci", lifecycle) {
        Ok(outcome) if outcome.success => {
            if !warm_cycle_performed {
                warm_cycle_performed = true;
            }
            steps.push(BootStep {
                name: "swap_to_vfio".into(),
                status: StepStatus::Ok,
                detail: Some(format!("duration={}ms", outcome.duration.as_millis())),
                duration_ms: step_start.elapsed().as_millis() as u64,
            });
        }
        Ok(outcome) => {
            steps.push(BootStep {
                name: "swap_to_vfio".into(),
                status: StepStatus::Failed,
                detail: Some(format!(
                    "vfio-pci not bound (actual={})",
                    outcome.to_driver.as_deref().unwrap_or("none")
                )),
                duration_ms: step_start.elapsed().as_millis() as u64,
            });
            return fail(steps, "swap back to vfio-pci failed (not bound)".into());
        }
        Err(e) => {
            steps.push(BootStep {
                name: "swap_to_vfio".into(),
                status: StepStatus::Failed,
                detail: Some(format!("swap error: {e}")),
                duration_ms: step_start.elapsed().as_millis() as u64,
            });
            return fail(steps, format!("swap back to vfio-pci failed: {e}"));
        }
    }

    // ── Step 6: Post-warm firmware verification ───────────────────────
    let step_start = Instant::now();
    let firmware_alive = probe_firmware(bdf).unwrap_or(false);
    steps.push(BootStep {
        name: "verify_firmware".into(),
        status: if firmware_alive { StepStatus::Ok } else { StepStatus::Failed },
        detail: Some(if firmware_alive {
            "firmware alive — mailbox registers match documented values".into()
        } else {
            "firmware NOT alive — BAR0 still shows raw SRAM".into()
        }),
        duration_ms: step_start.elapsed().as_millis() as u64,
    });

    let summary = if firmware_alive {
        format!(
            "sovereign boot succeeded — firmware alive ({}ms)",
            start.elapsed().as_millis()
        )
    } else {
        format!(
            "warm cycle completed but firmware did not survive driver swap ({}ms)",
            start.elapsed().as_millis()
        )
    };

    BootResult {
        bdf: bdf.to_string(),
        initial_driver,
        warm_cycle_performed,
        final_driver: sysfs::read_current_driver(bdf),
        firmware_alive,
        steps,
        success: firmware_alive,
        summary,
    }
}

/// Probe BAR0 to check if firmware is running.
///
/// Returns `Some(true)` if firmware identity register matches,
/// `Some(false)` if VFIO opens but firmware is dead,
/// `None` if VFIO cannot open.
fn probe_firmware(bdf: &str) -> Option<bool> {
    let backend = VfioBackend::init(bdf).ok()?;
    let device_id = backend.read_bar0_u32(0x0000);
    // AKD1000 firmware writes 0x194000a1 to offset 0x0000 when alive
    Some(device_id == 0x194000a1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_firmware_nonexistent_returns_none() {
        assert_eq!(probe_firmware("9999:99:99.9"), None);
    }
}
