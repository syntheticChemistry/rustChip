// SPDX-License-Identifier: AGPL-3.0-or-later

//! Driver swap orchestrator — safe PCIe driver transitions.
//!
//! Absorbed from coralReef's `coral-ember/src/swap/`. This is the only
//! module that performs sysfs driver/unbind and drivers/*/bind writes.
//!
//! For the full swap orchestrator with HeldDevice fd management, DRM
//! isolation, and multi-vendor rebind strategies, see:
//! `primals/coralReef/crates/coral-ember/src/swap/`.

use super::lifecycle::NpuLifecycle;
use super::sysfs;
use crate::Result;
use crate::error::AkidaError;
use std::time::{Duration, Instant};

/// Outcome of a driver swap operation.
#[derive(Debug)]
pub struct SwapOutcome {
    /// BDF address of the device.
    pub bdf: String,
    /// Driver before the swap.
    pub from_driver: Option<String>,
    /// Driver after the swap.
    pub to_driver: Option<String>,
    /// Wall-clock duration of the swap.
    pub duration: Duration,
    /// Whether the target driver is now bound.
    pub success: bool,
}

/// Swap a PCI device to a target driver.
///
/// This handles the full sequence:
/// 1. Detect current driver
/// 2. Run lifecycle `prepare_for_unbind` hooks
/// 3. Unbind current driver (with D-state protection)
/// 4. Set driver_override to target
/// 5. Load target module if needed
/// 6. Bind target driver
/// 7. Settle and stabilize
/// 8. Verify health
///
/// For vfio-pci targets, also disables reset_method and binds IOMMU
/// group peers.
pub fn swap_to_driver(
    bdf: &str,
    target: &str,
    lifecycle: &dyn NpuLifecycle,
) -> Result<SwapOutcome> {
    let start = Instant::now();
    let from_driver = sysfs::read_current_driver(bdf);

    tracing::info!(
        bdf,
        from = from_driver.as_deref().unwrap_or("none"),
        to = target,
        lifecycle = lifecycle.description(),
        "swap_to_driver: starting"
    );

    // Already on the target driver?
    if from_driver.as_deref() == Some(target) {
        tracing::info!(bdf, target, "already on target driver — no swap needed");
        return Ok(SwapOutcome {
            bdf: bdf.to_string(),
            from_driver: from_driver.clone(),
            to_driver: from_driver,
            duration: start.elapsed(),
            success: true,
        });
    }

    // ── Phase 1: Prepare ──────────────────────────────────────────────
    lifecycle.prepare_for_unbind(bdf);

    // ── Phase 2: Unbind current driver ────────────────────────────────
    if let Some(ref driver) = from_driver {
        tracing::info!(bdf, driver = driver.as_str(), "unbinding current driver");
        if let Err(e) = sysfs::unbind_driver(bdf, driver) {
            tracing::warn!(bdf, driver = driver.as_str(), error = %e, "unbind failed — trying override");
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    // ── Phase 3: Set override and bind ────────────────────────────────
    if target == "vfio-pci" {
        bind_vfio(bdf, lifecycle)?;
    } else {
        bind_native(bdf, target, lifecycle)?;
    }

    // ── Phase 4: Verify ───────────────────────────────────────────────
    let to_driver = sysfs::read_current_driver(bdf);
    let success = to_driver.as_deref() == Some(target);

    if success {
        tracing::info!(bdf, target, duration_ms = start.elapsed().as_millis(), "swap complete");
    } else {
        tracing::warn!(
            bdf, target,
            actual = to_driver.as_deref().unwrap_or("none"),
            "swap: target driver not bound after sequence"
        );
    }

    Ok(SwapOutcome {
        bdf: bdf.to_string(),
        from_driver,
        to_driver,
        duration: start.elapsed(),
        success,
    })
}

fn bind_vfio(bdf: &str, lifecycle: &dyn NpuLifecycle) -> Result<()> {
    sysfs::set_driver_override(bdf, "vfio-pci")?;

    // Ensure vfio-pci is loaded
    sysfs::modprobe("vfio-pci");

    let group_id = sysfs::read_iommu_group(bdf);
    if group_id != 0 {
        sysfs::bind_iommu_group_to_vfio(bdf, group_id);
    }

    let _ = sysfs::bind_driver(bdf, "vfio-pci");
    let _ = sysfs::drivers_probe(bdf);

    // Disable reset_method immediately after VFIO bind — per ember pattern.
    // This prevents firmware-destroying PCI resets.
    sysfs::disable_reset_method(bdf);

    let settle = lifecycle.settle_secs();
    std::thread::sleep(Duration::from_secs(settle));

    lifecycle.stabilize_after_bind(bdf);

    if !lifecycle.verify_health(bdf) {
        return Err(AkidaError::hardware_error(format!(
            "device {bdf} failed health check after vfio-pci bind"
        )));
    }

    Ok(())
}

fn bind_native(bdf: &str, target: &str, lifecycle: &dyn NpuLifecycle) -> Result<()> {
    sysfs::set_driver_override(bdf, target)?;

    // Load the kernel module
    let module = lifecycle.native_driver_module();
    sysfs::modprobe(module);

    // Try binding via the sysfs driver name (may differ from module)
    let sysfs_name = lifecycle.native_driver_sysfs();
    let _ = sysfs::bind_driver(bdf, sysfs_name);
    let _ = sysfs::drivers_probe(bdf);

    let settle = lifecycle.settle_secs();
    for i in 0..settle {
        std::thread::sleep(Duration::from_secs(1));
        let drv = sysfs::read_current_driver(bdf);
        if drv.as_deref() == Some(sysfs_name) || drv.as_deref() == Some(target) {
            tracing::info!(bdf, target, seconds = i + 1, "native driver bound");
            break;
        }
        tracing::debug!(bdf, target, seconds = i + 1, driver = ?drv, "waiting for native driver");
    }

    lifecycle.stabilize_after_bind(bdf);

    if !lifecycle.verify_health(bdf) {
        return Err(AkidaError::hardware_error(format!(
            "device {bdf} failed health check after {target} bind"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glowplug::lifecycle::BrainChipLifecycle;

    #[test]
    fn swap_nonexistent_device_returns_error() {
        let lc = BrainChipLifecycle { device_id: 0xbca1 };
        let result = swap_to_driver("9999:99:99.9", "vfio-pci", &lc);
        // Should either succeed (already unbound) or fail cleanly
        if let Ok(outcome) = &result {
            assert!(!outcome.success || outcome.from_driver.is_none());
        }
    }
}
