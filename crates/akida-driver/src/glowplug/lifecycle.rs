// SPDX-License-Identifier: AGPL-3.0-or-later

//! NPU vendor lifecycle hooks for safe driver transitions.
//!
//! Absorbed from coralReef's `coral-ember/src/vendor_lifecycle/`.
//! Simplified to NPU-focused profiles (no GPU/DRM/SMU/HBM2 complexity).
//!
//! For the full multi-vendor GPU lifecycle (NVIDIA Volta quirks, AMD D3cold,
//! Intel Xe, etc.), see `primals/coralReef/crates/coral-ember/src/vendor_lifecycle/`.

use super::sysfs;

/// BrainChip vendor ID in PCI config space.
pub const BRAINCHIP_VENDOR: u16 = 0x1e7c;

/// Vendor-specific lifecycle hooks for NPU driver transitions.
///
/// Implementors encode hardware-specific knowledge about safe driver
/// swaps. Each method maps to a phase of the swap sequence.
pub trait NpuLifecycle: std::fmt::Debug + Send + Sync {
    /// Human-readable chip family description.
    fn description(&self) -> &str;

    /// Called before any driver unbind. Pin power, disable resets, etc.
    fn prepare_for_unbind(&self, bdf: &str);

    /// Seconds to wait for driver initialization after bind.
    fn settle_secs(&self) -> u64;

    /// Called after a driver binds and settles. Re-pin power, etc.
    fn stabilize_after_bind(&self, bdf: &str);

    /// Post-bind health check. Returns false if device is unhealthy.
    fn verify_health(&self, bdf: &str) -> bool;

    /// The kernel module name for the native driver (e.g. "akida_pcie").
    fn native_driver_module(&self) -> &str;

    /// The sysfs driver directory name (may differ from module name).
    fn native_driver_sysfs(&self) -> &str;
}

// ── BrainChip Akida ──────────────────────────────────────────────────

/// BrainChip Akida NPU lifecycle — simple PCIe accelerator profile.
#[derive(Debug)]
pub struct BrainChipLifecycle {
    /// PCI device ID (0xbca1 for AKD1000).
    pub device_id: u16,
}

impl NpuLifecycle for BrainChipLifecycle {
    fn description(&self) -> &str {
        "BrainChip Akida (simple PCIe accelerator, no GPU quirks)"
    }

    fn prepare_for_unbind(&self, bdf: &str) {
        sysfs::pin_power(bdf);
        sysfs::disable_reset_method(bdf);
    }

    fn settle_secs(&self) -> u64 {
        3
    }

    fn stabilize_after_bind(&self, bdf: &str) {
        sysfs::pin_power(bdf);
        sysfs::disable_reset_method(bdf);
    }

    fn verify_health(&self, bdf: &str) -> bool {
        let power = sysfs::read_power_state(bdf);
        if power.as_deref() == Some("D3cold") {
            tracing::error!(bdf, "BrainChip Akida in D3cold after bind");
            return false;
        }
        true
    }

    fn native_driver_module(&self) -> &str {
        "akida_pcie"
    }

    fn native_driver_sysfs(&self) -> &str {
        "akida"
    }
}

// ── Generic NPU ──────────────────────────────────────────────────────

/// Generic NPU lifecycle for devices without vendor-specific quirks.
///
/// Provides conservative defaults. Use as a base for new NPU vendors
/// (Intel Loihi, SynSense, etc.) until empirical testing reveals
/// device-specific requirements.
#[derive(Debug)]
pub struct GenericNpuLifecycle {
    /// PCI vendor ID.
    pub vendor_id: u16,
    /// PCI device ID.
    pub device_id: u16,
    /// Kernel module name for the native driver.
    pub module_name: String,
    /// sysfs driver directory name.
    pub sysfs_name: String,
}

impl NpuLifecycle for GenericNpuLifecycle {
    fn description(&self) -> &str {
        "Generic NPU (conservative defaults)"
    }

    fn prepare_for_unbind(&self, bdf: &str) {
        sysfs::pin_power(bdf);
        sysfs::disable_reset_method(bdf);
    }

    fn settle_secs(&self) -> u64 {
        5
    }

    fn stabilize_after_bind(&self, bdf: &str) {
        sysfs::pin_power(bdf);
    }

    fn verify_health(&self, bdf: &str) -> bool {
        let power = sysfs::read_power_state(bdf);
        power.as_deref() != Some("D3cold")
    }

    fn native_driver_module(&self) -> &str {
        &self.module_name
    }

    fn native_driver_sysfs(&self) -> &str {
        &self.sysfs_name
    }
}

// ── Detection ────────────────────────────────────────────────────────

/// Auto-detect the appropriate lifecycle for a PCI device.
pub fn detect_lifecycle(bdf: &str) -> Box<dyn NpuLifecycle> {
    let vendor_id = sysfs::read_pci_id(bdf, "vendor");
    let device_id = sysfs::read_pci_id(bdf, "device");

    tracing::info!(
        bdf,
        vendor = format!("0x{vendor_id:04x}"),
        device = format!("0x{device_id:04x}"),
        "detecting NPU lifecycle"
    );

    if vendor_id == BRAINCHIP_VENDOR {
        tracing::info!(bdf, "lifecycle: BrainChip Akida");
        Box::new(BrainChipLifecycle { device_id })
    } else {
        tracing::info!(
            bdf,
            vendor = format!("0x{vendor_id:04x}"),
            "lifecycle: unknown NPU vendor, using conservative defaults"
        );
        Box::new(GenericNpuLifecycle {
            vendor_id,
            device_id,
            module_name: String::from("unknown"),
            sysfs_name: String::from("unknown"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brainchip_lifecycle_basics() {
        let lc = BrainChipLifecycle { device_id: 0xbca1 };
        assert!(lc.description().contains("BrainChip"));
        assert_eq!(lc.settle_secs(), 3);
        assert_eq!(lc.native_driver_module(), "akida_pcie");
        assert_eq!(lc.native_driver_sysfs(), "akida");
    }

    #[test]
    fn generic_lifecycle_basics() {
        let lc = GenericNpuLifecycle {
            vendor_id: 0x1234,
            device_id: 0x5678,
            module_name: "test_npu".into(),
            sysfs_name: "test_npu".into(),
        };
        assert!(lc.description().contains("Generic"));
        assert_eq!(lc.settle_secs(), 5);
        assert_eq!(lc.native_driver_module(), "test_npu");
    }
}
