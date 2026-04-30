// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sysfs helpers for PCIe device management.
//!
//! Absorbed from coralReef's `coral-ember/src/sysfs.rs`. Provides safe
//! sysfs reads and writes with D-state isolation (subprocess watchdog).
//!
//! # D-state isolation
//!
//! Sysfs writes to `driver/unbind`, `drivers/*/bind`, and `remove` can
//! enter uninterruptible kernel sleep (D-state). A thread in D-state
//! cannot be killed — even SIGKILL is deferred. To survive this, risky
//! sysfs writes use a short-lived child process with a timeout.

use crate::error::AkidaError;
use crate::Result;
use std::time::Duration;

const SYSFS_WRITE_TIMEOUT: Duration = Duration::from_secs(10);

// ── Path helpers ──────────────────────────────────────────────────────

fn sysfs_pci_device(bdf: &str, attr: &str) -> String {
    format!("/sys/bus/pci/devices/{bdf}/{attr}")
}

fn sysfs_pci_driver_bind(driver: &str) -> String {
    format!("/sys/bus/pci/drivers/{driver}/bind")
}

fn sysfs_pci_driver_unbind(driver: &str) -> String {
    format!("/sys/bus/pci/drivers/{driver}/unbind")
}

// ── Reads ─────────────────────────────────────────────────────────────

/// Read the currently bound driver for a PCI device, or None if unbound.
pub fn read_current_driver(bdf: &str) -> Option<String> {
    std::fs::read_link(sysfs_pci_device(bdf, "driver"))
        .ok()
        .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
}

/// Read the IOMMU group ID for a PCI device.
pub fn read_iommu_group(bdf: &str) -> u32 {
    std::fs::read_link(sysfs_pci_device(bdf, "iommu_group"))
        .ok()
        .and_then(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(0)
}

/// Read a PCI ID field (vendor, device). Returns 0 on failure.
pub fn read_pci_id(bdf: &str, field: &str) -> u16 {
    let path = sysfs_pci_device(bdf, field);
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| {
            let trimmed = s.trim().trim_start_matches("0x");
            u16::from_str_radix(trimmed, 16).ok()
        })
        .unwrap_or(0)
}

/// Read the PCIe power state (D0, D3hot, D3cold, etc.).
pub fn read_power_state(bdf: &str) -> Option<String> {
    let path = sysfs_pci_device(bdf, "power_state");
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

// ── Writes ────────────────────────────────────────────────────────────

/// Direct sysfs write — for paths that never enter D-state
/// (power/control, d3cold_allowed, reset_method, driver_override).
///
/// Empty values are written as `"\n"` because the kernel ignores
/// zero-byte writes (the sysfs store function is never invoked).
pub fn sysfs_write_direct(path: &str, value: &str) -> Result<()> {
    let bytes: &[u8] = if value.is_empty() { b"\n" } else { value.as_bytes() };
    std::fs::write(path, bytes).map_err(|e| {
        AkidaError::hardware_error(format!("sysfs write to {path}: {e}"))
    })
}

/// Process-isolated sysfs write with D-state protection.
///
/// Spawns a child process for the write. If the child enters D-state
/// and doesn't complete within the timeout, it is killed. The calling
/// thread stays responsive in all cases.
pub fn sysfs_write(path: &str, value: &str) -> Result<()> {
    guarded_sysfs_write(path, value, SYSFS_WRITE_TIMEOUT)
}

fn guarded_sysfs_write(path: &str, value: &str, timeout: Duration) -> Result<()> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("/usr/bin/env")
        .args([
            "sh", "-c",
            "printf '%s' \"$1\" > \"$2\"",
            "sysfs_write",
            value,
            path,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AkidaError::hardware_error(format!("sysfs write spawn to {path}: {e}")))?;

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    return Ok(());
                }
                let stderr = child
                    .stderr
                    .take()
                    .and_then(|mut s| {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut s, &mut buf).ok()?;
                        Some(buf)
                    })
                    .unwrap_or_default();
                return Err(AkidaError::hardware_error(format!(
                    "sysfs write to {path} failed ({status}): {stderr}"
                )));
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    tracing::error!(
                        path, value, timeout_secs = timeout.as_secs(),
                        pid = child.id(),
                        "sysfs write TIMED OUT — child likely in D-state, killing"
                    );
                    let _ = child.kill();
                    let reaped = (0..10).any(|_| match child.try_wait() {
                        Ok(Some(_)) | Err(_) => true,
                        Ok(None) => {
                            std::thread::sleep(Duration::from_millis(100));
                            false
                        }
                    });
                    if !reaped {
                        tracing::warn!(
                            path, pid = child.id(),
                            "sysfs write child still in D-state — abandoning zombie"
                        );
                    }
                    return Err(AkidaError::hardware_error(format!(
                        "sysfs write to {path} timed out after {}s (D-state)",
                        timeout.as_secs()
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(AkidaError::hardware_error(format!(
                    "sysfs write waitpid for {path}: {e}"
                )));
            }
        }
    }
}

// ── Power management ──────────────────────────────────────────────────

/// Pin power state to prevent D3 transitions during driver swaps.
pub fn pin_power(bdf: &str) {
    let _ = sysfs_write_direct(&sysfs_pci_device(bdf, "power/control"), "on");
    let _ = sysfs_write_direct(&sysfs_pci_device(bdf, "d3cold_allowed"), "0");
}

/// Pin power on upstream PCI bridges to prevent slot power-down.
pub fn pin_bridge_power(bdf: &str) {
    let device_path = sysfs_pci_device(bdf, "");
    let Ok(real_path) = std::fs::canonicalize(device_path.trim_end_matches('/')) else {
        return;
    };

    let mut current = real_path.parent();
    while let Some(parent) = current {
        let power_control = parent.join("power/control");
        let d3cold = parent.join("d3cold_allowed");

        if power_control.exists() {
            let _ = sysfs_write_direct(power_control.to_str().unwrap_or(""), "on");
            let _ = sysfs_write_direct(d3cold.to_str().unwrap_or(""), "0");
        }

        if parent
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("pci"))
        {
            break;
        }
        current = parent.parent();
    }
}

/// Disable the kernel's reset_method for a device. This prevents
/// firmware-destroying PCI resets during driver transitions.
pub fn disable_reset_method(bdf: &str) {
    let path = sysfs_pci_device(bdf, "reset_method");
    if let Err(e) = sysfs_write_direct(&path, "") {
        tracing::warn!(bdf, error = %e, "failed to disable reset_method");
    } else {
        tracing::debug!(bdf, "reset_method disabled");
    }
}

// ── Driver bind/unbind ────────────────────────────────────────────────

/// Unbind the current driver from a PCI device.
pub fn unbind_driver(bdf: &str, driver: &str) -> Result<()> {
    sysfs_write(&sysfs_pci_driver_unbind(driver), bdf)
}

/// Set the driver_override for a PCI device.
pub fn set_driver_override(bdf: &str, driver: &str) -> Result<()> {
    sysfs_write_direct(&sysfs_pci_device(bdf, "driver_override"), driver)
}

/// Bind a PCI device to a specific driver.
pub fn bind_driver(bdf: &str, driver: &str) -> Result<()> {
    sysfs_write(&sysfs_pci_driver_bind(driver), bdf)
}

/// Trigger the kernel's driver probe for a PCI device.
pub fn drivers_probe(bdf: &str) -> Result<()> {
    sysfs_write("/sys/bus/pci/drivers_probe", bdf)
}

/// Load a kernel module via modprobe. Returns true if successful.
pub fn modprobe(module: &str) -> bool {
    let sysfs_mod = format!("/sys/module/{}", module.replace('-', "_"));
    if std::path::Path::new(&sysfs_mod).exists() {
        tracing::debug!(module, "kernel module already loaded");
        return true;
    }

    tracing::info!(module, "loading kernel module");
    match std::process::Command::new("modprobe").arg(module).output() {
        Ok(out) if out.status.success() => {
            tracing::info!(module, "kernel module loaded");
            std::thread::sleep(Duration::from_millis(500));
            true
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            tracing::warn!(module, %stderr, "modprobe failed");
            false
        }
        Err(e) => {
            tracing::warn!(module, error = %e, "modprobe not available");
            false
        }
    }
}

/// Check if a kernel module is available (installed, not necessarily loaded).
pub fn module_available(module: &str) -> bool {
    std::process::Command::new("modinfo")
        .arg(module)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ── IOMMU group operations ───────────────────────────────────────────

/// Bind all IOMMU group peers to vfio-pci.
pub fn bind_iommu_group_to_vfio(primary_bdf: &str, group_id: u32) {
    for_each_iommu_peer(primary_bdf, group_id, |peer_bdf| {
        let driver = read_current_driver(&peer_bdf);
        if driver.as_deref() == Some("vfio-pci") {
            return;
        }
        tracing::info!(peer = %peer_bdf, group = group_id, "binding IOMMU group peer to vfio-pci");
        if driver.is_some() {
            let _ = sysfs_write(
                &format!("/sys/bus/pci/devices/{peer_bdf}/driver/unbind"),
                &peer_bdf,
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        let _ = set_driver_override(&peer_bdf, "vfio-pci");
        let _ = bind_driver(&peer_bdf, "vfio-pci");
        std::thread::sleep(Duration::from_millis(200));
    });
}

fn for_each_iommu_peer(primary_bdf: &str, group_id: u32, mut f: impl FnMut(String)) {
    let group_path = format!("/sys/kernel/iommu_groups/{group_id}/devices");
    let Ok(entries) = std::fs::read_dir(group_path) else {
        return;
    };
    for entry in entries.flatten() {
        let peer_bdf = entry.file_name().to_string_lossy().to_string();
        if peer_bdf == primary_bdf {
            continue;
        }
        f(peer_bdf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_current_driver_nonexistent_returns_none() {
        assert_eq!(read_current_driver("9999:99:99.9"), None);
    }

    #[test]
    fn read_power_state_nonexistent_returns_none() {
        assert_eq!(read_power_state("9999:99:99.9"), None);
    }

    #[test]
    fn read_pci_id_nonexistent_returns_zero() {
        assert_eq!(read_pci_id("9999:99:99.9", "vendor"), 0);
    }

    #[test]
    fn sysfs_write_direct_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustchip_sysfs_direct_test");
        sysfs_write_direct(path.to_str().unwrap(), "on").unwrap();
        let read_back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, "on");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn guarded_write_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("rustchip_sysfs_guarded_test");
        sysfs_write(path.to_str().unwrap(), "test_value").unwrap();
        let read_back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, "test_value");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn guarded_write_nonexistent_path_is_error() {
        let err = sysfs_write("/nonexistent-rustchip-path/nope", "1").unwrap_err();
        assert!(err.to_string().contains("sysfs write"));
    }
}
