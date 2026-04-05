// SPDX-License-Identifier: AGPL-3.0-or-later

//! NPU hardware setup and initialization
//!
//! This module provides pure Rust implementations for setting up Akida NPU hardware,
//! replacing shell scripts with compiled code that's portable across systems.

use crate::error::{AkidaError, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Setup Akida NPU kernel driver
pub struct NpuSetup {
    driver_path: Option<PathBuf>,
    pkexec_available: bool,
}

impl NpuSetup {
    /// Create new NPU setup manager
    pub fn new() -> Self {
        Self {
            driver_path: None,
            pkexec_available: find_in_path("pkexec").is_some(),
        }
    }

    /// Run complete NPU setup
    ///
    /// # Errors
    ///
    /// Returns error if any setup step fails (hardware not found, driver missing,
    /// insufficient permissions, etc.).
    pub fn run(&mut self) -> Result<()> {
        info!("Akida NPU Setup");

        // Check hardware
        check_hardware()?;

        // Find driver
        self.find_driver()?;

        // Enable PCIe devices
        self.enable_pcie_devices()?;

        // Load kernel module
        self.load_kernel_module()?;

        // Verify device nodes
        verify_device_nodes()?;

        // Setup permissions
        self.setup_permissions()?;

        info!("NPU setup complete!");
        Ok(())
    }

    /// Find Akida kernel module
    fn find_driver(&mut self) -> Result<()> {
        info!("Looking for Akida kernel module...");

        // Check if already loaded
        if is_module_loaded()? {
            info!("Kernel module already loaded");
            return Ok(());
        }

        // Build search paths - check environment variable first, then standard locations
        let mut search_paths = Vec::new();

        // 1. Check AKIDA_DRIVER_PATH environment variable (highest priority)
        if let Ok(custom_path) = std::env::var("AKIDA_DRIVER_PATH") {
            search_paths.push(PathBuf::from(custom_path));
        }

        // 2. Standard kernel module locations
        if let Ok(kver) = kernel_version() {
            // Standard extra modules location
            search_paths.push(
                PathBuf::from("/lib/modules")
                    .join(&kver)
                    .join("extra/akida-pcie.ko"),
            );
            // Updates location
            search_paths.push(
                PathBuf::from("/lib/modules")
                    .join(&kver)
                    .join("updates/akida-pcie.ko"),
            );
            // Kernel tree location
            search_paths.push(
                PathBuf::from("/lib/modules")
                    .join(&kver)
                    .join("kernel/drivers/misc/akida-pcie.ko"),
            );
        }

        // 3. System-wide location
        search_paths.push(PathBuf::from("/usr/local/lib/akida/akida-pcie.ko"));

        for path in search_paths {
            if path.exists() {
                info!("Found driver: {}", path.display());
                self.driver_path = Some(path);
                return Ok(());
            }
        }

        Err(AkidaError::hardware_error(
            "Akida kernel module not found. Set AKIDA_DRIVER_PATH environment variable \
            or install the driver to /lib/modules/$(uname -r)/extra/akida-pcie.ko",
        ))
    }

    /// Enable `PCIe` devices
    fn enable_pcie_devices(&self) -> Result<()> {
        info!("Enabling PCIe devices...");

        // Find Akida PCIe addresses using shared constants
        let filter = crate::pcie_ids::lspci_filter();
        let output = Command::new("lspci").arg("-d").arg(&filter).output()?;

        let devices = String::from_utf8_lossy(&output.stdout);

        for line in devices.lines() {
            if let Some(address) = line.split_whitespace().next() {
                let full_address = format!("0000:{address}");
                self.enable_device(&full_address)?;
            }
        }

        Ok(())
    }

    /// Enable single `PCIe` device
    fn enable_device(&self, address: &str) -> Result<()> {
        let enable_path = format!("/sys/bus/pci/devices/{address}/enable");

        debug!("Enabling device: {address}");

        // Try direct write first (if we have permissions)
        if fs::write(&enable_path, "1").is_ok() {
            info!("Enabled {address} (direct)");
            return Ok(());
        }

        // Need privilege escalation
        if !self.pkexec_available {
            return Err(AkidaError::hardware_error(
                "Need root to enable device. Install pkexec or run as root.",
            ));
        }

        let status = Command::new("pkexec")
            .arg("sh")
            .arg("-c")
            .arg(format!("echo 1 > {enable_path}"))
            .status()?;

        if !status.success() {
            return Err(AkidaError::hardware_error(format!(
                "Failed to enable device {address} (pkexec)"
            )));
        }

        info!("Enabled {address} (pkexec)");
        Ok(())
    }

    /// Load kernel module
    fn load_kernel_module(&self) -> Result<()> {
        info!("Loading kernel module...");

        // Already loaded?
        if is_module_loaded()? {
            return Ok(());
        }

        let driver_path = self
            .driver_path
            .as_ref()
            .ok_or_else(|| AkidaError::hardware_error("Driver path not set"))?;

        // Try direct insmod first
        if let Ok(status) = Command::new("insmod").arg(driver_path).status()
            && status.success()
        {
            info!("Module loaded (direct)");
            return Ok(());
        }

        // Need privilege escalation
        if !self.pkexec_available {
            return Err(AkidaError::hardware_error(
                "Need root to load module. Install pkexec or run as root.",
            ));
        }

        let status = Command::new("pkexec")
            .arg("insmod")
            .arg(driver_path)
            .status()?;

        if !status.success() {
            return Err(AkidaError::hardware_error("Failed to load kernel module"));
        }

        info!("Module loaded (pkexec)");

        // Verify it loaded
        if !is_module_loaded()? {
            return Err(AkidaError::hardware_error(
                "Module loaded but not showing in lsmod",
            ));
        }

        Ok(())
    }

    /// Setup device permissions
    fn setup_permissions(&self) -> Result<()> {
        info!("Setting up permissions...");

        // Check if devices are already accessible
        if Path::new("/dev/akida0").exists()
            && !fs::metadata("/dev/akida0")?.permissions().readonly()
        {
            info!("Permissions already OK");
            return Ok(());
        }

        // Need to set permissions
        if !self.pkexec_available {
            warn!("Cannot set permissions without pkexec. Devices may not be accessible.");
            return Ok(());
        }

        let status = Command::new("pkexec")
            .arg("chmod")
            .arg("666")
            .arg("/dev/akida*")
            .status()?;

        if status.success() {
            info!("Permissions set");
        } else {
            warn!("Failed to set permissions. May need manual chmod.");
        }

        Ok(())
    }
}

impl Default for NpuSetup {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if Akida hardware is present (associated function, no self needed)
fn check_hardware() -> Result<()> {
    info!("Checking for Akida hardware...");

    let filter = crate::pcie_ids::lspci_filter();
    let output = Command::new("lspci").arg("-d").arg(&filter).output()?;

    if !output.status.success() || output.stdout.is_empty() {
        return Err(AkidaError::hardware_error(format!(
            "No Akida NPU hardware detected. Run 'lspci -d {filter}' to verify."
        )));
    }

    let devices = String::from_utf8_lossy(&output.stdout);
    let count = devices.lines().count();
    info!("Found {count} Akida device(s)");

    Ok(())
}

/// Check if kernel module is loaded (associated function, no self needed)
fn is_module_loaded() -> Result<bool> {
    let output = Command::new("lsmod").output()?;
    let modules = String::from_utf8_lossy(&output.stdout);
    Ok(modules.contains("akida_pcie"))
}

/// Verify device nodes exist (associated function, no self needed)
fn verify_device_nodes() -> Result<()> {
    info!("Verifying device nodes...");

    // Wait up to 5 seconds for udev
    for _ in 0..50 {
        if Path::new("/dev/akida0").exists() {
            info!("Device nodes created");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    Err(AkidaError::hardware_error(
        "Device nodes not created. Check dmesg for errors.",
    ))
}

/// Get current kernel version
fn kernel_version() -> Result<String> {
    let output = Command::new("uname")
        .arg("-r")
        .output()
        .map_err(AkidaError::from)?;
    String::from_utf8(output.stdout)
        .map(|s| s.trim().to_string())
        .map_err(|e| AkidaError::hardware_error(format!("Invalid kernel version string: {e}")))
}

/// Pure Rust replacement for `which::which()`.
/// Searches PATH for an executable, returning the first match.
fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_version() {
        let version = kernel_version().unwrap();
        assert!(!version.is_empty());
        println!("Kernel version: {version}");
    }

    #[test]
    fn npu_setup_new_and_default_construct() {
        let a = NpuSetup::new();
        let b = NpuSetup::default();
        // Both start with no driver path; pkexec presence is environment-specific.
        assert!(a.driver_path.is_none());
        assert!(b.driver_path.is_none());
    }

    #[test]
    fn find_in_path_finds_common_system_binary() {
        // Exercise PATH search used for pkexec detection (no production change).
        let sh = find_in_path("sh");
        assert!(sh.is_some(), "expected `sh` on PATH in test environment");
        assert!(sh.unwrap().is_file());
    }

    #[test]
    fn find_in_path_returns_none_for_impossible_name() {
        assert!(find_in_path("akida_nonexistent_binary_xyz_42").is_none());
    }

    #[test]
    fn is_module_loaded_runs_lsmod() {
        let loaded = is_module_loaded().expect("lsmod should run");
        // akida_pcie may or may not be present; call must succeed.
        let _ = loaded;
    }
}
