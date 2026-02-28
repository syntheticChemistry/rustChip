//! Runtime device discovery
//!
//! Discovers Akida devices at runtime by scanning `/dev/akida*` and PCIe sysfs.
//! No hardcoded device lists—pure runtime discovery following primal self-knowledge pattern.

use crate::capabilities::Capabilities;
use crate::device::AkidaDevice;
use crate::error::{AkidaError, Result};
use std::path::{Path, PathBuf};

/// Device manager for runtime discovery and access
#[derive(Debug)]
pub struct DeviceManager {
    devices: Vec<DeviceInfo>,
}

/// Information about a discovered device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index (0, 1, 2, ...)
    pub index: usize,

    /// Device file path (/dev/akida0, etc.)
    pub path: PathBuf,

    /// PCIe bus address (0000:a1:00.0, etc.)
    pub pcie_address: String,

    /// Device capabilities (discovered at runtime)
    pub capabilities: Capabilities,
}

impl DeviceManager {
    /// Discover all Akida devices on the system
    ///
    /// This scans for `/dev/akida*` devices and queries their capabilities
    /// via sysfs. Pure runtime discovery—no assumptions.
    ///
    /// # Errors
    ///
    /// Returns `AkidaError::NoDevicesFound` if no devices are detected.
    pub fn discover() -> Result<Self> {
        tracing::info!("Discovering Akida devices...");

        let mut devices = Vec::new();

        // Scan for /dev/akida* devices (up to 16)
        for index in 0..16 {
            let path = PathBuf::from(format!("/dev/akida{index}"));

            if !path.exists() {
                continue;
            }

            tracing::debug!("Found device file: {}", path.display());

            // Find corresponding PCIe address
            let pcie_address = Self::find_pcie_address(index)?;

            // Query capabilities
            match Capabilities::query(index, &pcie_address) {
                Ok(capabilities) => {
                    tracing::info!(
                        "Device {}: {} @ {} (PCIe Gen{} x{}, {} NPUs, {}MB)",
                        index,
                        format!("{:?}", capabilities.chip_version),
                        pcie_address,
                        capabilities.pcie.generation,
                        capabilities.pcie.lanes,
                        capabilities.npu_count,
                        capabilities.memory_mb
                    );

                    devices.push(DeviceInfo {
                        index,
                        path,
                        pcie_address,
                        capabilities,
                    });
                }
                Err(e) => {
                    tracing::warn!("Failed to query capabilities for device {index}: {e}");
                }
            }
        }

        if devices.is_empty() {
            tracing::error!("No Akida devices found");
            return Err(AkidaError::NoDevicesFound);
        }

        tracing::info!("Discovered {} Akida device(s)", devices.len());

        Ok(Self { devices })
    }

    /// Get number of discovered devices
    #[must_use]
    pub const fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get slice of all devices
    #[must_use]
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Get device info by index
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidIndex` if the index is out of bounds.
    pub fn device(&self, index: usize) -> Result<&DeviceInfo> {
        self.devices
            .iter()
            .find(|d| d.index == index)
            .ok_or(AkidaError::InvalidIndex {
                index,
                count: self.devices.len(),
            })
    }

    /// Open device by index
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot be opened or the index is invalid.
    pub fn open(&self, index: usize) -> Result<AkidaDevice> {
        let info = self.device(index)?;
        AkidaDevice::open(info)
    }

    /// Open first available device
    ///
    /// # Errors
    ///
    /// Returns an error if no devices are available or the device cannot be opened.
    pub fn open_first(&self) -> Result<AkidaDevice> {
        let info = self.devices.first().ok_or(AkidaError::NoDevicesFound)?;
        AkidaDevice::open(info)
    }

    /// Open all devices
    ///
    /// # Errors
    ///
    /// Returns an error if any device cannot be opened.
    pub fn open_all(&self) -> Result<Vec<AkidaDevice>> {
        self.devices.iter().map(AkidaDevice::open).collect()
    }

    /// Find PCIe address for a device index
    ///
    /// Scans `/sys/bus/pci/devices/*/` for matching Akida vendor/device IDs
    fn find_pcie_address(device_index: usize) -> Result<String> {
        use crate::pcie_ids::{ALL_DEVICE_IDS, BRAINCHIP_VENDOR_ID};

        let pci_devices_path = Path::new("/sys/bus/pci/devices");

        let entries = std::fs::read_dir(pci_devices_path).map_err(|e| {
            AkidaError::capability_query_failed(format!("Cannot read PCIe devices: {e}"))
        })?;

        let mut matches = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();

            // Read vendor ID
            let vendor_id = Self::read_hex_sysfs(&path.join("vendor")).ok();

            // Read device ID
            let device_id = Self::read_hex_sysfs(&path.join("device")).ok();

            if let (Some(vendor), Some(device)) = (vendor_id, device_id) {
                if vendor == BRAINCHIP_VENDOR_ID && ALL_DEVICE_IDS.contains(&device) {
                    let pcie_addr = entry.file_name().to_string_lossy().to_string();
                    matches.push(pcie_addr);
                }
            }
        }

        // Sort to ensure consistent ordering
        matches.sort();

        matches.get(device_index).cloned().ok_or_else(|| {
            AkidaError::capability_query_failed(format!(
                "No PCIe address found for device {device_index}"
            ))
        })
    }

    /// Read a hexadecimal value from sysfs
    fn read_hex_sysfs(path: &Path) -> Result<u16> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AkidaError::capability_query_failed(format!("Cannot read {}: {e}", path.display()))
        })?;

        let trimmed = content.trim().trim_start_matches("0x");

        u16::from_str_radix(trimmed, 16)
            .map_err(|e| AkidaError::capability_query_failed(format!("Invalid hex value: {e}")))
    }
}

impl DeviceInfo {
    /// Get device index
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Get device path
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get PCIe address
    #[must_use]
    pub fn pcie_address(&self) -> &str {
        &self.pcie_address
    }

    /// Get device capabilities
    #[must_use]
    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_discovery() {
        // This test requires actual hardware
        match DeviceManager::discover() {
            Ok(manager) => {
                println!("✅ Found {} device(s)", manager.device_count());
                for device in manager.devices() {
                    println!("  Device {}: {}", device.index, device.path.display());
                    println!("    PCIe: {}", device.pcie_address);
                    println!("    NPUs: {}", device.capabilities.npu_count);
                }
            }
            Err(AkidaError::NoDevicesFound) => {
                println!("ℹ️  No devices found (hardware required)");
            }
            Err(e) => {
                eprintln!("Discovery error (expected if no hardware): {e}");
            }
        }
    }
}
