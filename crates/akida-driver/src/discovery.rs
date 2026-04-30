// SPDX-License-Identifier: AGPL-3.0-or-later

//! Runtime device discovery
//!
//! Discovers Akida devices at runtime by scanning `/dev/akida*` and `PCIe` sysfs.
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

    /// `PCIe` bus address (0000:a1:00.0, etc.)
    pub pcie_address: String,

    /// Device capabilities (discovered at runtime)
    pub capabilities: Capabilities,
}

impl DeviceManager {
    /// Discover all Akida devices on the system.
    ///
    /// Two discovery paths, tried in order:
    /// 1. **Kernel driver** — `/dev/akida*` device nodes (classic path)
    /// 2. **VFIO / sysfs** — scan PCIe bus for BrainChip vendor ID, works when
    ///    the device is bound to `vfio-pci` or has no driver at all.
    ///
    /// # Errors
    ///
    /// Returns `AkidaError::NoDevicesFound` if no devices are detected via
    /// either path.
    pub fn discover() -> Result<Self> {
        tracing::info!("Discovering Akida devices...");

        let mut devices = Vec::new();

        // Path 1: kernel driver (/dev/akida*)
        for index in 0..16 {
            let path = PathBuf::from(format!("/dev/akida{index}"));

            if !path.exists() {
                continue;
            }

            tracing::debug!("Found device file: {}", path.display());

            let pcie_address = Self::find_pcie_address(index)?;

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

        // Path 2: VFIO / sysfs fallback — scan PCIe bus for BrainChip devices
        if devices.is_empty() {
            tracing::debug!("No /dev/akida* found, scanning PCIe sysfs for VFIO-bound devices");
            devices = Self::discover_via_sysfs()?;
        }

        if devices.is_empty() {
            tracing::error!("No Akida devices found");
            return Err(AkidaError::NoDevicesFound);
        }

        tracing::info!("Discovered {} Akida device(s)", devices.len());

        Ok(Self { devices })
    }

    /// Scan `/sys/bus/pci/devices/` for BrainChip vendor IDs.
    ///
    /// Works regardless of which driver is bound (vfio-pci, akida_pcie, or
    /// none). This is the pure-VFIO discovery path.
    fn discover_via_sysfs() -> Result<Vec<DeviceInfo>> {
        use crate::pcie_ids::{ALL_DEVICE_IDS, BRAINCHIP_VENDOR_ID};

        let pci_devices_path = Path::new("/sys/bus/pci/devices");
        let entries = std::fs::read_dir(pci_devices_path).map_err(|e| {
            AkidaError::capability_query_failed(format!("Cannot read PCIe devices: {e}"))
        })?;

        let mut pcie_addrs: Vec<String> = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();
            let vendor_id = Self::read_hex_sysfs(&path.join("vendor")).ok();
            let device_id = Self::read_hex_sysfs(&path.join("device")).ok();

            if let (Some(vendor), Some(device)) = (vendor_id, device_id)
                && vendor == BRAINCHIP_VENDOR_ID
                && ALL_DEVICE_IDS.contains(&device)
            {
                pcie_addrs.push(entry.file_name().to_string_lossy().to_string());
            }
        }

        pcie_addrs.sort();

        let mut devices = Vec::new();
        for (index, pcie_address) in pcie_addrs.into_iter().enumerate() {
            let driver_link = format!("/sys/bus/pci/devices/{pcie_address}/driver");
            let driver_name = std::fs::read_link(&driver_link)
                .ok()
                .and_then(|t| t.file_name().map(|n| n.to_string_lossy().to_string()));

            let device_path = match driver_name.as_deref() {
                Some("vfio-pci") => {
                    let group = crate::vfio::iommu_group(&pcie_address)
                        .unwrap_or(0);
                    PathBuf::from(format!("/dev/vfio/{group}"))
                }
                _ => PathBuf::from(format!("/sys/bus/pci/devices/{pcie_address}")),
            };

            match Capabilities::from_sysfs(&pcie_address) {
                Ok(capabilities) => {
                    tracing::info!(
                        "VFIO/sysfs device {}: {:?} @ {} ({} NPUs, {} MB, driver={:?})",
                        index,
                        capabilities.chip_version,
                        pcie_address,
                        capabilities.npu_count,
                        capabilities.memory_mb,
                        driver_name.as_deref().unwrap_or("none"),
                    );

                    devices.push(DeviceInfo {
                        index,
                        path: device_path,
                        pcie_address,
                        capabilities,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "VFIO/sysfs: cannot query capabilities for {pcie_address}: {e}"
                    );
                }
            }
        }

        Ok(devices)
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

    /// Find `PCIe` address for a device index
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

            if let (Some(vendor), Some(device)) = (vendor_id, device_id)
                && vendor == BRAINCHIP_VENDOR_ID
                && ALL_DEVICE_IDS.contains(&device)
            {
                let pcie_addr = entry.file_name().to_string_lossy().to_string();
                matches.push(pcie_addr);
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

    /// Get `PCIe` address
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
    use crate::capabilities::{Capabilities, ChipVersion, PcieConfig, WeightMutationSupport};
    use std::path::{Path, PathBuf};

    fn sample_capabilities() -> Capabilities {
        Capabilities {
            chip_version: ChipVersion::Akd1000,
            npu_count: 80,
            memory_mb: 8,
            pcie: PcieConfig::new(2, 1),
            power_mw: None,
            temperature_c: None,
            mesh: None,
            clock_mode: None,
            batch: None,
            weight_mutation: WeightMutationSupport::None,
        }
    }

    #[test]
    fn device_info_accessors_roundtrip() {
        let caps = sample_capabilities();
        let info = DeviceInfo {
            index: 2,
            path: PathBuf::from("/dev/akida2"),
            pcie_address: "0000:c1:00.0".to_string(),
            capabilities: caps.clone(),
        };
        assert_eq!(info.index(), 2);
        assert_eq!(info.path(), Path::new("/dev/akida2"));
        assert_eq!(info.pcie_address(), "0000:c1:00.0");
        assert_eq!(info.capabilities().npu_count, caps.npu_count);
    }

    #[test]
    fn lspci_filter_matches_vendor_device_pattern() {
        let f = crate::pcie_ids::lspci_filter();
        assert_eq!(f.chars().filter(|c| *c == ':').count(), 1);
        assert!(f.contains(':'));
    }

    #[test]
    fn all_device_ids_is_non_empty() {
        assert!(!crate::pcie_ids::ALL_DEVICE_IDS.is_empty());
    }

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
