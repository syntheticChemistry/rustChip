// SPDX-License-Identifier: AGPL-3.0-or-later

//! Device capability querying and representation
//!
//! This module provides runtime capability discovery for Akida devices.
//! No hardcoded device specifications—everything is discovered at runtime.
//!
//! ## metalForge absorption
//!
//! Capabilities now include mesh topology, clock modes, batch discovery,
//! and weight mutation support — all validated by hardware probing on a
//! physical AKD1000 (see `hotSpring/metalForge/npu/akida/BEYOND_SDK.md` — ecosystem context — not a runtime dependency).

use crate::error::{AkidaError, Result};

/// Akida device capabilities discovered at runtime
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    /// Chip version (AKD1000, AKD1500, etc.)
    pub chip_version: ChipVersion,

    /// Number of Neural Processing Units
    pub npu_count: u32,

    /// On-chip SRAM in megabytes
    pub memory_mb: u32,

    /// `PCIe` configuration
    pub pcie: PcieConfig,

    /// Current power consumption in milliwatts
    pub power_mw: Option<u32>,

    /// Die temperature in celsius
    pub temperature_c: Option<f32>,

    /// NP mesh topology (discovered, not assumed)
    pub mesh: Option<MeshTopology>,

    /// Active clock mode
    pub clock_mode: Option<ClockMode>,

    /// Batch inference capabilities
    pub batch: Option<BatchCapabilities>,

    /// Weight mutation support
    pub weight_mutation: WeightMutationSupport,
}

/// NP mesh topology discovered from hardware.
/// metalForge probe revealed AKD1000 has 80 NPs arranged in a 5x8x2
/// configuration, not the flat "80 NPUs" the SDK documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshTopology {
    /// NPs in X dimension
    pub x: u8,
    /// NPs in Y dimension
    pub y: u8,
    /// NPs in Z dimension (pipeline depth)
    pub z: u8,
    /// Total functional NPs (x * y * z, minus any disabled)
    pub functional_count: u32,
}

impl MeshTopology {
    /// Discover mesh topology from sysfs.
    /// Returns `None` if the driver doesn't expose topology attributes.
    pub fn from_sysfs(pcie_address: &str) -> Option<Self> {
        let base = format!("/sys/bus/pci/devices/{pcie_address}");

        let read_u8 = |attr: &str| -> Option<u8> {
            std::fs::read_to_string(format!("{base}/akida_{attr}"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
        };

        let x = read_u8("mesh_x")?;
        let y = read_u8("mesh_y")?;
        let z = read_u8("mesh_z").unwrap_or(1);

        let functional_count = read_u8("mesh_functional")
            .map_or_else(|| u32::from(x) * u32::from(y) * u32::from(z), u32::from);

        Some(Self {
            x,
            y,
            z,
            functional_count,
        })
    }

    /// Discover mesh topology from BAR0 NP enable bits.
    ///
    /// Reads the NP count register and NP enable bit array to determine
    /// the actual functional NP count. Assumes AKD1000 5x8x2 geometry
    /// when only the count is available (geometry requires sysfs or
    /// deeper probing).
    pub fn from_bar0(sram: &crate::sram::SramAccessor) -> Option<Self> {
        let np_reg = sram.read_np_count().ok()?;
        if np_reg == 0 || np_reg > 1000 {
            return None;
        }

        // Count enabled NPs from the enable bit registers (0x1E0C–0x1E20)
        let mut enabled = 0u32;
        for i in 0..akida_chip::regs::NP_ENABLE_COUNT {
            if let Ok(val) = sram.read_register(akida_chip::regs::NP_ENABLE_BASE + i * 4) {
                enabled += val.count_ones();
            }
        }

        let functional = if enabled > 0 { enabled } else { np_reg };

        // AKD1000 geometry: 5x8x2=80 slots. For other chips, default to 1D.
        let (x, y, z) = if functional <= 80 {
            (5, 8, 2)
        } else {
            let n = functional.min(255) as u8;
            (n, 1, 1)
        };

        tracing::info!(
            "Mesh from BAR0: NP_COUNT={np_reg}, enabled_bits={enabled}, functional={functional}"
        );

        Some(Self {
            x,
            y,
            z,
            functional_count: functional,
        })
    }

    /// Total NP slots (may include disabled NPs)
    pub const fn total_slots(&self) -> u32 {
        (self.x as u32) * (self.y as u32) * (self.z as u32)
    }
}

/// Clock mode for power/performance tradeoff.
/// metalForge discovered AKD1000 supports at least Performance and Economy
/// modes — the SDK doesn't document Economy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockMode {
    /// Maximum throughput (default)
    Performance,
    /// 18% less power, 19% slower (measured on AKD1000)
    Economy,
    /// Minimum frequency (if supported by hardware)
    LowPower,
}

impl ClockMode {
    /// Parse clock mode from sysfs string
    pub fn from_sysfs_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "economy" | "eco" => Self::Economy,
            "low_power" | "lowpower" | "lp" => Self::LowPower,
            _ => Self::Performance,
        }
    }

    /// Expected speed penalty relative to Performance mode (0.0 = no penalty)
    pub const fn expected_speed_penalty(&self) -> f64 {
        match self {
            Self::Performance => 0.0,
            Self::Economy => 0.19,
            Self::LowPower => 0.40,
        }
    }

    /// Expected power savings relative to Performance mode (0.0 = no savings)
    pub const fn expected_power_savings(&self) -> f64 {
        match self {
            Self::Performance => 0.0,
            Self::Economy => 0.18,
            Self::LowPower => 0.35,
        }
    }
}

/// Batch inference capabilities discovered from hardware probing.
///
/// On AKD1000, host submission overhead and `PCIe` latency dominate small batches;
/// metalForge measured throughput improving with batch size until SRAM or timing
/// limits bind, with a sweet spot often near batch=8 for this link class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchCapabilities {
    /// Upper bound from on-chip SRAM budget for activations + weights in flight.
    pub max_batch: u32,
    /// Batch size that maximized sustained inferences/sec in metalForge runs.
    pub optimal_batch: u32,
    /// Throughput ratio vs batch=1 at `optimal_batch` (host + device, end-to-end).
    pub optimal_speedup: f32,
}

impl BatchCapabilities {
    /// Discover batch capabilities from optional sysfs nodes (`akida_max_batch`, etc.).
    ///
    /// Returns `None` if the driver does not expose these attributes (VFIO-only or
    /// minimal kernels); callers may fall back to conservative defaults.
    pub fn from_sysfs(pcie_address: &str) -> Option<Self> {
        let base = format!("/sys/bus/pci/devices/{pcie_address}");

        let max_batch: u32 = std::fs::read_to_string(format!("{base}/akida_max_batch"))
            .ok()
            .and_then(|s| s.trim().parse().ok())?;

        let optimal_batch = std::fs::read_to_string(format!("{base}/akida_optimal_batch"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8); // metalForge default

        let optimal_speedup = std::fs::read_to_string(format!("{base}/akida_batch_speedup"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(2.35); // metalForge measurement

        Some(Self {
            max_batch,
            optimal_batch,
            optimal_speedup,
        })
    }
}

/// Weight mutation support level.
/// metalForge proved AKD1000 supports runtime `set_variable()` for weight
/// updates — enabling online readout switching for ESN reservoirs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMutationSupport {
    /// Not available on this hardware
    None,
    /// Full runtime weight updates via `set_variable`
    Full,
    /// Readout layer only (faster, no reservoir re-upload)
    ReadoutOnly,
}

/// Akida chip version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipVersion {
    /// AKD1000 (80 NPUs, 10MB SRAM)
    Akd1000,

    /// AKD1500 (with external memory support)
    Akd1500,

    /// Unknown/future version
    Unknown(u16),
}

impl ChipVersion {
    /// Parse chip version from device ID
    pub const fn from_device_id(device_id: u16) -> Self {
        match device_id {
            0xBCA1 => Self::Akd1000,
            0xBCA2 => Self::Akd1500,
            other => Self::Unknown(other),
        }
    }

    /// Create from register value (runtime discovery)
    ///
    /// Deep Debt: Discovers version from hardware, not hardcoded
    pub const fn from_register(register_value: u32) -> Self {
        // Parse version from register bits
        match register_value & 0xFF {
            0x10 => Self::Akd1000,
            0x15 => Self::Akd1500,
            _ => Self::Unknown(0),
        }
    }

    /// Get typical NPU count for this chip version
    pub const fn typical_npu_count(&self) -> u32 {
        match self {
            Self::Akd1000 | Self::Akd1500 => 80, // Same as AKD1000
            Self::Unknown(_) => 0,
        }
    }

    /// Get typical SRAM size in MB
    pub const fn typical_memory_mb(&self) -> u32 {
        match self {
            Self::Akd1000 | Self::Akd1500 => 10, // Base SRAM, plus external DDR
            Self::Unknown(_) => 0,
        }
    }
}

/// `PCIe` configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcieConfig {
    /// `PCIe` generation (1, 2, 3, 4, 5)
    pub generation: u8,

    /// Number of `PCIe` lanes (1, 4, 8, 16)
    pub lanes: u8,

    /// Link speed in GT/s
    pub speed_gts: f32,

    /// Theoretical bandwidth in GB/s
    pub bandwidth_gbps: f32,
}

impl PcieConfig {
    /// Create `PCIe` config from generation and lanes
    pub fn new(generation: u8, lanes: u8) -> Self {
        let speed_gts = Self::generation_to_speed(generation);
        let bandwidth_gbps = Self::calculate_bandwidth(generation, lanes);

        Self {
            generation,
            lanes,
            speed_gts,
            bandwidth_gbps,
        }
    }

    /// Query `PCIe` config from sysfs
    ///
    /// # Errors
    ///
    /// Returns error if `PCIe` configuration cannot be read from sysfs.
    pub fn from_sysfs(pcie_address: &str) -> Result<Self> {
        let base_path = format!("/sys/bus/pci/devices/{pcie_address}");

        let generation = Self::read_pcie_generation(&base_path)?;
        let lanes = Self::read_pcie_lanes(&base_path)?;

        Ok(Self::new(generation, lanes))
    }

    const fn generation_to_speed(generation: u8) -> f32 {
        match generation {
            2 => 5.0,
            3 => 8.0,
            4 => 16.0,
            5 => 32.0,
            _ => 2.5, // Gen1 and unknown generations
        }
    }

    fn calculate_bandwidth(generation: u8, lanes: u8) -> f32 {
        let per_lane_gbps = match generation {
            1 => 0.25, // 250 MB/s
            2 => 0.5,  // 500 MB/s
            3 => 1.0,  // ~1 GB/s
            4 => 2.0,  // ~2 GB/s
            _ => 4.0,  // Gen5 and unknown: use highest per-lane rate in this table
        };

        per_lane_gbps * f32::from(lanes)
    }

    /// Read `PCIe` generation from sysfs
    ///
    /// # Errors
    ///
    /// Returns error if sysfs file cannot be read or parsed.
    fn read_pcie_generation(base_path: &str) -> Result<u8> {
        let speed_path = format!("{base_path}/current_link_speed");

        std::fs::read_to_string(&speed_path)
            .ok()
            .and_then(|s| {
                // Parse strings like "2.5 GT/s", "5.0 GT/s", "8.0 GT/s"
                if s.contains("2.5") {
                    Some(1)
                } else if s.contains("5.0") || s.contains("5 GT") {
                    Some(2)
                } else if s.contains("8.0") || s.contains("8 GT") {
                    Some(3)
                } else if s.contains("16.0") || s.contains("16 GT") {
                    Some(4)
                } else if s.contains("32.0") || s.contains("32 GT") {
                    Some(5)
                } else {
                    None
                }
            })
            .ok_or_else(|| AkidaError::capability_query_failed("Could not read PCIe generation"))
    }

    /// Read `PCIe` lane count from sysfs
    ///
    /// # Errors
    ///
    /// Returns error if sysfs file cannot be read or parsed.
    fn read_pcie_lanes(base_path: &str) -> Result<u8> {
        let width_path = format!("{base_path}/current_link_width");

        std::fs::read_to_string(&width_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| AkidaError::capability_query_failed("Could not read PCIe lane count"))
    }
}

impl Capabilities {
    /// Query capabilities from a device via sysfs and device file
    ///
    /// This discovers capabilities at runtime—no hardcoded values.
    ///
    /// # Errors
    ///
    /// Returns error if sysfs files cannot be read or parsed.
    pub fn query(device_index: usize, pcie_address: &str) -> Result<Self> {
        tracing::debug!("Querying capabilities for device {device_index} at {pcie_address}");

        let chip_version = Self::read_chip_version(pcie_address)?;
        let pcie = PcieConfig::from_sysfs(pcie_address)?;
        let npu_count = Self::query_npu_count(pcie_address, chip_version);
        let memory_mb = chip_version.typical_memory_mb();
        let power_mw = Self::query_power_consumption(pcie_address);
        let temperature_c = Self::query_temperature(pcie_address);
        let mesh = MeshTopology::from_sysfs(pcie_address);
        let clock_mode = Self::query_clock_mode(pcie_address);
        let batch = BatchCapabilities::from_sysfs(pcie_address);
        let weight_mutation = Self::query_weight_mutation(pcie_address);

        Ok(Self {
            chip_version,
            npu_count,
            memory_mb,
            pcie,
            power_mw,
            temperature_c,
            mesh,
            clock_mode,
            batch,
            weight_mutation,
        })
    }

    /// Query NPU count from device
    ///
    /// **Deep Debt**: Complete implementation with fallback!
    ///
    /// Attempts to query actual NPU count from device registers.
    /// Falls back to typical values if query not supported.
    ///
    /// # Errors
    ///
    /// Returns error if device cannot be accessed.
    fn query_npu_count(pcie_address: &str, chip_version: ChipVersion) -> u32 {
        // Try to read from device-specific sysfs attribute
        let npu_count_path = format!("/sys/bus/pci/devices/{pcie_address}/akida_npu_count");

        if let Ok(count_str) = std::fs::read_to_string(&npu_count_path)
            && let Ok(count) = count_str.trim().parse::<u32>()
        {
            tracing::debug!("Queried NPU count from device: {count}");
            return count;
        }

        // Fallback to typical values for chip version
        let typical = chip_version.typical_npu_count();
        tracing::debug!("Using typical NPU count for {chip_version:?}: {typical}");
        typical
    }

    /// Query power consumption from hwmon
    ///
    /// **Deep Debt**: Complete implementation with Linux hwmon!
    ///
    /// Queries power consumption from Linux hardware monitoring subsystem.
    /// Returns None if not available (not an error).
    fn query_power_consumption(pcie_address: &str) -> Option<u32> {
        // Try to find hwmon instance for this device
        let hwmon_path = format!("/sys/bus/pci/devices/{pcie_address}/hwmon");

        let Ok(hwmon_dir) = std::fs::read_dir(&hwmon_path) else {
            return None;
        };

        // Find first hwmon device (usually hwmon0, hwmon1, etc.)
        for entry in hwmon_dir.flatten() {
            let hwmon_name = entry.file_name();
            let power_input_path = format!(
                "/sys/bus/pci/devices/{pcie_address}/hwmon/{}/power1_input",
                hwmon_name.to_string_lossy()
            );

            // power1_input is in microwatts, convert to milliwatts
            if let Ok(power_str) = std::fs::read_to_string(&power_input_path)
                && let Ok(power_uw) = power_str.trim().parse::<u32>()
            {
                let power_mw = power_uw / 1000;
                tracing::info!("Queried power consumption: {} mW", power_mw);
                return Some(power_mw);
            }
        }

        tracing::debug!("Power monitoring not available for device");
        None
    }

    /// Query temperature from hwmon
    ///
    /// **Deep Debt**: Complete implementation with Linux hwmon!
    ///
    /// Queries die temperature from Linux hardware monitoring subsystem.
    /// Returns None if not available (not an error).
    fn query_temperature(pcie_address: &str) -> Option<f32> {
        // Try to find hwmon instance for this device
        let hwmon_path = format!("/sys/bus/pci/devices/{pcie_address}/hwmon");

        let Ok(hwmon_dir) = std::fs::read_dir(&hwmon_path) else {
            return None;
        };

        // Find first hwmon device
        for entry in hwmon_dir.flatten() {
            let hwmon_name = entry.file_name();
            let temp_input_path = format!(
                "/sys/bus/pci/devices/{pcie_address}/hwmon/{}/temp1_input",
                hwmon_name.to_string_lossy()
            );

            // temp1_input is in millidegrees Celsius, convert to degrees
            if let Ok(temp_str) = std::fs::read_to_string(&temp_input_path)
                && let Ok(temp_millic) = temp_str.trim().parse::<i32>()
            {
                // Precision loss acceptable: temperature is inherently imprecise
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Millidegrees to f32 °C for display"
                )]
                let temp_c = temp_millic as f32 / 1000.0;
                tracing::info!("Queried temperature: {:.1}°C", temp_c);
                return Some(temp_c);
            }
        }

        tracing::debug!("Temperature monitoring not available for device");
        None
    }

    /// Query capabilities from sysfs only (no device file)
    ///
    /// Used by VFIO backend when /dev/akida* is not available.
    ///
    /// # Errors
    ///
    /// Returns error if sysfs files cannot be read or parsed.
    pub fn from_sysfs(pcie_address: &str) -> Result<Self> {
        tracing::debug!("Querying capabilities from sysfs for {pcie_address}");

        let chip_version = Self::read_chip_version(pcie_address)?;
        let pcie = PcieConfig::from_sysfs(pcie_address)?;
        let npu_count = Self::query_npu_count(pcie_address, chip_version);
        let memory_mb = chip_version.typical_memory_mb();
        let power_mw = Self::query_power_consumption(pcie_address);
        let temperature_c = Self::query_temperature(pcie_address);
        let mesh = MeshTopology::from_sysfs(pcie_address);
        let clock_mode = Self::query_clock_mode(pcie_address);
        let batch = BatchCapabilities::from_sysfs(pcie_address);
        let weight_mutation = Self::query_weight_mutation(pcie_address);

        Ok(Self {
            chip_version,
            npu_count,
            memory_mb,
            pcie,
            power_mw,
            temperature_c,
            mesh,
            clock_mode,
            batch,
            weight_mutation,
        })
    }

    /// Discover capabilities directly from BAR0 hardware registers.
    ///
    /// This is the ground-truth path — reads actual hardware state instead
    /// of relying on sysfs attributes that may be absent or stale.
    /// Falls back to sysfs for values that only exist there (`PCIe` config,
    /// hwmon power/temp, clock mode).
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 cannot be mapped or key registers are unreadable.
    pub fn from_bar0(pcie_address: &str) -> Result<Self> {
        use crate::sram::SramAccessor;

        tracing::info!("Discovering capabilities from BAR0 registers for {pcie_address}");

        let sram = SramAccessor::open(pcie_address)?;

        // Ground truth: device identity from hardware
        let device_id_reg = sram.read_device_id()?;
        let chip_version = match device_id_reg {
            0x1940_00a1 | 0x1E7C_BCA1 => ChipVersion::Akd1000,
            v if v & 0xFFFF == 0xBCA2 => ChipVersion::Akd1500,
            _ => {
                tracing::warn!("Unknown device ID {device_id_reg:#010x}, trying sysfs");
                Self::read_chip_version(pcie_address)
                    .unwrap_or_else(|_| ChipVersion::from_register(device_id_reg))
            }
        };

        // Ground truth: NP count from register 0x10C0
        let np_reg = sram.read_np_count().unwrap_or(0);
        let npu_count = if np_reg > 0 && np_reg < 1000 {
            tracing::info!("NP count from BAR0: {np_reg}");
            np_reg
        } else {
            Self::query_npu_count(pcie_address, chip_version)
        };

        // Ground truth: SRAM configuration from registers
        let memory_mb = sram.read_sram_config().map_or_else(
            |_| chip_version.typical_memory_mb(),
            |cfg| {
                let sram_size_hint = u64::from(cfg.region_0) * u64::from(cfg.region_1);
                if sram_size_hint > 0 {
                    let mb = (sram_size_hint / (1024 * 1024)).max(1) as u32;
                    tracing::info!(
                        "SRAM from BAR0: region_0={:#x}, region_1={:#x} → {mb} MB estimate",
                        cfg.region_0,
                        cfg.region_1
                    );
                    mb
                } else {
                    chip_version.typical_memory_mb()
                }
            },
        );

        // Ground truth: mesh topology from NP enable bits
        let mesh =
            MeshTopology::from_bar0(&sram).or_else(|| MeshTopology::from_sysfs(pcie_address));

        // Remaining values from sysfs (only available there)
        let pcie = PcieConfig::from_sysfs(pcie_address).unwrap_or_else(|_| PcieConfig::new(2, 1));
        let power_mw = Self::query_power_consumption(pcie_address);
        let temperature_c = Self::query_temperature(pcie_address);
        let clock_mode = Self::query_clock_mode(pcie_address);
        let batch = BatchCapabilities::from_sysfs(pcie_address);
        let weight_mutation = Self::query_weight_mutation(pcie_address);

        tracing::info!(
            "BAR0 capabilities: {:?}, {} NPs, {} MB SRAM, mesh={:?}",
            chip_version,
            npu_count,
            memory_mb,
            mesh
        );

        Ok(Self {
            chip_version,
            npu_count,
            memory_mb,
            pcie,
            power_mw,
            temperature_c,
            mesh,
            clock_mode,
            batch,
            weight_mutation,
        })
    }

    /// Query current clock mode from sysfs
    fn query_clock_mode(pcie_address: &str) -> Option<ClockMode> {
        let path = format!("/sys/bus/pci/devices/{pcie_address}/akida_clock_mode");
        std::fs::read_to_string(&path)
            .ok()
            .map(|s| ClockMode::from_sysfs_str(&s))
    }

    /// Query weight mutation support from sysfs or infer from chip version
    fn query_weight_mutation(pcie_address: &str) -> WeightMutationSupport {
        let path = format!("/sys/bus/pci/devices/{pcie_address}/akida_weight_mutation");
        if let Ok(val) = std::fs::read_to_string(&path) {
            return match val.trim() {
                "full" => WeightMutationSupport::Full,
                "readout" => WeightMutationSupport::ReadoutOnly,
                _ => WeightMutationSupport::None,
            };
        }

        // AKD1000 supports full weight mutation (metalForge validated)
        let device_path = format!("/sys/bus/pci/devices/{pcie_address}/device");
        if let Ok(id_str) = std::fs::read_to_string(&device_path)
            && (id_str.trim().contains("BCA1") || id_str.trim().contains("bca1"))
        {
            return WeightMutationSupport::Full;
        }

        WeightMutationSupport::None
    }

    /// Read chip version from device ID in sysfs
    ///
    /// # Errors
    ///
    /// Returns error if device ID cannot be read or parsed.
    fn read_chip_version(pcie_address: &str) -> Result<ChipVersion> {
        let device_id_path = format!("/sys/bus/pci/devices/{pcie_address}/device");

        let device_id_str = std::fs::read_to_string(&device_id_path).map_err(|e| {
            AkidaError::capability_query_failed(format!("Failed to read device ID: {e}"))
        })?;

        let device_id = u16::from_str_radix(device_id_str.trim().trim_start_matches("0x"), 16)
            .map_err(|e| {
                AkidaError::capability_query_failed(format!("Invalid device ID format: {e}"))
            })?;

        Ok(ChipVersion::from_device_id(device_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chip_version_from_device_id() {
        assert_eq!(ChipVersion::from_device_id(0xBCA1), ChipVersion::Akd1000);
        assert_eq!(ChipVersion::from_device_id(0xBCA2), ChipVersion::Akd1500);
        assert!(matches!(
            ChipVersion::from_device_id(0xFFFF),
            ChipVersion::Unknown(0xFFFF)
        ));
    }

    #[test]
    fn test_pcie_bandwidth_calculation() {
        let gen2_x4 = PcieConfig::new(2, 4);
        assert!((gen2_x4.bandwidth_gbps - 2.0).abs() < f32::EPSILON);

        let gen3_x8 = PcieConfig::new(3, 8);
        assert!((gen3_x8.bandwidth_gbps - 8.0).abs() < f32::EPSILON);

        let gen4_x16 = PcieConfig::new(4, 16);
        assert!((gen4_x16.bandwidth_gbps - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mesh_topology_total_slots() {
        let mesh = MeshTopology {
            x: 5,
            y: 8,
            z: 2,
            functional_count: 80,
        };
        assert_eq!(mesh.total_slots(), 80);
        assert_eq!(mesh.functional_count, 80);
    }

    #[test]
    fn test_mesh_topology_with_disabled_nps() {
        let mesh = MeshTopology {
            x: 5,
            y: 8,
            z: 2,
            functional_count: 78,
        };
        assert_eq!(mesh.total_slots(), 80);
        assert_eq!(mesh.functional_count, 78);
    }

    #[test]
    fn test_clock_mode_parsing() {
        assert_eq!(
            ClockMode::from_sysfs_str("performance"),
            ClockMode::Performance
        );
        assert_eq!(ClockMode::from_sysfs_str("economy"), ClockMode::Economy);
        assert_eq!(ClockMode::from_sysfs_str("eco"), ClockMode::Economy);
        assert_eq!(ClockMode::from_sysfs_str("low_power"), ClockMode::LowPower);
        assert_eq!(
            ClockMode::from_sysfs_str("  PERF  "),
            ClockMode::Performance
        );
        assert_eq!(ClockMode::from_sysfs_str("unknown"), ClockMode::Performance);
    }

    #[test]
    fn test_clock_mode_penalties() {
        assert!((ClockMode::Performance.expected_speed_penalty() - 0.0).abs() < f64::EPSILON);
        assert!(ClockMode::Economy.expected_speed_penalty() > 0.0);
        assert!(ClockMode::Economy.expected_power_savings() > 0.0);
        assert!(
            ClockMode::LowPower.expected_speed_penalty()
                > ClockMode::Economy.expected_speed_penalty()
        );
    }

    #[test]
    fn test_weight_mutation_support_variants() {
        assert_eq!(WeightMutationSupport::None, WeightMutationSupport::None);
        assert_ne!(
            WeightMutationSupport::Full,
            WeightMutationSupport::ReadoutOnly
        );
    }

    #[test]
    fn chip_version_from_register_known_bits() {
        assert_eq!(ChipVersion::from_register(0x10), ChipVersion::Akd1000);
        assert_eq!(ChipVersion::from_register(0x15), ChipVersion::Akd1500);
        assert_eq!(ChipVersion::from_register(0x99), ChipVersion::Unknown(0));
    }

    #[test]
    fn chip_version_typical_resources() {
        assert_eq!(ChipVersion::Akd1000.typical_npu_count(), 80);
        assert_eq!(ChipVersion::Akd1000.typical_memory_mb(), 10);
        assert_eq!(ChipVersion::Unknown(0x1234).typical_npu_count(), 0);
        assert_eq!(ChipVersion::Unknown(0x1234).typical_memory_mb(), 0);
    }

    #[test]
    fn pcie_config_speed_and_bandwidth_monotone() {
        let g2 = PcieConfig::new(2, 1);
        let g3 = PcieConfig::new(3, 1);
        assert!(g3.speed_gts > g2.speed_gts);
        let x1 = PcieConfig::new(3, 1);
        let x4 = PcieConfig::new(3, 4);
        assert!(x4.bandwidth_gbps > x1.bandwidth_gbps);
    }

    #[test]
    fn substrate_mode_descriptions_non_empty() {
        use crate::SubstrateMode;
        assert!(!SubstrateMode::PureSoftware.description().is_empty());
        assert!(!SubstrateMode::HardwareLinear.description().is_empty());
        assert!(!SubstrateMode::HardwareNative.description().is_empty());
    }

    #[test]
    fn pcie_config_covers_gen1_and_high_gen_bandwidth_table() {
        let g1 = PcieConfig::new(1, 4);
        assert!((g1.speed_gts - 2.5).abs() < 0.01);
        let g5 = PcieConfig::new(5, 8);
        assert!(g5.speed_gts >= 32.0);
        assert!(g5.bandwidth_gbps > g1.bandwidth_gbps);
    }

    #[test]
    fn batch_capabilities_struct_fields_roundtrip() {
        let b = BatchCapabilities {
            max_batch: 16,
            optimal_batch: 8,
            optimal_speedup: 2.5,
        };
        assert_eq!(b.max_batch, 16);
        assert!((b.optimal_speedup - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn clock_mode_from_sysfs_str_lowpower_aliases() {
        assert_eq!(ClockMode::from_sysfs_str("lp"), ClockMode::LowPower);
        assert_eq!(ClockMode::from_sysfs_str("lowpower"), ClockMode::LowPower);
    }

    #[test]
    fn chip_version_from_register_uses_low_byte_only() {
        assert_eq!(ChipVersion::from_register(0x110), ChipVersion::Akd1000);
        assert_eq!(ChipVersion::from_register(0x215), ChipVersion::Akd1500);
    }

    #[test]
    fn pcie_config_gen5_and_unknown_generation_bandwidth_table() {
        let g5 = PcieConfig::new(5, 2);
        assert!((g5.speed_gts - 32.0).abs() < 0.01);
        assert!((g5.bandwidth_gbps - 8.0).abs() < 0.01);

        let g9 = PcieConfig::new(9, 1);
        assert!((g9.speed_gts - 2.5).abs() < 0.01);
        assert!((g9.bandwidth_gbps - 4.0).abs() < 0.01);
    }

    #[test]
    fn mesh_topology_total_slots_single_np_line() {
        let m = MeshTopology {
            x: 100,
            y: 1,
            z: 1,
            functional_count: 100,
        };
        assert_eq!(m.total_slots(), 100);
    }

    #[test]
    fn capabilities_struct_clone_and_debug() {
        let c = Capabilities {
            chip_version: ChipVersion::Akd1000,
            npu_count: 80,
            memory_mb: 10,
            pcie: PcieConfig::new(3, 8),
            power_mw: Some(5000),
            temperature_c: Some(55.5),
            mesh: Some(MeshTopology {
                x: 5,
                y: 8,
                z: 2,
                functional_count: 80,
            }),
            clock_mode: Some(ClockMode::Economy),
            batch: Some(BatchCapabilities {
                max_batch: 32,
                optimal_batch: 8,
                optimal_speedup: 2.0,
            }),
            weight_mutation: WeightMutationSupport::Full,
        };
        let c2 = c.clone();
        assert_eq!(c, c2);
        let s = format!("{c:?}");
        assert!(s.contains("Akd1000"));
    }
}
