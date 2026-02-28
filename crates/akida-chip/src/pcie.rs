//! PCIe identifiers and timing constants.
//!
//! Source: AKD1000 hardware + AKD1500 datasheet v1.2 (June 2025).

/// BrainChip vendor ID (PCI-SIG assigned).
pub const BRAINCHIP_VENDOR_ID: u16 = 0x1E7C;

/// Device IDs for Akida family.
pub mod device_id {
    /// AKD1000 — PCIe x1 Gen2 reference board (`lspci: 1e7c:bca1`).
    pub const AKD1000: u16 = 0xBCA1;
    /// AKD1500 — PCIe x2 Gen2, 7×7 mm BGA169 (`lspci: 1e7c:a500`).
    pub const AKD1500: u16 = 0xA500;
    /// AKD1500 alternate ID (some revisions).
    pub const AKD1500_ALT: u16 = 0xBCA2;
}

/// All known Akida device IDs.
pub const ALL_DEVICE_IDS: &[u16] = &[
    device_id::AKD1000,
    device_id::AKD1500,
    device_id::AKD1500_ALT,
];

/// Measured PCIe Gen2 x1 round-trip latency (µs).
///
/// Minimum achievable latency for a write + read cycle over PCIe x1 Gen2.
/// This is hardware-limited — no software optimisation can beat it.
/// Source: BEYOND_SDK.md Discovery 3.
pub const PCIE_GEN2_X1_ROUNDTRIP_US: u64 = 650;

/// Optimal batch size for PCIe amortisation (Discovery 3).
///
/// `batch=8` gives 2.4× throughput over `batch=1` by spreading the
/// ~650 µs PCIe round-trip cost across 8 inference samples.
pub const OPTIMAL_BATCH_SIZE: usize = 8;

/// Sustained DMA throughput measured on AKD1000 (MB/s).
///
/// Source: wetSpring Exp 194, hotSpring Exp 022 (Feb 2026).
pub const MEASURED_DMA_THROUGHPUT_MB_S: u32 = 37;

/// Format a `vendor:device` string for use with `lspci -d`.
#[must_use]
pub fn lspci_filter() -> String {
    format!("{:04x}:{:04x}", BRAINCHIP_VENDOR_ID, device_id::AKD1000)
}

/// Chip variant discovered at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChipVariant {
    /// AKD1000 — PCIe x1 Gen2, 80 NPs, 8 MB SRAM.
    Akd1000,
    /// AKD1500 — PCIe x2 Gen2, BGA169, GPIO, SPI, SLEEP pin.
    Akd1500,
    /// Unknown / future variant.
    Unknown(u16),
}

impl ChipVariant {
    /// Identify variant from PCI device ID.
    #[must_use]
    pub const fn from_device_id(id: u16) -> Self {
        match id {
            device_id::AKD1000 => Self::Akd1000,
            device_id::AKD1500 | device_id::AKD1500_ALT => Self::Akd1500,
            other => Self::Unknown(other),
        }
    }

    /// Typical NP count for this variant.
    #[must_use]
    pub const fn np_count(&self) -> u32 {
        match self {
            Self::Akd1000 | Self::Akd1500 => 80,
            Self::Unknown(_) => 0,
        }
    }

    /// On-chip SRAM in megabytes.
    #[must_use]
    pub const fn sram_mb(&self) -> u32 {
        match self {
            Self::Akd1000 => 8,
            Self::Akd1500 => 10, // 100 KB/NPU × 32 NPUs + 1 MB dual-port
            Self::Unknown(_) => 0,
        }
    }

    /// PCIe lane count.
    #[must_use]
    pub const fn pcie_lanes(&self) -> u8 {
        match self {
            Self::Akd1000 => 1,
            Self::Akd1500 => 2,
            Self::Unknown(_) => 1,
        }
    }
}
