// SPDX-License-Identifier: AGPL-3.0-or-later

//! `PCIe` BAR layout for AKD1000 / AKD1500.
//!
//! Measured via sysfs `/sys/bus/pci/devices/{addr}/resource` probing.
//! Source: `BEYOND_SDK.md` Discovery 8.
//!
//! ```text
//! BAR  Address          Size    Type                  Purpose
//! ──── ──────────────── ─────── ─────────────────────────────────────────────
//!  0   0x84000000       16 MB   32-bit non-prefetch   Register space (MMIO)
//!  1   0x4000000000     16 GB   64-bit prefetchable   NP mesh / SRAM window
//!  3   0x4400000000     32 MB   64-bit prefetchable   Secondary memory
//!  5   0x7000           128 B   I/O ports             Control ports
//!  6   0x85000000       512 KB  Expansion ROM         Firmware
//! ```
//!
//! BAR1 exposes the full NP mesh address decode range (16 GB) for 78 NPs.
//! With 78 NPs, each could have ~200 MB of addressable space.  The first 64 KB
//! reads as all-zeros, indicating sparse mapping — data appears at NP-specific
//! offsets after programming.

/// BAR0 — control register space (16 MB, MMIO).
pub mod bar0 {
    /// Typical physical address (may vary per system).
    pub const TYPICAL_ADDR: u64 = 0x8400_0000;
    /// Size in bytes.
    pub const SIZE: u64 = 16 * 1024 * 1024; // 16 MB
    /// BAR index for VFIO region queries.
    pub const VFIO_INDEX: u32 = 0;
    /// Whether the BAR is 64-bit decoded.
    pub const IS_64BIT: bool = false;
    /// Whether the region is prefetchable.
    pub const IS_PREFETCHABLE: bool = false;
}

/// BAR1 — NP mesh / SRAM window (16 GB, 64-bit prefetchable).
///
/// **Discovery 8 (`BEYOND_SDK.md)`:** The 16 GB address space is the full NP
/// mesh decode range, far larger than the 8 MB physical SRAM spec.
/// With 78 NPs, each could address ~200 MB.
pub mod bar1 {
    /// Typical physical address.
    pub const TYPICAL_ADDR: u64 = 0x0040_0000_0000;
    /// Decode range in bytes.
    pub const DECODE_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GB
    /// Physical SRAM behind this BAR.
    pub const PHYSICAL_SRAM: u64 = 8 * 1024 * 1024; // 8 MB
    /// BAR index for VFIO region queries.
    pub const VFIO_INDEX: u32 = 1;
    /// Whether the BAR is 64-bit decoded.
    pub const IS_64BIT: bool = true;
    /// Whether the region is prefetchable.
    pub const IS_PREFETCHABLE: bool = true;

    /// Per-NP address stride (hypothetical, not yet confirmed).
    /// If 78 NPs share 16 GB uniformly: 16 GB / 78 ≈ 210 MB each.
    pub const PER_NP_STRIDE: u64 = DECODE_SIZE / 78;
}

/// BAR3 — secondary memory (32 MB, 64-bit prefetchable).
pub mod bar3 {
    /// Typical physical address.
    pub const TYPICAL_ADDR: u64 = 0x0044_0000_0000;
    /// Region size in bytes.
    pub const SIZE: u64 = 32 * 1024 * 1024; // 32 MB
    /// BAR index for VFIO region queries.
    pub const VFIO_INDEX: u32 = 3;
    /// Whether the BAR is 64-bit decoded.
    pub const IS_64BIT: bool = true;
    /// Whether the region is prefetchable.
    pub const IS_PREFETCHABLE: bool = true;
}

/// BAR index enumeration for ergonomic VFIO region queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Bar {
    /// BAR0 — control registers.
    Control = 0,
    /// BAR1 — NP mesh / SRAM window.
    Mesh = 1,
    /// BAR3 — secondary memory.
    Secondary = 3,
}

impl Bar {
    /// Typical size of this BAR in bytes.
    #[must_use]
    pub const fn typical_size(&self) -> u64 {
        match self {
            Self::Control => bar0::SIZE,
            Self::Mesh => bar1::DECODE_SIZE,
            Self::Secondary => bar3::SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bar0;
    use super::bar1;
    use super::bar3;
    use super::*;

    #[test]
    fn bar0_mmio_size_and_vfio_index() {
        assert_eq!(bar0::VFIO_INDEX, 0);
        assert_eq!(bar0::SIZE, 16 * 1024 * 1024);
        assert!(!bar0::IS_64BIT);
        assert!(!bar0::IS_PREFETCHABLE);
    }

    #[test]
    fn bar1_mesh_decode_exceeds_physical_sram() {
        assert!(bar1::DECODE_SIZE > bar1::PHYSICAL_SRAM);
        assert!(bar1::IS_64BIT);
        assert!(bar1::IS_PREFETCHABLE);
        assert_eq!(bar1::PER_NP_STRIDE, bar1::DECODE_SIZE / 78);
    }

    #[test]
    fn bar3_secondary_size_matches_table() {
        assert_eq!(bar3::SIZE, 32 * 1024 * 1024);
        assert_eq!(bar3::VFIO_INDEX, 3);
    }

    #[test]
    fn bar_typical_size_matches_module_constants() {
        assert_eq!(Bar::Control.typical_size(), bar0::SIZE);
        assert_eq!(Bar::Mesh.typical_size(), bar1::DECODE_SIZE);
        assert_eq!(Bar::Secondary.typical_size(), bar3::SIZE);
    }

    #[test]
    fn bar_enum_discriminant_matches_pci_index() {
        assert_eq!(Bar::Control as u32, 0);
        assert_eq!(Bar::Mesh as u32, 1);
        assert_eq!(Bar::Secondary as u32, 3);
    }
}
