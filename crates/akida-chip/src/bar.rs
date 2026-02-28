//! PCIe BAR layout for AKD1000 / AKD1500.
//!
//! Measured via sysfs `/sys/bus/pci/devices/{addr}/resource` probing.
//! Source: BEYOND_SDK.md Discovery 8.
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
    pub const TYPICAL_ADDR: u64 = 0x84000000;
    /// Size in bytes.
    pub const SIZE: u64 = 16 * 1024 * 1024; // 16 MB
    /// BAR index for VFIO region queries.
    pub const VFIO_INDEX: u32 = 0;
    /// Properties.
    pub const IS_64BIT: bool = false;
    pub const IS_PREFETCHABLE: bool = false;
}

/// BAR1 — NP mesh / SRAM window (16 GB, 64-bit prefetchable).
///
/// **Discovery 8 (BEYOND_SDK.md):** The 16 GB address space is the full NP
/// mesh decode range, far larger than the 8 MB physical SRAM spec.
/// With 78 NPs, each could address ~200 MB.
pub mod bar1 {
    /// Typical physical address.
    pub const TYPICAL_ADDR: u64 = 0x4000_0000_00;
    /// Decode range in bytes.
    pub const DECODE_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GB
    /// Physical SRAM behind this BAR.
    pub const PHYSICAL_SRAM: u64 = 8 * 1024 * 1024; // 8 MB
    /// BAR index for VFIO region queries.
    pub const VFIO_INDEX: u32 = 1;
    pub const IS_64BIT: bool = true;
    pub const IS_PREFETCHABLE: bool = true;

    /// Per-NP address stride (hypothetical, not yet confirmed).
    /// If 78 NPs share 16 GB uniformly: 16 GB / 78 ≈ 210 MB each.
    pub const PER_NP_STRIDE: u64 = DECODE_SIZE / 78;
}

/// BAR3 — secondary memory (32 MB, 64-bit prefetchable).
pub mod bar3 {
    pub const TYPICAL_ADDR: u64 = 0x4400_0000_00;
    pub const SIZE: u64 = 32 * 1024 * 1024; // 32 MB
    pub const VFIO_INDEX: u32 = 3;
    pub const IS_64BIT: bool = true;
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
            Self::Control  => bar0::SIZE,
            Self::Mesh     => bar1::DECODE_SIZE,
            Self::Secondary => bar3::SIZE,
        }
    }
}
