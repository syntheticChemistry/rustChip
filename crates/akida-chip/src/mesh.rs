//! NP mesh topology and routing model.
//!
//! Established by hardware probing and C++ engine symbol analysis.
//! Source: BEYOND_SDK.md Discoveries 2, 5, 9.
//!
//! ## Key findings
//!
//! - AKD1000 has **80 NPs** in a **5×8×2** mesh (78 functional in production)
//! - FC layers **merge into a single hardware pass** via SkipDMA routing
//!   (NP-to-NP transfer without PCIe round-trip)
//! - Multiple SRAM types: 64-bit filter SRAM, 51-bit threshold SRAM,
//!   32-bit event/status SRAM
//! - Three hardware variants in C++ engine: `akida::v1` (AKD1000),
//!   `akida::v2` (Akida 2.0), `akida::pico`

/// NP mesh topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshTopology {
    /// NPs in X dimension.
    pub x: u8,
    /// NPs in Y dimension.
    pub y: u8,
    /// NPs in Z dimension (pipeline depth).
    pub z: u8,
    /// Functional NPs (may be less than x×y×z due to disabled NPs).
    pub functional: u32,
}

impl MeshTopology {
    /// AKD1000 reference topology (confirmed by probing).
    pub const AKD1000: Self = Self { x: 5, y: 8, z: 2, functional: 78 };

    /// Total NP slots in the mesh.
    #[must_use]
    pub const fn total_slots(&self) -> u32 {
        (self.x as u32) * (self.y as u32) * (self.z as u32)
    }

    /// Disabled NP count.
    #[must_use]
    pub const fn disabled(&self) -> u32 {
        self.total_slots() - self.functional
    }
}

/// NP capabilities per node.
#[derive(Debug, Clone, Copy)]
pub struct NpCapabilities {
    /// NPU cores per NP node.
    pub npus_per_node: u32,
    /// MACs per NPU core.
    pub macs_per_npu: u32,
    /// Local SRAM per NPU in KB (configurable range).
    pub sram_per_npu_kb_min: u32,
    pub sram_per_npu_kb_max: u32,
    /// Weight precision bits (Akida 1.0).
    pub weight_bits: &'static [u8],
    /// Activation precision bits.
    pub activation_bits: &'static [u8],
}

/// AKD1000 NP capability profile (from HARDWARE.md).
pub const AKD1000_NP: NpCapabilities = NpCapabilities {
    npus_per_node:        4,
    macs_per_npu:         128,
    sram_per_npu_kb_min:  50,
    sram_per_npu_kb_max:  130,
    weight_bits:          &[1, 2, 4],
    activation_bits:      &[1, 2, 4],
};

/// Total MACs for AKD1000.
pub const AKD1000_TOTAL_MACS: u32 =
    MeshTopology::AKD1000.functional * AKD1000_NP.npus_per_node * AKD1000_NP.macs_per_npu;
    // = 78 × 4 × 128 = 39,936

/// SkipDMA — NP-to-NP data routing without PCIe round-trip.
///
/// Discovered by C++ engine symbol analysis (Discovery 2 and 9).
/// This mechanism is why deep FC chains execute as a single hardware pass.
pub mod skip_dma {
    /// Maximum FC depth that still merges into a single pass.
    /// Tested up to depth=8 (9 layers) with identical latency.
    pub const MAX_MERGED_DEPTH: usize = 8;

    /// Per-layer latency overhead when using SkipDMA (µs).
    /// Discovery 2: 8 layers costs only 3 µs more than 2 layers.
    pub const PER_LAYER_OVERHEAD_US: u64 = 0; // effectively zero
}

/// SRAM types discovered in C++ engine (Discovery 9).
pub mod sram_types {
    /// 64-bit filter SRAM for convolution kernels.
    pub const FSRAM_64B: &str = "get_fsram_64b_memory_size";
    /// 51-bit threshold SRAM — more precision than the "4-bit everything" spec.
    pub const TSRAM_51B: &str = "get_tsram_51b_memory_size";
    /// 32-bit event SRAM.
    pub const EVSRAM_32B: &str = "get_evsram_32b_memory_size";
    /// 32-bit status SRAM.
    pub const STSRAM_32B: &str = "get_stsram_32b_memory_size";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn akd1000_mesh_geometry() {
        let mesh = MeshTopology::AKD1000;
        assert_eq!(mesh.total_slots(), 80);
        assert_eq!(mesh.functional, 78);
        assert_eq!(mesh.disabled(), 2);
    }

    #[test]
    fn total_macs() {
        // 78 NPs × 4 NPUs × 128 MACs = 39,936
        assert_eq!(AKD1000_TOTAL_MACS, 39_936);
    }
}
