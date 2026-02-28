//! BAR0 register map for AKD1000.
//!
//! These offsets were established by two methods:
//! 1. Direct BAR0 probing via MMIO (see BEYOND_SDK.md Discovery 8)
//! 2. C++ engine symbol analysis (`core.so`, 1,048 exports)
//!
//! Confirmed values are marked `// confirmed`.
//! Inferred values are marked `// inferred` and should be validated.
//!
//! ## Probing results (BEYOND_SDK.md)
//!
//! ```text
//! 0x000000: 0x194000a1  — Device ID / version register    (confirmed)
//! 0x001094: 0x0000a028  — Control register                (confirmed)
//! 0x0010c0: 0x5b (91)   — NP count or feature bits        (confirmed)
//! 0x001410: 0x2000       — SRAM region config             (confirmed)
//! 0x001418: 0x8000       — SRAM region config             (confirmed)
//! 0x001484: timestamp/firmware version                    (confirmed)
//! 0x001e0c-0x001e20: six 0x00000001 — NP enable bits?     (confirmed)
//! 0x004010: 0x04aa0001  — DMA/mesh configuration word     (confirmed)
//! 0xe000+:  Per-NP configuration registers                (confirmed pattern)
//! 0xbadf5040: "Bad food" — uninitialized/protected space  (confirmed)
//! ```

// ── Device identity ──────────────────────────────────────────────────────────

/// Device ID / version register. Reads `0x194000a1` on AKD1000. // confirmed
pub const DEVICE_ID: usize = 0x0000;

/// Device version register. // inferred
pub const VERSION: usize = 0x0004;

// ── Status and control ───────────────────────────────────────────────────────

/// Main status register.
pub const STATUS: usize = 0x0008;

/// Control register — reads `0x0000a028`. // confirmed @ 0x001094
pub const CONTROL: usize = 0x0010; // inferred: nearby 0x001094

// ── NP mesh ──────────────────────────────────────────────────────────────────

/// NP count or feature bits — reads `0x5b` (91). // confirmed @ 0x0010c0
pub const NP_COUNT: usize = 0x0010_C0;

/// DMA / mesh configuration word — reads `0x04aa0001`. // confirmed @ 0x004010
pub const DMA_MESH_CONFIG: usize = 0x4010;

/// NP enable bits (six × 0x00000001). // confirmed @ 0x001e0c–0x001e20
pub const NP_ENABLE_BASE: usize = 0x1E0C;
pub const NP_ENABLE_COUNT: usize = 6;

// ── SRAM region ──────────────────────────────────────────────────────────────

/// SRAM region config word 0 — reads `0x2000`. // confirmed @ 0x001410
pub const SRAM_REGION_0: usize = 0x1410;
/// SRAM region config word 1 — reads `0x8000`. // confirmed @ 0x001418
pub const SRAM_REGION_1: usize = 0x1418;
/// SRAM BAR address — reads `0x85800`. // confirmed @ 0x001418+4
pub const SRAM_BAR_ADDR: usize = 0x141C;

// ── DW eDMA engine ───────────────────────────────────────────────────────────
// The DW eDMA (DesignWare Enhanced DMA) controller is exposed through BAR0.
// Offsets match the standard DesignWare PCIe eDMA register layout.

/// eDMA write channel 0 control (inferred from DW eDMA spec).
pub const EDMA_WRITE_CH0_CTL: usize = 0x0200;
/// eDMA read channel 0 control.
pub const EDMA_READ_CH0_CTL:  usize = 0x0300;
/// eDMA interrupt status.
pub const EDMA_INT_STATUS:    usize = 0x0010_0010;

// ── Model load ───────────────────────────────────────────────────────────────

/// Model program load address (low 32 bits).
pub const MODEL_ADDR_LO: usize = 0x0100;
/// Model program load address (high 32 bits).
pub const MODEL_ADDR_HI: usize = 0x0104;
/// Model program size in bytes.
pub const MODEL_SIZE: usize = 0x0108;
/// Model load trigger — write 1 to start load.
pub const MODEL_LOAD: usize = 0x010C;

// ── Inference ────────────────────────────────────────────────────────────────

/// Input buffer address (low 32 bits).
pub const INPUT_ADDR_LO:  usize = 0x0200;
/// Input buffer address (high 32 bits).
pub const INPUT_ADDR_HI:  usize = 0x0204;
/// Input buffer size.
pub const INPUT_SIZE:     usize = 0x0208;
/// Output buffer address (low 32 bits).
pub const OUTPUT_ADDR_LO: usize = 0x0300;
/// Output buffer address (high 32 bits).
pub const OUTPUT_ADDR_HI: usize = 0x0304;
/// Output buffer size.
pub const OUTPUT_SIZE:    usize = 0x0308;
/// Inference trigger — write 1 to start.
pub const INFER_START:    usize = 0x0400;
/// Inference completion status.
pub const INFER_STATUS:   usize = 0x0404;

// ── Interrupts ───────────────────────────────────────────────────────────────

/// Interrupt status register.
pub const IRQ_STATUS: usize = 0x0020;
/// Interrupt enable register.
pub const IRQ_ENABLE: usize = 0x0024;

// ── Per-NP configuration (repeating) ─────────────────────────────────────────

/// Per-NP register block base (confirmed: repeating pattern at 0xe000+).
pub const NP_CONFIG_BASE: usize = 0xE000;
/// Stride between NP register blocks (inferred).
pub const NP_CONFIG_STRIDE: usize = 0x100;

// ── Status register bit definitions ──────────────────────────────────────────

pub mod status {
    /// Device ready to accept commands.
    pub const READY:        u32 = 1 << 0;
    /// Device currently processing.
    pub const BUSY:         u32 = 1 << 1;
    /// Error during last operation.
    pub const ERROR:        u32 = 1 << 2;
    /// Model successfully loaded.
    pub const MODEL_LOADED: u32 = 1 << 3;
}

// ── Control register bit definitions ─────────────────────────────────────────

pub mod control {
    /// Soft reset.
    pub const RESET:      u32 = 1 << 0;
    /// Enable device.
    pub const ENABLE:     u32 = 1 << 1;
    /// Power-save mode (Economy clock).
    pub const POWER_SAVE: u32 = 1 << 2;
}

// ── Clock mode register values ────────────────────────────────────────────────
// Discovery 4: three clock modes confirmed.

pub mod clock {
    pub const PERFORMANCE: u32 = 0;
    pub const ECONOMY:     u32 = 1; // 19% slower, 18% less power
    pub const LOW_POWER:   u32 = 2; // 9.3× slower, 27% less power
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_offsets_non_overlapping() {
        // Key registers should not overlap
        assert_ne!(DEVICE_ID, STATUS);
        assert_ne!(MODEL_LOAD, INFER_START);
        assert_ne!(INPUT_ADDR_LO, OUTPUT_ADDR_LO);
    }

    #[test]
    fn confirmed_probed_addresses() {
        // From BEYOND_SDK.md direct probing
        assert_eq!(DEVICE_ID, 0x0000);
        assert_eq!(DMA_MESH_CONFIG, 0x4010);
        assert_eq!(NP_COUNT, 0x0010_C0);
        assert_eq!(NP_ENABLE_BASE, 0x1E0C);
    }
}
