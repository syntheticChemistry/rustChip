// SPDX-License-Identifier: AGPL-3.0-or-later

//! SRAM address model for AKD1000 / AKD1500.
//!
//! The Akida chip has multiple SRAM types per NP, accessible through
//! BAR0 registers and the BAR1 NP mesh window:
//!
//! | SRAM type       | Width | Purpose                          |
//! |-----------------|-------|----------------------------------|
//! | Filter SRAM     | 64-b  | Convolution kernels / weights    |
//! | Threshold SRAM  | 51-b  | Activation thresholds / biases   |
//! | Event SRAM      | 32-b  | Spike events / activations       |
//! | Status SRAM     | 32-b  | Layer status / control           |
//!
//! These were discovered via C++ engine symbol analysis (`core.so`,
//! 1,048 exports) — see `mesh::sram_types` for the accessor names.
//!
//! ## BAR1 SRAM window
//!
//! BAR1 decodes 16 GB for 78 NPs. Physical SRAM is ~8 MB total.
//! The decode range is sparse — most addresses return zeros.
//! Accessible regions correspond to programmed NP weight/activation storage.
//!
//! ## BAR0 SRAM control registers
//!
//! SRAM region configuration at offsets 0x1410–0x141C controls the
//! BAR1 → physical SRAM mapping.

use crate::bar;
use crate::mesh::MeshTopology;

/// SRAM region descriptor — a contiguous, addressable memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SramRegion {
    /// Offset from the start of the BAR.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
    /// Which SRAM type this corresponds to.
    pub kind: SramKind,
    /// NP index that owns this region (if per-NP).
    pub np_index: Option<u32>,
}

/// SRAM type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SramKind {
    /// 64-bit filter SRAM — convolution kernels / FC weights.
    Filter,
    /// 51-bit threshold SRAM — activation thresholds, more precision than spec.
    Threshold,
    /// 32-bit event SRAM — spike events and activations.
    Event,
    /// 32-bit status SRAM — layer status and control.
    Status,
    /// Unknown / unmapped region.
    Unknown,
}

impl core::fmt::Display for SramKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Filter => write!(f, "Filter(64b)"),
            Self::Threshold => write!(f, "Threshold(51b)"),
            Self::Event => write!(f, "Event(32b)"),
            Self::Status => write!(f, "Status(32b)"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// BAR1 SRAM layout model for a given mesh topology.
///
/// Computes per-NP address windows within the BAR1 decode range.
/// The actual mapping is hardware-dependent and requires probing
/// to confirm which offsets are readable.
#[derive(Debug, Clone)]
pub struct Bar1Layout {
    /// Total decode range of BAR1.
    pub decode_size: u64,
    /// Number of functional NPs.
    pub np_count: u32,
    /// Stride between NP address windows.
    pub np_stride: u64,
    /// Per-NP SRAM size estimate (physical).
    pub per_np_sram_bytes: u64,
}

impl Bar1Layout {
    /// Compute BAR1 layout from mesh topology.
    #[must_use]
    pub fn from_topology(mesh: &MeshTopology) -> Self {
        let np_count = mesh.functional;
        let decode_size = bar::bar1::DECODE_SIZE;
        let np_stride = if np_count > 0 {
            decode_size / u64::from(np_count)
        } else {
            decode_size
        };
        // Physical SRAM per NP: 8 MB total / 78 NPs ≈ 105 KB
        let per_np_sram_bytes = if np_count > 0 {
            bar::bar1::PHYSICAL_SRAM / u64::from(np_count)
        } else {
            0
        };

        Self {
            decode_size,
            np_count,
            np_stride,
            per_np_sram_bytes,
        }
    }

    /// AKD1000 default layout.
    #[must_use]
    pub fn akd1000() -> Self {
        Self::from_topology(&MeshTopology::AKD1000)
    }

    /// Construct from runtime-discovered NP count.
    ///
    /// Use this when you have the actual NP count from BAR0 but don't
    /// have the full `MeshTopology` structure.
    #[must_use]
    pub fn from_np_count(np_count: u32) -> Self {
        let decode_size = bar::bar1::DECODE_SIZE;
        let np_stride = if np_count > 0 {
            decode_size / u64::from(np_count)
        } else {
            decode_size
        };
        let per_np_sram_bytes = if np_count > 0 {
            bar::bar1::PHYSICAL_SRAM / u64::from(np_count)
        } else {
            0
        };

        Self {
            decode_size,
            np_count,
            np_stride,
            per_np_sram_bytes,
        }
    }

    /// Construct with a custom physical SRAM size (discovered from registers).
    #[must_use]
    pub fn from_discovered(np_count: u32, physical_sram: u64) -> Self {
        let decode_size = bar::bar1::DECODE_SIZE;
        let np_stride = if np_count > 0 {
            decode_size / u64::from(np_count)
        } else {
            decode_size
        };
        let per_np_sram_bytes = if np_count > 0 {
            physical_sram / u64::from(np_count)
        } else {
            0
        };

        Self {
            decode_size,
            np_count,
            np_stride,
            per_np_sram_bytes,
        }
    }

    /// Get the base offset for a specific NP within BAR1.
    #[must_use]
    pub fn np_base_offset(&self, np_index: u32) -> Option<u64> {
        if np_index >= self.np_count {
            return None;
        }
        Some(u64::from(np_index) * self.np_stride)
    }

    /// Generate probe offsets for systematic SRAM discovery.
    ///
    /// Returns a list of (offset, description) pairs to read during probing.
    /// The probe strategy:
    /// 1. First 64 KB (known to be zeros — confirms sparse mapping)
    /// 2. Start of each NP's address window
    /// 3. Power-of-two boundaries within each NP window
    #[must_use]
    pub fn probe_offsets(&self, max_nps: u32) -> Vec<ProbePoint> {
        let mut points = Vec::new();

        // Phase 1: Global probes — first 64 KB at 4 KB intervals
        for k in 0..16 {
            let offset = k * 4096;
            points.push(ProbePoint {
                offset,
                description: format!("global+{offset:#x}"),
            });
        }

        // Phase 2: Per-NP base probes
        let np_limit = max_nps.min(self.np_count);
        for np in 0..np_limit {
            let base = u64::from(np) * self.np_stride;

            // Probe the first few pages of each NP's window
            for page in 0..8u64 {
                let offset = base + page * 4096;
                if offset < self.decode_size {
                    points.push(ProbePoint {
                        offset,
                        description: format!("NP{np}+{:#x}", page * 4096),
                    });
                }
            }

            // Also probe at per-NP SRAM boundary estimates
            for &sram_offset in &[0u64, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x18000] {
                let offset = base + sram_offset;
                if offset < self.decode_size && !points.iter().any(|p| p.offset == offset) {
                    points.push(ProbePoint {
                        offset,
                        description: format!("NP{np}+{sram_offset:#x}"),
                    });
                }
            }
        }

        points
    }
}

/// A point to probe during SRAM discovery.
#[derive(Debug, Clone)]
pub struct ProbePoint {
    /// Offset within BAR1.
    pub offset: u64,
    /// Human-readable description.
    pub description: String,
}

/// SRAM probe result — what we found at a given offset.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Offset that was probed.
    pub offset: u64,
    /// Description of the probe point.
    pub description: String,
    /// Whether the read succeeded.
    pub readable: bool,
    /// The 32-bit value read (if readable).
    pub value: Option<u32>,
    /// Whether the region appears to contain data (non-zero).
    pub has_data: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn akd1000_layout() {
        let layout = Bar1Layout::akd1000();
        assert_eq!(layout.np_count, 78);
        assert_eq!(layout.decode_size, 16 * 1024 * 1024 * 1024);
        assert!(layout.np_stride > 200 * 1024 * 1024);
        assert!(layout.per_np_sram_bytes > 100 * 1024);
    }

    #[test]
    fn np_base_offsets() {
        let layout = Bar1Layout::akd1000();
        assert_eq!(layout.np_base_offset(0), Some(0));
        assert_eq!(layout.np_base_offset(1), Some(layout.np_stride));
        assert_eq!(layout.np_base_offset(78), None);
    }

    #[test]
    fn probe_points_generated() {
        let layout = Bar1Layout::akd1000();
        let points = layout.probe_offsets(2);
        assert!(!points.is_empty());
        assert!(points[0].offset == 0);
    }
}
