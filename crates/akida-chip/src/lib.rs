// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon model for `BrainChip` Akida AKD1000 / AKD1500.
//!
//! This crate has **no dependencies** and **no hardware access** — it is a
//! pure model of the silicon: register addresses, BAR layout, NP mesh
//! topology, `PCIe` identifiers, and the `FlatBuffer` program format.
//!
//! Everything here was established by direct hardware probing; see
//! `docs/BEYOND_SDK.md` for methodology and raw measurements.
//!
//! # Crate organisation
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`pcie`] | Vendor/device IDs, PCIe Gen2 x1 timing constants |
//! | [`bar`] | BAR layout (BAR0 16 MB control, BAR1 16 GB mesh, BAR3 32 MB) |
//! | [`regs`] | BAR0 register map — all offsets and bit definitions |
//! | [`mesh`] | NP mesh topology (5×8×2, 78 functional, SkipDMA routing) |
//! | [`program`] | FlatBuffer `program_info` / `program_data` format |

#![forbid(unsafe_code)]
#![warn(clippy::expect_used, clippy::unwrap_used)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod bar;
pub mod mesh;
pub mod pcie;
pub mod program;
pub mod regs;
pub mod sram;

#[cfg(test)]
mod tests {
    /// Smoke test: public modules resolve and compile together.
    #[test]
    fn crate_public_surface_is_reachable() {
        let _ = crate::pcie::BRAINCHIP_VENDOR_ID;
        let _ = crate::bar::Bar::Mesh.typical_size();
        let _ = crate::regs::DEVICE_ID;
        let _ = crate::mesh::MeshTopology::AKD1000.total_slots();
        let _ = crate::program::typical_sizes::PROGRAM_INFO_BYTES;
        let _ = crate::sram::Bar1Layout::akd1000().np_count;
    }
}
