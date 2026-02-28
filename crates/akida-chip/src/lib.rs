//! Silicon model for BrainChip Akida AKD1000 / AKD1500.
//!
//! This crate has **no dependencies** and **no hardware access** — it is a
//! pure model of the silicon: register addresses, BAR layout, NP mesh
//! topology, PCIe identifiers, and the FlatBuffer program format.
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
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod bar;
pub mod mesh;
pub mod pcie;
pub mod program;
pub mod regs;
