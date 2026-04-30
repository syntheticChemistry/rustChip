// SPDX-License-Identifier: AGPL-3.0-or-later

//! VFIO device lifecycle — absorbed from coralReef's `coral-ember`/`coral-glowplug`.
//!
//! This module is a standalone, NPU-focused subset of coralReef's VFIO
//! passthrough infrastructure. It provides everything rustChip needs to
//! manage PCIe device driver transitions without depending on coralReef.
//!
//! For the full-scale GPU + NPU orchestrator with SCM_RIGHTS fd passing,
//! ring persistence, multi-vendor lifecycle, and systemd integration, see:
//! `primals/coralReef/crates/coral-ember/` and `coral-glowplug/`.
//!
//! # Architecture
//!
//! ```text
//! glowplug (this module)
//! ├── sysfs       — safe sysfs reads/writes with D-state protection
//! ├── lifecycle   — vendor-specific device lifecycle hooks (NPU-focused)
//! ├── swap        — driver bind/unbind orchestration
//! └── sovereign   — warm boot: firmware init via kernel driver, then VFIO takeover
//! ```
//!
//! # Why "glowplug"?
//!
//! A glow plug pre-heats a diesel engine so it can cold-start. This module
//! pre-heats NPU firmware so VFIO can take over a live device.

pub mod lifecycle;
pub mod sovereign;
pub mod swap;
pub mod sysfs;

pub use lifecycle::{BrainChipLifecycle, GenericNpuLifecycle, NpuLifecycle};
pub use sovereign::{BootResult, BootStep, StepStatus, sovereign_boot};
pub use swap::{SwapOutcome, swap_to_driver};
