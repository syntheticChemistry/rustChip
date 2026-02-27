//! Pure Rust driver for BrainChip Akida neuromorphic processors.
//!
//! This crate provides the full software stack for AKD1000 / AKD1500 access.
//! No Python. No C++ SDK. No vendor MetaTF.
//!
//! # Backend hierarchy
//!
//! ```text
//! Primary (no kernel module required):
//!   VfioBackend  — VFIO/IOMMU + full DMA (preferred for production)
//!
//! Fallback (when C akida_pcie module is loaded):
//!   KernelBackend — /dev/akida* read/write
//!
//! Development:
//!   UserspaceBackend — BAR mmap, no DMA
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use akida_driver::DeviceManager;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mgr  = DeviceManager::discover()?;
//! let caps = mgr.devices()[0].capabilities();
//!
//! println!("{:?} — {} NPs, {} MB SRAM, PCIe Gen{} x{}",
//!          caps.chip_version, caps.npu_count, caps.memory_mb,
//!          caps.pcie.generation, caps.pcie.lanes);
//!
//! let model_bytes = std::fs::read("model.fbz")?;
//! let mut dev = mgr.open_first()?;
//! dev.write(&model_bytes)?;
//! let mut out = vec![0u8; 1024];
//! dev.read(&mut out)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Measured results (AKD1000, PCIe x1 Gen2, Feb 2026)
//!
//! | Metric | Value |
//! |--------|-------|
//! | DMA throughput (sustained) | 37 MB/s |
//! | Single inference | 54 µs / 18,500 Hz |
//! | Batch=8 | 390 µs/sample / 20,700 /s |
//! | Energy per inference | 1.4 µJ |
//! | 24-hour production calls (Exp 022) | 5,978 |

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]

mod backend;
pub mod backends;
mod capabilities;
mod device;
mod discovery;
mod error;
pub mod hybrid;
mod inference;
mod io;
mod loading;
pub mod mmio;
pub mod setup;
pub mod vfio;

/// Hardware identification constants (re-exported from akida-chip).
pub mod pcie_ids {
    pub use akida_chip::pcie::{
        lspci_filter, ChipVariant, ALL_DEVICE_IDS, BRAINCHIP_VENDOR_ID,
        MEASURED_DMA_THROUGHPUT_MB_S, OPTIMAL_BATCH_SIZE, PCIE_GEN2_X1_ROUNDTRIP_US,
    };
    pub use akida_chip::pcie::device_id;
}

pub use backend::{select_backend, BackendSelection, BackendType, ModelHandle, NpuBackend};
pub use backends::software::{pack_software_model, SoftwareBackend};
pub use backends::UserspaceBackend;
pub use capabilities::{
    BatchCapabilities, Capabilities, ChipVersion, ClockMode, MeshTopology, PcieConfig,
    WeightMutationSupport,
};
pub use device::{AkidaDevice, DeviceHandle};
pub use discovery::{DeviceInfo, DeviceManager};
pub use error::{AkidaError, Result};
pub use hybrid::{EsnSubstrate, EsnWeights, HybridEsn, SubstrateInfo, SubstrateMode, SubstrateSelector};
pub use inference::{InferenceConfig, InferenceExecutor, InferenceResult};
pub use loading::{LoadConfig, LoadMetrics, ModelLoader, ModelProgram, NpuConfig};
pub use vfio::VfioBackend;

/// Commonly used types.
pub mod prelude {
    pub use crate::{
        AkidaDevice, AkidaError, Capabilities, DeviceManager, EsnSubstrate, EsnWeights,
        HybridEsn, InferenceConfig, InferenceExecutor, InferenceResult, LoadConfig, ModelLoader,
        ModelProgram, NpuConfig, Result, SubstrateMode, SubstrateSelector, VfioBackend,
    };
}
