// SPDX-License-Identifier: AGPL-3.0-or-later

//! Backend abstraction for NPU drivers
//!
//! Provides unified interface for kernel and userspace backends.
//! Deep debt compliant: capability-based, runtime discovery, no hardcoding.

use crate::capabilities::Capabilities;
use crate::error::Result;
use std::fmt::Debug;

/// NPU backend trait - unified interface for kernel and userspace drivers
///
/// Deep Debt Principles:
/// - Runtime capability discovery (no hardcoded values)
/// - Agnostic design (works with any backend)
/// - Primal self-knowledge pattern
pub trait NpuBackend: Debug + Send + Sync {
    /// Initialize backend with runtime discovery
    ///
    /// No hardcoded device lists - discovers capabilities at runtime
    ///
    /// # Errors
    ///
    /// Returns error if device cannot be found or initialized.
    fn init(device_id: &str) -> Result<Self>
    where
        Self: Sized;

    /// Get runtime-discovered capabilities
    ///
    /// Returns actual hardware capabilities, not assumptions
    fn capabilities(&self) -> &Capabilities;

    /// Load model to NPU
    ///
    /// Backend chooses optimal transfer method (DMA or PIO)
    ///
    /// # Errors
    ///
    /// Returns error if model cannot be loaded (transfer failure, invalid format).
    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle>;

    /// Load reservoir weights for echo state networks
    ///
    /// `w_in`: input -> reservoir weights
    /// `w_res`: reservoir -> reservoir (recurrent) weights
    ///
    /// # Errors
    ///
    /// Returns error if weights cannot be loaded (size mismatch, transfer failure).
    fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()>;

    /// Run inference
    ///
    /// Backend manages synchronization (interrupts or polling)
    ///
    /// # Errors
    ///
    /// Returns error if inference fails (hardware error, timeout).
    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>>;

    /// Measure current power draw
    ///
    /// Returns actual measurement, not estimate
    ///
    /// # Errors
    ///
    /// Returns error if power measurement is unavailable.
    fn measure_power(&self) -> Result<f32>;

    /// Get backend type for debugging
    fn backend_type(&self) -> BackendType;

    /// Check if backend is ready
    fn is_ready(&self) -> bool;

    /// Verify the last model load by reading back from SRAM.
    ///
    /// Compares a sample of on-chip data against the original program bytes.
    /// Returns the number of bytes verified and whether they matched.
    ///
    /// Default: no-op (not all backends support SRAM readback).
    ///
    /// # Errors
    ///
    /// Concrete backends return errors only when SRAM readback fails; this default
    /// implementation always succeeds.
    fn verify_load(&mut self, _expected: &[u8]) -> Result<LoadVerification> {
        Ok(LoadVerification::unsupported())
    }

    /// Mutate weights directly in on-chip SRAM.
    ///
    /// Writes `data` at `offset` within the model's weight region,
    /// bypassing DMA for near-zero-latency small updates.
    /// This is the fast path for `set_variable()` — ESN readout swaps,
    /// online learning updates, etc.
    ///
    /// Default: falls back to `load_model` (full DMA re-upload).
    ///
    /// # Errors
    ///
    /// Returns error if SRAM is not accessible or offset is out of bounds.
    fn mutate_weights(&mut self, _offset: usize, _data: &[u8]) -> Result<()> {
        Err(crate::error::AkidaError::capability_query_failed(
            "Direct weight mutation not supported by this backend",
        ))
    }

    /// Read raw bytes from on-chip SRAM at a given offset.
    ///
    /// Used for diagnostics, verification, and state inspection.
    ///
    /// Default: not supported.
    ///
    /// # Errors
    ///
    /// Returns error if SRAM is not accessible.
    fn read_sram(&mut self, _offset: usize, _length: usize) -> Result<Vec<u8>> {
        Err(crate::error::AkidaError::capability_query_failed(
            "SRAM read not supported by this backend",
        ))
    }
}

/// Model handle returned after loading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelHandle(u32);

impl ModelHandle {
    /// Create new model handle
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get model ID
    pub const fn id(&self) -> u32 {
        self.0
    }
}

/// Result of model load verification via SRAM readback.
#[derive(Debug, Clone)]
pub struct LoadVerification {
    /// Whether the verification is supported by this backend.
    pub supported: bool,
    /// Number of bytes sampled for verification.
    pub bytes_checked: usize,
    /// Number of bytes that matched expected data.
    pub bytes_matched: usize,
    /// Whether the load is verified correct.
    pub verified: bool,
}

impl LoadVerification {
    /// Backend doesn't support SRAM readback.
    pub const fn unsupported() -> Self {
        Self {
            supported: false,
            bytes_checked: 0,
            bytes_matched: 0,
            verified: false,
        }
    }

    /// Create a successful verification result.
    pub const fn ok(bytes_checked: usize) -> Self {
        Self {
            supported: true,
            bytes_checked,
            bytes_matched: bytes_checked,
            verified: true,
        }
    }

    /// Create a failed verification result.
    pub const fn mismatch(bytes_checked: usize, bytes_matched: usize) -> Self {
        Self {
            supported: true,
            bytes_checked,
            bytes_matched,
            verified: false,
        }
    }
}

/// Backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Kernel driver (/dev/akida*)
    Kernel,

    /// Userspace driver (mmap `PCIe` BARs)
    Userspace,

    /// VFIO driver (pure Rust with DMA)
    Vfio,

    /// Software (virtual NPU) — f32 CPU simulation, no hardware required
    Software,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kernel => write!(f, "Kernel"),
            Self::Userspace => write!(f, "Userspace"),
            Self::Vfio => write!(f, "VFIO"),
            Self::Software => write!(f, "Software (VirtualNPU)"),
        }
    }
}

/// Backend selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// Automatically select best available
    Auto,

    /// Force kernel driver
    Kernel,

    /// Force userspace driver
    Userspace,

    /// Force VFIO driver (pure Rust with DMA)
    Vfio,

    /// Force software (virtual NPU) backend — for CI and cross-substrate comparison
    Software,
}

/// Select appropriate backend based on availability and requirements
///
/// Deep Debt: Runtime discovery, no assumptions about environment
///
/// # Errors
///
/// Returns error if no suitable backend can be initialized for the given device.
pub fn select_backend(selection: BackendSelection, device_id: &str) -> Result<Box<dyn NpuBackend>> {
    use crate::backends::kernel::KernelBackend;
    use crate::backends::software::SoftwareBackend;
    use crate::backends::userspace::UserspaceBackend;
    use crate::vfio::VfioBackend;

    match selection {
        BackendSelection::Auto => {
            // Try kernel first (best performance with C module)
            if let Ok(backend) = KernelBackend::init(device_id) {
                tracing::info!("Using kernel backend for {device_id}");
                return Ok(Box::new(backend));
            }

            // Try VFIO second (pure Rust with DMA)
            if let Ok(backend) = VfioBackend::init(device_id) {
                tracing::info!("Using VFIO backend for {device_id}");
                return Ok(Box::new(backend));
            }

            // Fall back to userspace (pure Rust, no DMA)
            tracing::info!("Kernel/VFIO unavailable, using userspace for {device_id}");
            UserspaceBackend::init(device_id).map(|b| Box::new(b) as Box<dyn NpuBackend>)
        }

        BackendSelection::Kernel => {
            KernelBackend::init(device_id).map(|b| Box::new(b) as Box<dyn NpuBackend>)
        }

        BackendSelection::Userspace => {
            UserspaceBackend::init(device_id).map(|b| Box::new(b) as Box<dyn NpuBackend>)
        }

        BackendSelection::Vfio => {
            VfioBackend::init(device_id).map(|b| Box::new(b) as Box<dyn NpuBackend>)
        }

        BackendSelection::Software => {
            SoftwareBackend::init(device_id).map(|b| Box::new(b) as Box<dyn NpuBackend>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_handle_roundtrip() {
        let h = ModelHandle::new(42);
        assert_eq!(h.id(), 42);
    }

    #[test]
    fn load_verification_constructors() {
        let u = LoadVerification::unsupported();
        assert!(!u.supported);
        let ok = LoadVerification::ok(100);
        assert!(ok.verified && ok.bytes_matched == 100);
        let bad = LoadVerification::mismatch(10, 3);
        assert!(!bad.verified && bad.bytes_matched == 3);
    }

    #[test]
    fn backend_type_display_and_selection_software() {
        assert!(BackendType::Software.to_string().contains("Software"));
        let b = select_backend(BackendSelection::Software, "0").expect("software backend");
        assert_eq!(b.backend_type(), BackendType::Software);
    }

    #[test]
    fn backend_type_display_covers_all_variants() {
        assert_eq!(BackendType::Kernel.to_string(), "Kernel");
        assert_eq!(BackendType::Userspace.to_string(), "Userspace");
        assert_eq!(BackendType::Vfio.to_string(), "VFIO");
    }

    #[test]
    fn software_backend_default_trait_methods() {
        let mut b = select_backend(BackendSelection::Software, "0").expect("software");
        let v = b.verify_load(&[1, 2, 3]).expect("default verify_load");
        assert!(!v.supported);
        assert!(b.mutate_weights(0, &[1]).is_err());
        assert!(b.read_sram(0, 4).is_err());
        let _ = v;
    }
}
