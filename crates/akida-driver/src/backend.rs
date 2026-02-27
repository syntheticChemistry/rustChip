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
    /// w_in: input -> reservoir weights
    /// w_res: reservoir -> reservoir (recurrent) weights
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

/// Backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Kernel driver (/dev/akida*)
    Kernel,

    /// Userspace driver (mmap PCIe BARs)
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
