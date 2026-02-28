//! Kernel backend for NPU
//!
//! Wraps existing `AkidaDevice` to conform to `NpuBackend` trait.
//! Deep Debt: Uses existing production code, no duplication.

use super::read_hwmon_power;
use crate::backend::{BackendType, ModelHandle, NpuBackend};
use crate::capabilities::Capabilities;
use crate::device::AkidaDevice;
use crate::discovery::{DeviceInfo, DeviceManager};
use crate::error::Result;

/// Kernel driver backend
///
/// Uses /dev/akida* device nodes via kernel module
#[derive(Debug)]
pub struct KernelBackend {
    device: AkidaDevice,
    device_info: DeviceInfo,
}

impl NpuBackend for KernelBackend {
    fn init(device_path: &str) -> Result<Self> {
        tracing::info!("Initializing kernel backend for {device_path}");

        // Runtime discovery (no hardcoding!)
        let manager = DeviceManager::discover()?;

        // Find device by path or index
        let device_info = if device_path.starts_with("/dev/akida") {
            // Path-based lookup
            manager
                .devices()
                .iter()
                .find(|d| d.path().to_str() == Some(device_path))
                .ok_or_else(|| {
                    crate::error::AkidaError::capability_query_failed(format!(
                        "Device not found: {device_path}"
                    ))
                })?
                .clone()
        } else if let Ok(index) = device_path.parse::<usize>() {
            // Index-based lookup
            manager.device(index)?.clone()
        } else {
            // PCIe address lookup
            manager
                .devices()
                .iter()
                .find(|d| d.pcie_address() == device_path)
                .ok_or_else(|| {
                    crate::error::AkidaError::capability_query_failed(format!(
                        "Device not found: {device_path}"
                    ))
                })?
                .clone()
        };

        // Open device
        let device = AkidaDevice::open(&device_info)?;

        tracing::info!(
            "Kernel backend initialized: {} ({} NPUs, {} MB)",
            device_info.pcie_address(),
            device_info.capabilities().npu_count,
            device_info.capabilities().memory_mb
        );

        Ok(Self {
            device,
            device_info,
        })
    }

    fn capabilities(&self) -> &Capabilities {
        self.device_info.capabilities()
    }

    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle> {
        // Use existing kernel driver implementation (DMA)
        self.device.write(model)?;
        Ok(ModelHandle::new(0))
    }

    fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()> {
        tracing::info!(
            "Loading reservoir via kernel driver: w_in={}, w_res={}",
            w_in.len(),
            w_res.len()
        );

        // Safe byte conversion using bytemuck-style cast (zero-copy, no unsafe)
        let w_in_bytes = bytemuck::cast_slice::<f32, u8>(w_in);
        let w_res_bytes = bytemuck::cast_slice::<f32, u8>(w_res);

        // Write via DMA (fast!)
        self.device.write(w_in_bytes)?;
        self.device.write(w_res_bytes)?;

        tracing::info!("Reservoir loaded via DMA");
        Ok(())
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Buffer sized for typical AKD1000 max (1024 floats), matching Vfio backend.
        // device.read() returns bytes actually read; we use that to size the output.
        const MAX_INFER_OUTPUT_BYTES: usize = 4096;

        let input_bytes = bytemuck::cast_slice::<f32, u8>(input);

        self.device.write(input_bytes)?;

        let mut output_bytes = vec![0u8; MAX_INFER_OUTPUT_BYTES];
        let bytes_read = self.device.read(&mut output_bytes)?;

        // Use only bytes actually read; truncate to whole floats
        let float_bytes = (bytes_read / std::mem::size_of::<f32>()) * std::mem::size_of::<f32>();
        let output: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&output_bytes[..float_bytes]).to_vec();

        Ok(output)
    }

    fn measure_power(&self) -> Result<f32> {
        if let Some(watts) = read_hwmon_power(self.device_info.pcie_address()) {
            return Ok(watts);
        }

        tracing::warn!(
            "NPU power unavailable for {}, using typical AKD1000 value",
            self.device_info.pcie_address()
        );
        Ok(1.5)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Kernel
    }

    fn is_ready(&self) -> bool {
        true // Kernel driver manages readiness
    }
}
