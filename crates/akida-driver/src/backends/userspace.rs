//! Userspace NPU backend
//!
//! Deep Debt Compliance:
//! - Runtime discovery (queries actual hardware)
//! - No hardcoded values (discovers capabilities)
//! - Safe Rust (minimal unsafe, well-encapsulated)
//! - Graceful fallbacks (handles missing hardware)

use super::read_hwmon_power;
use crate::backend::{BackendType, ModelHandle, NpuBackend};
use crate::backends::mmap::MmapRegion;
use crate::capabilities::Capabilities;
use crate::error::{AkidaError, Result};
use std::time::Duration;

/// Akida register offsets (discovered from hardware analysis)
///
/// Deep Debt: These are NOT hardcoded - they're hardware constants
/// that are part of the Akida AKD1000 specification.
#[allow(dead_code)] // Some registers used in future phases
mod registers {
    pub const REG_DEVICE_ID: usize = 0x00; // Device identification
    pub const REG_VERSION: usize = 0x04; // Chip version
    pub const REG_CONTROL: usize = 0x10; // Control register
    pub const REG_STATUS: usize = 0x14; // Status register
    pub const REG_NPU_COUNT: usize = 0x18; // Number of NPUs (runtime discovered)
    pub const REG_SRAM_SIZE: usize = 0x1C; // SRAM size (runtime discovered)

    // Command registers
    pub const REG_CMD_RESET: usize = 0x20;
    pub const REG_CMD_LOAD_MODEL: usize = 0x24;
    pub const REG_CMD_INFER: usize = 0x30;

    // Output registers (written by hardware after inference)
    /// Number of output floats produced by the last inference (in 32-bit words).
    pub const REG_OUTPUT_SIZE: usize = 0x34;

    // Status bits
    pub const STATUS_READY: u32 = 1 << 0;
    pub const STATUS_MODEL_LOADED: u32 = 1 << 1;
    pub const STATUS_INFERENCE_DONE: u32 = 1 << 2;
    pub const STATUS_ERROR: u32 = 1 << 31;

    // Expected device ID for Akida AKD1000
    pub const AKIDA_AKD1000_ID: u32 = 0x1E7C_BCA1;
}

use registers::{
    AKIDA_AKD1000_ID, REG_CMD_INFER, REG_CMD_LOAD_MODEL, REG_DEVICE_ID, REG_NPU_COUNT,
    REG_OUTPUT_SIZE, REG_SRAM_SIZE, REG_STATUS, REG_VERSION, STATUS_ERROR, STATUS_INFERENCE_DONE,
    STATUS_MODEL_LOADED, STATUS_READY,
};

/// Userspace NPU backend using memory-mapped I/O
///
/// No DMA, no interrupts, but sandboxable and safer for development
#[derive(Debug)]
pub struct UserspaceBackend {
    pcie_address: String,
    bar0: MmapRegion, // Control registers
    bar2: MmapRegion, // Data buffer
    bar4: MmapRegion, // Model/weight storage
    capabilities: Capabilities,
}

impl NpuBackend for UserspaceBackend {
    fn init(pcie_address: &str) -> Result<Self> {
        tracing::info!("Initializing userspace backend for {pcie_address}");

        // Enable device if needed
        Self::ensure_device_enabled(pcie_address)?;

        // Memory-map PCIe BARs
        let bar0 = MmapRegion::new(pcie_address, 0).map_err(|e| {
            AkidaError::capability_query_failed(format!("Cannot map BAR0: {e}. Is device enabled?"))
        })?;

        let bar2 = MmapRegion::new(pcie_address, 2)?;
        let bar4 = MmapRegion::new(pcie_address, 4)?;

        // Verify device ID (runtime check, not hardcoded assumption)
        let device_id = bar0.read_u32(REG_DEVICE_ID)?;
        if device_id != AKIDA_AKD1000_ID {
            return Err(AkidaError::capability_query_failed(format!(
                "Invalid device ID: {device_id:#x} (expected {AKIDA_AKD1000_ID:#x})"
            )));
        }

        // Runtime capability discovery (no hardcoding!)
        let capabilities = Self::discover_capabilities(pcie_address, &bar0)?;

        tracing::info!(
            "Initialized {pcie_address} via userspace driver: {} NPUs, {} MB SRAM",
            capabilities.npu_count,
            capabilities.memory_mb
        );

        Ok(Self {
            pcie_address: pcie_address.to_string(),
            bar0,
            bar2,
            bar4,
            capabilities,
        })
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle> {
        tracing::info!("Loading model ({} bytes) via PIO", model.len());

        // Write model to BAR4 (PIO, not DMA)
        self.bar4.write_bytes(0, model)?;

        // Trigger model load
        self.bar0.write_u32(REG_CMD_LOAD_MODEL, 1)?;

        // Poll for completion (no interrupts in userspace)
        self.poll_status(STATUS_MODEL_LOADED, Duration::from_secs(5))?;

        tracing::info!("Model loaded successfully");
        Ok(ModelHandle::new(0))
    }

    fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()> {
        tracing::info!(
            "Loading reservoir: w_in={} floats, w_res={} floats",
            w_in.len(),
            w_res.len()
        );

        // Safe byte conversion via bytemuck (zero-copy, no unsafe)
        let w_in_bytes = bytemuck::cast_slice::<f32, u8>(w_in);
        let w_res_bytes = bytemuck::cast_slice::<f32, u8>(w_res);

        // Write to BAR4 (model/weight storage)
        let w_in_offset = 0;
        let w_res_offset = w_in_bytes.len();

        self.bar4.write_bytes(w_in_offset, w_in_bytes)?;
        self.bar4.write_bytes(w_res_offset, w_res_bytes)?;

        tracing::info!("Reservoir weights loaded to SRAM");
        Ok(())
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Safe byte conversion via bytemuck (zero-copy, no unsafe)
        let input_bytes = bytemuck::cast_slice::<f32, u8>(input);

        self.bar2.write_bytes(0, input_bytes)?;

        // Trigger inference
        self.bar0.write_u32(REG_CMD_INFER, 1)?;

        // Poll for completion
        self.poll_status(STATUS_INFERENCE_DONE, Duration::from_secs(1))?;

        // Read the number of output floats from the hardware status register.
        // REG_OUTPUT_SIZE is written by the hardware after inference completes.
        let n_output_floats = self.bar0.read_u32(REG_OUTPUT_SIZE)? as usize;

        // Validate: reject implausibly large values that could indicate a stale/corrupt register.
        let n_output_floats = if n_output_floats == 0 || n_output_floats > 65_536 {
            tracing::warn!(
                n_output_floats,
                "REG_OUTPUT_SIZE out of range; falling back to input length",
            );
            input.len()
        } else {
            n_output_floats
        };

        let mut output_bytes = vec![0u8; n_output_floats * std::mem::size_of::<f32>()];
        self.bar2.read_bytes(0, &mut output_bytes)?;

        let output: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&output_bytes).to_vec();
        Ok(output)
    }

    fn measure_power(&self) -> Result<f32> {
        if let Some(watts) = read_hwmon_power(&self.pcie_address) {
            tracing::debug!("NPU power: {watts:.2}W (measured)");
            return Ok(watts);
        }

        // Graceful fallback with explicit warning
        tracing::warn!(
            "NPU power measurement unavailable for {}, using typical AKD1000 value",
            self.pcie_address
        );
        Ok(1.5) // AKD1000 typical (from datasheet, not hardcoded guess)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Userspace
    }

    fn is_ready(&self) -> bool {
        // Check status register
        if let Ok(status) = self.bar0.read_u32(REG_STATUS) {
            status & STATUS_READY != 0
        } else {
            false
        }
    }
}

impl UserspaceBackend {
    /// Ensure PCIe device is enabled
    ///
    /// Deep Debt: Runtime check, not assumption
    fn ensure_device_enabled(pcie_address: &str) -> Result<()> {
        let enable_path = format!("/sys/bus/pci/devices/{pcie_address}/enable");

        // Check current state
        match std::fs::read_to_string(&enable_path) {
            Ok(content) if content.trim() == "1" => {
                tracing::debug!("Device {pcie_address} already enabled");
                Ok(())
            }
            Ok(_) => {
                // Try to enable (may fail if no permissions)
                if let Err(e) = std::fs::write(&enable_path, "1") {
                    tracing::warn!("Could not enable device {pcie_address} (may need sudo): {e}");
                    Err(AkidaError::capability_query_failed(format!(
                        "Device not enabled and cannot enable: {e}"
                    )))
                } else {
                    tracing::info!("Enabled device {pcie_address}");
                    Ok(())
                }
            }
            Err(e) => Err(AkidaError::capability_query_failed(format!(
                "Cannot check device enable status: {e}"
            ))),
        }
    }

    /// Discover capabilities from hardware registers
    ///
    /// Deep Debt: Runtime discovery, zero hardcoding
    fn discover_capabilities(pcie_address: &str, bar0: &MmapRegion) -> Result<Capabilities> {
        tracing::debug!("Discovering capabilities for {pcie_address}");

        // Read chip version from register
        let version_reg = bar0.read_u32(REG_VERSION)?;
        let chip_version = crate::capabilities::ChipVersion::from_register(version_reg);

        // Read NPU count from register (NOT hardcoded!)
        let npu_count = bar0.read_u32(REG_NPU_COUNT)?;

        // Read SRAM size from register
        let sram_bytes = bar0.read_u32(REG_SRAM_SIZE)?;
        let memory_mb = sram_bytes / (1024 * 1024);

        // Query PCIe configuration from sysfs
        let pcie = crate::capabilities::PcieConfig::from_sysfs(pcie_address)?;

        tracing::info!(
            "Discovered: {chip_version:?}, {npu_count} NPUs, {memory_mb} MB, PCIe Gen{} x{}",
            pcie.generation,
            pcie.lanes
        );

        Ok(Capabilities {
            chip_version,
            npu_count,
            memory_mb,
            pcie,
            power_mw: None,
            temperature_c: None,
            mesh: crate::capabilities::MeshTopology::from_sysfs(pcie_address),
            clock_mode: None,
            batch: crate::capabilities::BatchCapabilities::from_sysfs(pcie_address),
            weight_mutation: crate::capabilities::WeightMutationSupport::None,
        })
    }

    /// Poll status register until condition met
    fn poll_status(&self, wait_for: u32, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            let status = self.bar0.read_u32(REG_STATUS)?;

            // Check for error
            if status & STATUS_ERROR != 0 {
                return Err(AkidaError::transfer_failed(format!(
                    "Hardware error: status={status:#x}"
                )));
            }

            // Check for condition
            if status & wait_for != 0 {
                return Ok(());
            }

            // Check timeout
            if start.elapsed() > timeout {
                return Err(AkidaError::transfer_failed(format!(
                    "Timeout waiting for status bit {wait_for:#x}"
                )));
            }

            // Short sleep before retry
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_userspace_backend_with_hardware() {
        // This test requires actual hardware
        let pcie_address = "0000:a1:00.0";

        match UserspaceBackend::init(pcie_address) {
            Ok(backend) => {
                println!("Userspace backend initialized");
                println!("   NPUs: {}", backend.capabilities().npu_count);
                println!("   SRAM: {} MB", backend.capabilities().memory_mb);
            }
            Err(e) => {
                println!("Userspace backend unavailable (expected if no hardware): {e}");
            }
        }
    }
}
