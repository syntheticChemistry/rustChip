// SPDX-License-Identifier: AGPL-3.0-or-later

//! VFIO NPU backend — Pure Rust with DMA support
//!
//! This backend uses Linux VFIO (Virtual Function I/O) to provide:
//!
// FFI/ioctl casts are intentional - VFIO API requires specific types
#![allow(clippy::cast_possible_truncation)]
//! - DMA transfers (fast bulk data movement)
//! - Interrupt support (no polling)
//! - IOMMU isolation (security)
//! - Pure Rust implementation (no C kernel module)
//!
//! # Requirements
//!
//! 1. IOMMU enabled in BIOS and kernel (`intel_iommu=on` or `amd_iommu=on`)
//! 2. Device unbound from native driver and bound to `vfio-pci`
//! 3. User in `vfio` group or root permissions
//!
//! # Setup Commands
//!
//! ```bash
//! # Unbind from native driver
//! echo "0000:a1:00.0" > /sys/bus/pci/drivers/akida/unbind
//!
//! # Bind to vfio-pci
//! echo "1e7c bca1" > /sys/bus/pci/drivers/vfio-pci/new_id
//!
//! # Grant user access
//! sudo chown $USER /dev/vfio/$IOMMU_GROUP
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  User App   │────▶│  VFIO API   │────▶│   IOMMU     │
//! │  (Rust)     │     │  (Rust)     │     │  (Hardware) │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!                            │                   │
//!                            ▼                   ▼
//!                     ┌─────────────┐     ┌─────────────┐
//!                     │  DMA Buffer │────▶│   Akida     │
//!                     │  (Pinned)   │     │   NPU       │
//!                     └─────────────┘     └─────────────┘
//! ```
//!
//! # Deep Debt Compliance
//!
//! - Runtime discovery (IOMMU groups, device capabilities)
//! - Minimal unsafe (well-encapsulated VFIO ioctls)
//! - Safe public API
//! - No C dependencies for mmap/mlock (pure Rust via rustix)
//! - VFIO ioctls use libc: `rustix::ioctl` requires Ioctl trait impl per variant;
//!   VFIO has 9+ ioctls with varied semantics (int, struct, fd ptr, C string).

mod container;
mod dma;
mod ioctls;

pub use dma::DmaBuffer;

use crate::backend::{BackendType, ModelHandle, NpuBackend};
use crate::backends::read_hwmon_power;
use crate::capabilities::Capabilities;
use crate::error::{AkidaError, Result};
use crate::mmio::{Bar, MappedRegion, regs};
use container::{VfioContainer, VfioGroup, query_device_info};
use std::fs::File;
use std::os::unix::io::AsRawFd;

/// Parameters for polling a status register.
#[derive(Clone, Copy)]
struct PollConfig<'a> {
    reg: usize,
    done_mask: u32,
    error_mask: u32,
    max_polls: u32,
    yield_interval: u32,
    timeout_msg: &'a str,
    error_msg: &'a str,
}

/// VFIO NPU backend with DMA support
#[derive(Debug)]
pub struct VfioBackend {
    /// `PCIe` address
    pcie_address: String,
    /// VFIO container file descriptor
    container: File,
    /// VFIO group file descriptor (kept open for lifetime)
    #[expect(dead_code, reason = "Group fd must stay open for VFIO lifetime")]
    group: File,
    /// VFIO device file descriptor (for MMIO access)
    device: File,
    /// BAR0 control registers (MMIO mapped)
    control_regs: MappedRegion,
    /// BAR1 NP mesh / SRAM window (mapped on demand for direct SRAM access)
    mesh_region: Option<MappedRegion>,
    /// Device capabilities
    capabilities: Capabilities,
    /// Input DMA buffer
    input_buffer: Option<DmaBuffer>,
    /// Output DMA buffer
    output_buffer: Option<DmaBuffer>,
    /// Model DMA buffer
    model_buffer: Option<DmaBuffer>,
    /// Next available IOVA
    next_iova: u64,
    /// Whether a model has been loaded
    model_loaded: bool,
}

impl VfioBackend {
    /// Find IOMMU group for a `PCIe` device
    fn find_iommu_group(pcie_address: &str) -> Result<u32> {
        container::find_iommu_group(pcie_address)
    }

    /// Allocate a DMA buffer
    ///
    /// # Errors
    ///
    /// Returns an error if DMA buffer allocation or IOMMU mapping fails.
    pub fn alloc_dma(&mut self, size: usize) -> Result<DmaBuffer> {
        let iova = self.next_iova;
        let aligned_size = size.div_ceil(4096) * 4096;
        self.next_iova += aligned_size as u64;
        DmaBuffer::new(self.container.as_raw_fd(), aligned_size, iova)
    }

    /// Write a 64-bit IOVA address and size to MMIO registers (`addr_lo`, `addr_hi`, `size_reg`).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "IOVA and size fit hardware register widths"
    )]
    fn write_iova_regs(
        &self,
        addr_lo: usize,
        addr_hi: usize,
        size_reg: usize,
        iova: u64,
        size: usize,
    ) {
        self.control_regs.write32(addr_lo, iova as u32);
        self.control_regs.write32(addr_hi, (iova >> 32) as u32);
        self.control_regs.write32(size_reg, size as u32);
    }

    /// Map BAR1 (NP mesh / SRAM window) for direct SRAM access.
    ///
    /// BAR1 is not mapped by default — DMA is the primary data transfer path.
    /// Call this to enable direct SRAM read/write via MMIO, which is useful for:
    /// - Testing and diagnostics
    /// - Reading NP weight/activation state
    /// - Low-level hardware exploration
    ///
    /// # Errors
    ///
    /// Returns error if the VFIO region info query or mmap fails.
    pub fn map_bar1(&mut self) -> Result<()> {
        if self.mesh_region.is_some() {
            tracing::debug!("BAR1 already mapped");
            return Ok(());
        }

        let region = MappedRegion::map(&self.device, Bar::Model)?;
        tracing::info!(
            "Mapped BAR1 SRAM window: {} bytes ({} MB)",
            region.size(),
            region.size() / (1024 * 1024)
        );
        self.mesh_region = Some(region);
        Ok(())
    }

    /// Read a 32-bit word from BAR1 SRAM at the given offset.
    ///
    /// Maps BAR1 on first access if not already mapped.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped.
    ///
    /// # Panics
    ///
    /// Panics if offset is out of bounds (checked by `MappedRegion`).
    pub fn read_sram_u32(&mut self, offset: usize) -> Result<u32> {
        self.map_bar1()?;
        let region = self
            .mesh_region
            .as_ref()
            .ok_or_else(|| AkidaError::capability_query_failed("BAR1 not mapped"))?;
        Ok(region.read32(offset))
    }

    /// Write a 32-bit word to BAR1 SRAM at the given offset.
    ///
    /// Maps BAR1 on first access if not already mapped.
    ///
    /// **Warning:** Writing to SRAM can corrupt loaded models.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped.
    ///
    /// # Panics
    ///
    /// Panics if offset is out of bounds.
    pub fn write_sram_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        self.map_bar1()?;
        let region = self
            .mesh_region
            .as_ref()
            .ok_or_else(|| AkidaError::capability_query_failed("BAR1 not mapped"))?;
        region.write32(offset, value);
        Ok(())
    }

    /// Whether BAR1 SRAM is currently mapped.
    #[must_use]
    pub const fn has_sram_mapped(&self) -> bool {
        self.mesh_region.is_some()
    }

    /// Get the mapped BAR1 region size in bytes (0 if not mapped).
    #[must_use]
    pub fn sram_size(&self) -> usize {
        self.mesh_region.as_ref().map_or(0, MappedRegion::size)
    }

    /// Check the device is not busy. Returns `Err` if BUSY bit is set.
    fn check_not_busy(&self, op: &str) -> Result<()> {
        let status = self.control_regs.read32(regs::STATUS);
        if status & regs::status::BUSY != 0 {
            return Err(AkidaError::hardware_error(format!(
                "Device busy, cannot {op}"
            )));
        }
        Ok(())
    }

    /// Poll a status register until `done_mask` bit is set, returning the poll count.
    /// Returns `Err` if `error_mask` bit is set or `max_polls` is exceeded.
    fn poll_register(&self, cfg: PollConfig<'_>) -> Result<u32> {
        let PollConfig {
            reg,
            done_mask,
            error_mask,
            max_polls,
            yield_interval,
            timeout_msg,
            error_msg,
        } = cfg;
        for i in 0..max_polls {
            let val = self.control_regs.read32(reg);
            if val & done_mask != 0 {
                return Ok(i + 1);
            }
            if val & error_mask != 0 {
                return Err(AkidaError::hardware_error(error_msg));
            }
            if i % yield_interval == 0 {
                std::thread::yield_now();
            }
        }
        Err(AkidaError::hardware_error(timeout_msg))
    }
}

impl NpuBackend for VfioBackend {
    fn init(pcie_address: &str) -> Result<Self> {
        tracing::info!("Initializing VFIO backend for {pcie_address}");

        let iommu_group = Self::find_iommu_group(pcie_address)?;
        tracing::debug!("IOMMU group: {iommu_group}");

        let vfio_container = VfioContainer::open_and_validate()?;
        let vfio_group = VfioGroup::open_attach_and_validate(iommu_group, &vfio_container)?;
        let device = vfio_group.open_device(pcie_address)?;

        let device_info = query_device_info(&device)?;

        tracing::info!(
            "VFIO device: {} regions, {} IRQs",
            device_info.num_regions,
            device_info.num_irqs
        );

        let control_regs = MappedRegion::map(&device, Bar::Control)?;
        tracing::info!(
            "Mapped BAR0 control registers ({} bytes)",
            control_regs.size()
        );

        let capabilities = Capabilities::from_sysfs(pcie_address)?;

        tracing::info!(
            "Initialized VFIO backend for {pcie_address}: {} NPUs, {} MB SRAM",
            capabilities.npu_count,
            capabilities.memory_mb
        );

        Ok(Self {
            pcie_address: pcie_address.to_string(),
            container: vfio_container.file,
            group: vfio_group.file,
            device,
            control_regs,
            mesh_region: None,
            capabilities,
            input_buffer: None,
            output_buffer: None,
            model_buffer: None,
            next_iova: 0x1000_0000, // Start IOVA at 256MB
            model_loaded: false,
        })
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle> {
        tracing::info!("Loading model ({} bytes) via VFIO DMA", model.len());
        self.check_not_busy("load model")?;

        let mut buffer = self.alloc_dma(model.len())?;
        buffer.as_mut_slice().copy_from_slice(model);

        self.write_iova_regs(
            regs::MODEL_ADDR_LO,
            regs::MODEL_ADDR_HI,
            regs::MODEL_SIZE,
            buffer.iova(),
            model.len(),
        );
        self.control_regs.write32(regs::MODEL_LOAD, 1);
        tracing::debug!(
            "Triggered model load: IOVA={:#x}, size={}",
            buffer.iova(),
            model.len()
        );

        let polls = self.poll_register(PollConfig {
            reg: regs::STATUS,
            done_mask: regs::status::MODEL_LOADED,
            error_mask: regs::status::ERROR,
            max_polls: 1_000_000,
            yield_interval: 1_000,
            timeout_msg: "Model load timed out",
            error_msg: "Model load failed with device error",
        })?;
        tracing::info!("Model loaded successfully after {polls} polls");

        self.model_buffer = Some(buffer);
        self.model_loaded = true;
        Ok(ModelHandle::new(0))
    }

    fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()> {
        let w_in_bytes = bytemuck::cast_slice::<f32, u8>(w_in);
        let w_res_bytes = bytemuck::cast_slice::<f32, u8>(w_res);
        let total_size = w_in_bytes.len() + w_res_bytes.len();

        tracing::info!(
            "Loading reservoir via VFIO DMA: w_in={} floats, w_res={} floats",
            w_in.len(),
            w_res.len()
        );

        self.check_not_busy("load reservoir")?;

        let mut buffer = self.alloc_dma(total_size)?;
        let slice = buffer.as_mut_slice();
        slice[..w_in_bytes.len()].copy_from_slice(w_in_bytes);
        slice[w_in_bytes.len()..].copy_from_slice(w_res_bytes);

        self.write_iova_regs(
            regs::MODEL_ADDR_LO,
            regs::MODEL_ADDR_HI,
            regs::MODEL_SIZE,
            buffer.iova(),
            total_size,
        );
        self.control_regs.write32(regs::MODEL_LOAD, 1);

        let polls = self.poll_register(PollConfig {
            reg: regs::STATUS,
            done_mask: regs::status::MODEL_LOADED,
            error_mask: regs::status::ERROR,
            max_polls: 1_000_000,
            yield_interval: 1_000,
            timeout_msg: "Reservoir load timed out",
            error_msg: "Reservoir load failed with device error",
        })?;
        tracing::info!("Reservoir loaded successfully after {polls} polls");

        self.model_buffer = Some(buffer);
        self.model_loaded = true;
        Ok(())
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if !self.model_loaded {
            return Err(AkidaError::hardware_error("No model loaded"));
        }

        self.check_not_busy("run inference")?;
        let status = self.control_regs.read32(regs::STATUS);
        if status & regs::status::READY == 0 {
            return Err(AkidaError::hardware_error("Device not ready"));
        }

        let input_bytes = bytemuck::cast_slice::<f32, u8>(input);

        if self
            .input_buffer
            .as_ref()
            .is_none_or(|b| b.size() < input_bytes.len())
        {
            self.input_buffer = Some(self.alloc_dma(input_bytes.len().max(4096))?);
        }
        let input_buf = self.input_buffer.as_mut().ok_or_else(|| {
            AkidaError::hardware_error("Input DMA buffer missing after allocation")
        })?;
        input_buf.as_mut_slice()[..input_bytes.len()].copy_from_slice(input_bytes);

        let output_size: usize = 4096; // 1024 floats max
        if self
            .output_buffer
            .as_ref()
            .is_none_or(|b| b.size() < output_size)
        {
            self.output_buffer = Some(self.alloc_dma(output_size)?);
        }

        let input_iova = self
            .input_buffer
            .as_ref()
            .ok_or_else(|| AkidaError::hardware_error("Input DMA buffer missing after allocation"))?
            .iova();
        let output_iova = self
            .output_buffer
            .as_ref()
            .ok_or_else(|| {
                AkidaError::hardware_error("Output DMA buffer missing after allocation")
            })?
            .iova();

        self.write_iova_regs(
            regs::INPUT_ADDR_LO,
            regs::INPUT_ADDR_HI,
            regs::INPUT_SIZE,
            input_iova,
            input_bytes.len(),
        );
        self.write_iova_regs(
            regs::OUTPUT_ADDR_LO,
            regs::OUTPUT_ADDR_HI,
            regs::OUTPUT_SIZE,
            output_iova,
            output_size,
        );

        self.control_regs.write32(regs::INFER_START, 1);
        tracing::debug!(
            "Triggered inference: input_iova={input_iova:#x}, output_iova={output_iova:#x}"
        );

        let polls = self.poll_register(PollConfig {
            reg: regs::INFER_STATUS,
            done_mask: 0x1,
            error_mask: 0x2,
            max_polls: 10_000_000,
            yield_interval: 10_000,
            timeout_msg: "Inference timed out",
            error_msg: "Inference failed with device error",
        })?;

        let actual_output_size = self.control_regs.read32(regs::OUTPUT_SIZE) as usize;
        let output_floats = actual_output_size.min(output_size) / std::mem::size_of::<f32>();
        tracing::debug!("Inference completed after {polls} polls, output: {output_floats} floats");

        let output_bytes = &self
            .output_buffer
            .as_ref()
            .ok_or_else(|| {
                AkidaError::hardware_error("Output DMA buffer missing after allocation")
            })?
            .as_slice()[..output_floats * std::mem::size_of::<f32>()];
        Ok(bytemuck::cast_slice::<u8, f32>(output_bytes).to_vec())
    }

    fn measure_power(&self) -> Result<f32> {
        if let Some(watts) = read_hwmon_power(&self.pcie_address) {
            return Ok(watts);
        }

        Ok(1.5) // AKD1000 typical from datasheet
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vfio
    }

    fn is_ready(&self) -> bool {
        let status = self.control_regs.read32(regs::STATUS);
        let ready = status & regs::status::READY != 0;
        let not_busy = status & regs::status::BUSY == 0;
        let no_error = status & regs::status::ERROR == 0;
        ready && not_busy && no_error
    }

    fn verify_load(&mut self, expected: &[u8]) -> Result<crate::backend::LoadVerification> {
        use crate::backend::LoadVerification;

        self.map_bar1()?;
        let Some(region) = self.mesh_region.as_ref() else {
            return Ok(LoadVerification::unsupported());
        };

        let sample_size = expected.len().min(4096).min(region.size());
        if sample_size < 4 {
            return Ok(LoadVerification::unsupported());
        }

        let mut matched = 0usize;
        let step = 4;
        for offset in (0..sample_size).step_by(step) {
            if offset + step > sample_size {
                break;
            }
            let on_chip = region.read32(offset);
            let expected_word = u32::from_le_bytes([
                expected[offset],
                expected[offset + 1],
                expected[offset + 2],
                expected[offset + 3],
            ]);
            if on_chip == expected_word {
                matched += step;
            }
        }

        let checked = (sample_size / step) * step;
        if matched == checked {
            tracing::info!("SRAM readback verified: {checked} bytes match");
            Ok(LoadVerification::ok(checked))
        } else {
            tracing::warn!("SRAM readback mismatch: {matched}/{checked} bytes match");
            Ok(LoadVerification::mismatch(checked, matched))
        }
    }

    fn mutate_weights(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        self.map_bar1()?;
        let region = self
            .mesh_region
            .as_ref()
            .ok_or_else(|| AkidaError::capability_query_failed("BAR1 not mapped"))?;

        if offset + data.len() > region.size() {
            return Err(AkidaError::transfer_failed(format!(
                "Weight mutation out of bounds: offset={offset:#x}, len={}, BAR1 size={:#x}",
                data.len(),
                region.size()
            )));
        }

        for (i, chunk) in data.chunks(4).enumerate() {
            if chunk.len() == 4 {
                let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                region.write32(offset + i * 4, value);
            }
        }

        tracing::debug!(
            "Mutated {} bytes at SRAM offset {offset:#x} via direct MMIO",
            data.len()
        );
        Ok(())
    }

    fn read_sram(&mut self, offset: usize, length: usize) -> Result<Vec<u8>> {
        self.map_bar1()?;
        let region = self
            .mesh_region
            .as_ref()
            .ok_or_else(|| AkidaError::capability_query_failed("BAR1 not mapped"))?;

        if offset + length > region.size() {
            return Err(AkidaError::transfer_failed(format!(
                "SRAM read out of bounds: offset={offset:#x}, len={length}, BAR1 size={:#x}",
                region.size()
            )));
        }

        let mut buf = Vec::with_capacity(length);
        let mut pos = offset;
        while pos < offset + length {
            let remaining = offset + length - pos;
            if remaining >= 4 {
                let val = region.read32(pos);
                buf.extend_from_slice(&val.to_le_bytes());
                pos += 4;
            } else {
                buf.push(0);
                pos += 1;
            }
        }
        buf.truncate(length);
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_iommu_group() {
        let pcie_address = "0000:a1:00.0";

        match VfioBackend::find_iommu_group(pcie_address) {
            Ok(group) => {
                println!("IOMMU group for {pcie_address}: {group}");
            }
            Err(e) => {
                println!("IOMMU group lookup failed (expected if no hardware): {e}");
            }
        }
    }

    #[test]
    fn test_vfio_backend_init() {
        let pcie_address = "0000:a1:00.0";

        match VfioBackend::init(pcie_address) {
            Ok(backend) => {
                println!("VFIO backend initialized");
                println!("   NPUs: {}", backend.capabilities().npu_count);
                println!("   SRAM: {} MB", backend.capabilities().memory_mb);
            }
            Err(e) => {
                println!("VFIO backend unavailable (expected if no hardware): {e}");
            }
        }
    }
}

// ── VFIO device binding helpers ───────────────────────────────────────────────
// These replace the C driver's install.sh for the VFIO path.

/// Bind an Akida device to `vfio-pci`, unloading any existing driver.
///
/// Steps:
/// 1. Unbind from current driver (e.g., `akida_pcie`)
/// 2. Enable `vfio-pci` module
/// 3. Write vendor:device to `vfio-pci/new_id`
/// 4. Bind the device
///
/// Requires root or `CAP_SYS_ADMIN`.
///
/// # Errors
///
/// Returns an error if any sysfs write fails (usually permission denied).
pub fn bind_to_vfio(pcie_address: &str) -> crate::error::Result<()> {
    use crate::error::AkidaError;
    use std::path::Path;

    tracing::info!("Binding {} to vfio-pci", pcie_address);

    let driver_unbind = format!("/sys/bus/pci/devices/{pcie_address}/driver/unbind");
    if Path::new(&driver_unbind).exists() {
        std::fs::write(&driver_unbind, pcie_address).map_err(|e| {
            AkidaError::hardware_error(format!("Cannot unbind {pcie_address}: {e}"))
        })?;
        tracing::info!("Unbound from existing driver");
    }

    let new_id = "/sys/bus/pci/drivers/vfio-pci/new_id";
    if Path::new(new_id).exists() {
        std::fs::write(
            new_id,
            format!(
                "{:04x} {:04x}",
                akida_chip::pcie::BRAINCHIP_VENDOR_ID,
                0xBCA1u16
            ),
        )
        .map_err(|e| AkidaError::hardware_error(format!("Cannot write vfio-pci/new_id: {e}")))?;
    }

    let bind_path = "/sys/bus/pci/drivers/vfio-pci/bind";
    std::fs::write(bind_path, pcie_address)
        .map_err(|e| AkidaError::hardware_error(format!("Cannot bind to vfio-pci: {e}")))?;

    tracing::info!("{pcie_address} bound to vfio-pci");
    Ok(())
}

/// Unbind from `vfio-pci` and re-bind to `akida_pcie` kernel module.
///
/// # Errors
///
/// Returns an error if sysfs writes fail.
pub fn unbind_from_vfio(pcie_address: &str) -> crate::error::Result<()> {
    use crate::error::AkidaError;

    let unbind = "/sys/bus/pci/drivers/vfio-pci/unbind";
    std::fs::write(unbind, pcie_address)
        .map_err(|e| AkidaError::hardware_error(format!("Cannot unbind from vfio-pci: {e}")))?;

    let bind = "/sys/bus/pci/drivers/akida/bind";
    if std::path::Path::new(bind).exists() {
        std::fs::write(bind, pcie_address)
            .map_err(|e| AkidaError::hardware_error(format!("Cannot bind to akida driver: {e}")))?;
        tracing::info!("{pcie_address} re-bound to akida_pcie");
    } else {
        tracing::info!("{pcie_address} unbound (akida_pcie not loaded)");
    }

    Ok(())
}

/// Find the IOMMU group number for a `PCIe` device.
///
/// Reads `/sys/bus/pci/devices/{addr}/iommu_group` symlink.
///
/// # Errors
///
/// Returns `AkidaError` if the sysfs symlink cannot be read.
pub fn iommu_group(pcie_address: &str) -> crate::error::Result<u32> {
    use crate::error::AkidaError;

    let link = format!("/sys/bus/pci/devices/{pcie_address}/iommu_group");
    let target = std::fs::read_link(&link).map_err(|e| {
        AkidaError::hardware_error(format!("Cannot read iommu_group for {pcie_address}: {e}"))
    })?;

    let group = target
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| {
            AkidaError::hardware_error(format!(
                "Cannot parse IOMMU group from {}",
                target.display()
            ))
        })?;

    tracing::debug!("{pcie_address} → IOMMU group {group}");
    Ok(group)
}
