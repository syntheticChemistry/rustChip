// SPDX-License-Identifier: AGPL-3.0-or-later

//! Direct SRAM access for Akida devices.
//!
//! Provides read/write access to all on-chip SRAM through two paths:
//!
//! 1. **BAR0 register access** — control registers and per-NP config at
//!    known offsets (see `akida_chip::regs`).
//! 2. **BAR1 mesh window** — maps the full NP mesh SRAM (16 GB decode,
//!    8 MB physical) for direct weight/activation read/write.
//!
//! ## Usage
//!
//! ```no_run
//! use akida_driver::sram::SramAccessor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Discover device and open SRAM accessor
//! let mgr = akida_driver::DeviceManager::discover()?;
//! let pcie_addr = mgr.devices()[0].pcie_address();
//! let mut sram = SramAccessor::open(pcie_addr)?;
//!
//! // Read control registers (BAR0)
//! let device_id = sram.read_register(akida_chip::regs::DEVICE_ID)?;
//! println!("Device ID: {device_id:#010x}");
//!
//! // Probe BAR1 SRAM
//! let results = sram.probe_bar1(4)?; // probe first 4 NPs
//! for r in &results {
//!     if r.has_data {
//!         println!("{}: {:#010x}", r.description, r.value.unwrap_or(0));
//!     }
//! }
//!
//! // Read/write arbitrary SRAM
//! let data = sram.read_bar1(0x0000, 64)?;
//! sram.write_bar1(0x0000, &[0xDE, 0xAD, 0xBE, 0xEF])?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Safety
//!
//! Writing to SRAM can corrupt loaded models or cause device errors.
//! The accessor provides raw access — the caller is responsible for
//! ensuring writes don't interfere with active inference.

use crate::backends::mmap::MmapRegion;
use crate::error::{AkidaError, Result};
use akida_chip::regs;
use akida_chip::sram::{Bar1Layout, ProbeResult};

/// Direct SRAM accessor for Akida hardware.
///
/// Maps BAR0 (registers) and BAR1 (NP mesh SRAM) for raw read/write.
/// Constructed from a `PCIe` address, not a device path — works with
/// VFIO, sysfs mmap, or any access method that exposes BARs.
pub struct SramAccessor {
    /// BAR0 — control/status registers (16 MB).
    bar0: MmapRegion,
    /// BAR1 — NP mesh SRAM window (maps only the physical SRAM portion).
    bar1: Option<MmapRegion>,
    /// BAR1 layout model.
    layout: Bar1Layout,
    /// `PCIe` address.
    pcie_address: String,
}

impl SramAccessor {
    /// Open SRAM accessor for a device at the given `PCIe` address.
    ///
    /// Maps BAR0 immediately. BAR1 is mapped lazily on first SRAM access
    /// because some systems restrict BAR1 access.
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 cannot be mapped (device not accessible).
    pub fn open(pcie_address: &str) -> Result<Self> {
        tracing::info!("Opening SRAM accessor for {pcie_address}");

        let bar0 = MmapRegion::new(pcie_address, 0)?;
        tracing::info!("BAR0 mapped: {} bytes", bar0.size());

        // Try to discover layout from actual hardware registers
        let layout = Self::discover_layout(&bar0);

        Ok(Self {
            bar0,
            bar1: None,
            layout,
            pcie_address: pcie_address.to_string(),
        })
    }

    /// Discover BAR1 layout from BAR0 registers.
    ///
    /// Reads NP count and SRAM config to construct the actual layout
    /// instead of using hardcoded AKD1000 constants.
    fn discover_layout(bar0: &MmapRegion) -> Bar1Layout {
        // Try NP count register (0x10C0)
        let np_count = bar0
            .read_u32(regs::NP_COUNT)
            .ok()
            .filter(|&v| v > 0 && v < 1000)
            .unwrap_or(78);

        // Try SRAM region registers for physical SRAM size
        let sram_region_0 = bar0.read_u32(regs::SRAM_REGION_0).unwrap_or(0);
        let sram_region_1 = bar0.read_u32(regs::SRAM_REGION_1).unwrap_or(0);

        let physical_sram = if sram_region_0 > 0 && sram_region_1 > 0 {
            let estimate = u64::from(sram_region_0) * u64::from(sram_region_1);
            if estimate > 0 && estimate < 1024 * 1024 * 1024 {
                tracing::info!(
                    "SRAM size from registers: region_0={sram_region_0:#x}, region_1={sram_region_1:#x} → {estimate} bytes"
                );
                estimate
            } else {
                akida_chip::bar::bar1::PHYSICAL_SRAM
            }
        } else {
            akida_chip::bar::bar1::PHYSICAL_SRAM
        };

        tracing::info!("Layout: {np_count} NPs, {physical_sram} bytes physical SRAM");
        Bar1Layout::from_discovered(np_count, physical_sram)
    }

    /// Ensure BAR1 is mapped, mapping it on first access.
    fn ensure_bar1(&mut self) -> Result<&mut MmapRegion> {
        if self.bar1.is_none() {
            tracing::info!("Mapping BAR1 for {}", self.pcie_address);
            let bar1 = MmapRegion::new(&self.pcie_address, 1)?;
            tracing::info!(
                "BAR1 mapped: {} bytes ({} MB)",
                bar1.size(),
                bar1.size() / (1024 * 1024)
            );
            self.bar1 = Some(bar1);
        }
        // Safe: we just ensured it's Some
        self.bar1
            .as_mut()
            .ok_or_else(|| AkidaError::capability_query_failed("BAR1 mapping failed"))
    }

    // ── BAR0 register access ─────────────────────────────────────────────

    /// Read a 32-bit control register from BAR0.
    ///
    /// # Errors
    ///
    /// Returns error if the offset is out of bounds.
    pub fn read_register(&self, offset: usize) -> Result<u32> {
        self.bar0.read_u32(offset)
    }

    /// Write a 32-bit control register to BAR0.
    ///
    /// # Errors
    ///
    /// Returns error if the offset is out of bounds.
    pub fn write_register(&mut self, offset: usize, value: u32) -> Result<()> {
        self.bar0.write_u32(offset, value)
    }

    /// Read the device identity register.
    ///
    /// Returns `0x194000a1` on AKD1000.
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 is not accessible.
    pub fn read_device_id(&self) -> Result<u32> {
        self.read_register(regs::DEVICE_ID)
    }

    /// Read the NP count register.
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 is not accessible.
    pub fn read_np_count(&self) -> Result<u32> {
        self.read_register(regs::NP_COUNT)
    }

    /// Read SRAM region configuration registers.
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 is not accessible.
    pub fn read_sram_config(&self) -> Result<SramConfig> {
        Ok(SramConfig {
            region_0: self.read_register(regs::SRAM_REGION_0)?,
            region_1: self.read_register(regs::SRAM_REGION_1)?,
            bar_addr: self.read_register(regs::SRAM_BAR_ADDR)?,
        })
    }

    /// Read per-NP configuration register.
    ///
    /// # Errors
    ///
    /// Returns error if offset is out of bounds.
    pub fn read_np_config(&self, np_index: u32, reg_offset: usize) -> Result<u32> {
        let offset =
            regs::NP_CONFIG_BASE + (np_index as usize) * regs::NP_CONFIG_STRIDE + reg_offset;
        self.read_register(offset)
    }

    /// Dump all confirmed BAR0 registers.
    ///
    /// Returns a vec of (name, offset, value) tuples.
    ///
    /// # Errors
    ///
    /// Returns error if BAR0 is not accessible.
    pub fn dump_registers(&self) -> Result<Vec<RegisterDump>> {
        let regs_to_read: &[(&str, usize)] = &[
            ("DEVICE_ID", regs::DEVICE_ID),
            ("VERSION", regs::VERSION),
            ("STATUS", regs::STATUS),
            ("CONTROL", regs::CONTROL),
            ("NP_COUNT", regs::NP_COUNT),
            ("DMA_MESH_CONFIG", regs::DMA_MESH_CONFIG),
            ("SRAM_REGION_0", regs::SRAM_REGION_0),
            ("SRAM_REGION_1", regs::SRAM_REGION_1),
            ("SRAM_BAR_ADDR", regs::SRAM_BAR_ADDR),
            ("IRQ_STATUS", regs::IRQ_STATUS),
            ("IRQ_ENABLE", regs::IRQ_ENABLE),
        ];

        let mut results = Vec::with_capacity(regs_to_read.len());
        for &(name, offset) in regs_to_read {
            match self.read_register(offset) {
                Ok(value) => results.push(RegisterDump {
                    name: name.to_string(),
                    offset,
                    value,
                    readable: true,
                }),
                Err(_) => results.push(RegisterDump {
                    name: name.to_string(),
                    offset,
                    value: 0,
                    readable: false,
                }),
            }
        }

        // Also dump NP enable bits
        for i in 0..regs::NP_ENABLE_COUNT {
            let offset = regs::NP_ENABLE_BASE + i * 4;
            let name = format!("NP_ENABLE[{i}]");
            match self.read_register(offset) {
                Ok(value) => results.push(RegisterDump {
                    name,
                    offset,
                    value,
                    readable: true,
                }),
                Err(_) => results.push(RegisterDump {
                    name,
                    offset,
                    value: 0,
                    readable: false,
                }),
            }
        }

        Ok(results)
    }

    // ── BAR1 SRAM access ─────────────────────────────────────────────────

    /// Read bytes from BAR1 SRAM at a given offset.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped or offset is out of bounds.
    pub fn read_bar1(&mut self, offset: usize, length: usize) -> Result<Vec<u8>> {
        let bar1 = self.ensure_bar1()?;
        if offset + length > bar1.size() {
            return Err(AkidaError::transfer_failed(format!(
                "BAR1 read out of bounds: offset={offset:#x}, len={length}, size={:#x}",
                bar1.size()
            )));
        }
        let mut buf = vec![0u8; length];
        bar1.read_bytes(offset, &mut buf)?;
        Ok(buf)
    }

    /// Write bytes to BAR1 SRAM at a given offset.
    ///
    /// **Warning:** Writing to SRAM can corrupt loaded models.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped or offset is out of bounds.
    pub fn write_bar1(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let bar1 = self.ensure_bar1()?;
        if offset + data.len() > bar1.size() {
            return Err(AkidaError::transfer_failed(format!(
                "BAR1 write out of bounds: offset={offset:#x}, len={}, size={:#x}",
                data.len(),
                bar1.size()
            )));
        }
        bar1.write_bytes(offset, data)
    }

    /// Read a single 32-bit word from BAR1 SRAM.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped or offset is out of bounds.
    pub fn read_bar1_u32(&mut self, offset: usize) -> Result<u32> {
        let bar1 = self.ensure_bar1()?;
        bar1.read_u32(offset)
    }

    /// Write a single 32-bit word to BAR1 SRAM.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped or offset is out of bounds.
    pub fn write_bar1_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        let bar1 = self.ensure_bar1()?;
        bar1.write_u32(offset, value)
    }

    /// Probe BAR1 SRAM to discover accessible regions.
    ///
    /// Reads at computed probe offsets and reports which contain data.
    /// This is a non-destructive read-only operation.
    ///
    /// `max_nps`: how many NPs to probe (0 = global region only).
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped.
    pub fn probe_bar1(&mut self, max_nps: u32) -> Result<Vec<ProbeResult>> {
        let points = self.layout.probe_offsets(max_nps);
        let bar1 = self.ensure_bar1()?;
        let bar1_size = bar1.size();

        let mut results = Vec::with_capacity(points.len());

        for point in &points {
            #[allow(clippy::cast_possible_truncation)]
            let offset = point.offset as usize;
            if offset + 4 > bar1_size {
                results.push(ProbeResult {
                    offset: point.offset,
                    description: point.description.clone(),
                    readable: false,
                    value: None,
                    has_data: false,
                });
                continue;
            }

            match bar1.read_u32(offset) {
                Ok(value) => {
                    results.push(ProbeResult {
                        offset: point.offset,
                        description: point.description.clone(),
                        readable: true,
                        value: Some(value),
                        has_data: value != 0 && value != 0xFFFF_FFFF && value != 0xBADF_5040,
                    });
                }
                Err(_) => {
                    results.push(ProbeResult {
                        offset: point.offset,
                        description: point.description.clone(),
                        readable: false,
                        value: None,
                        has_data: false,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Scan a range of BAR1 looking for non-zero data.
    ///
    /// Returns offsets and values of all non-zero 32-bit words in the range.
    /// Useful for finding where data actually lives in the sparse mapping.
    ///
    /// # Errors
    ///
    /// Returns error if BAR1 cannot be mapped.
    pub fn scan_bar1_range(
        &mut self,
        start: usize,
        length: usize,
        stride: usize,
    ) -> Result<Vec<(usize, u32)>> {
        let bar1 = self.ensure_bar1()?;
        let bar1_size = bar1.size();
        let end = (start + length).min(bar1_size.saturating_sub(4));
        let step = stride.max(4);

        let mut hits = Vec::new();
        let mut offset = start;

        while offset <= end {
            if let Ok(value) = bar1.read_u32(offset) {
                if value != 0 && value != 0xFFFF_FFFF {
                    hits.push((offset, value));
                }
            }
            offset += step;
        }

        Ok(hits)
    }

    /// Get the BAR1 layout model.
    #[must_use]
    pub fn layout(&self) -> &Bar1Layout {
        &self.layout
    }

    /// Get the BAR0 mapped size.
    #[must_use]
    pub fn bar0_size(&self) -> usize {
        self.bar0.size()
    }

    /// Get the BAR1 mapped size (0 if not yet mapped).
    #[must_use]
    pub fn bar1_size(&self) -> usize {
        self.bar1.as_ref().map_or(0, MmapRegion::size)
    }
}

/// SRAM region configuration from BAR0 registers.
#[derive(Debug, Clone, Copy)]
pub struct SramConfig {
    /// SRAM region 0 (offset 0x1410).
    pub region_0: u32,
    /// SRAM region 1 (offset 0x1418).
    pub region_1: u32,
    /// SRAM BAR address (offset 0x141C).
    pub bar_addr: u32,
}

/// Result of reading a BAR0 register.
#[derive(Debug, Clone)]
pub struct RegisterDump {
    /// Register name.
    pub name: String,
    /// Offset within BAR0.
    pub offset: usize,
    /// Value read.
    pub value: u32,
    /// Whether the read succeeded.
    pub readable: bool,
}
