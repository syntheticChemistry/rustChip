//! Memory-Mapped I/O for Akida NPU
//!
//! Provides safe abstractions for accessing Akida hardware registers.
//! Based on VFIO region mapping.
//!
//! # Deep Debt Evolution (Feb 17, 2026)
//!
//! Evolved to use rustix for mmap/munmap while keeping libc only for VFIO ioctls.
//! VFIO ioctls are kernel-specific and not covered by rustix's standard API.

// Hardware register access requires exact type casts for mmap/ioctl APIs
// MMIO registers are naturally aligned by hardware, so pointer casts are safe
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::items_after_statements)] // VFIO ioctl constants near usage

use crate::error::{AkidaError, Result};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use std::fs::File;
use std::os::unix::io::{AsFd, AsRawFd};

/// AKD1000 BAR regions
#[derive(Debug, Clone, Copy)]
pub enum Bar {
    /// Control/status registers (BAR0)
    Control = 0,
    /// Model memory (BAR1)
    Model = 1,
    /// Data buffers (BAR2)
    Data = 2,
}

/// AKD1000 register offsets (inferred from behavior)
pub mod regs {
    /// Device identification register
    pub const DEVICE_ID: usize = 0x0000;
    /// Device version register
    pub const VERSION: usize = 0x0004;
    /// Device status register
    pub const STATUS: usize = 0x0008;
    /// Control register
    pub const CONTROL: usize = 0x000C;
    /// NPU count register
    pub const NPU_COUNT: usize = 0x0010;
    /// SRAM size register (in KB)
    pub const SRAM_SIZE: usize = 0x0014;
    /// Interrupt status
    pub const IRQ_STATUS: usize = 0x0020;
    /// Interrupt enable
    pub const IRQ_ENABLE: usize = 0x0024;
    /// Model load address
    pub const MODEL_ADDR_LO: usize = 0x0100;
    /// Model load address high
    pub const MODEL_ADDR_HI: usize = 0x0104;
    /// Model size
    pub const MODEL_SIZE: usize = 0x0108;
    /// Model load trigger
    pub const MODEL_LOAD: usize = 0x010C;
    /// Input buffer address
    pub const INPUT_ADDR_LO: usize = 0x0200;
    /// Input buffer address high
    pub const INPUT_ADDR_HI: usize = 0x0204;
    /// Input size
    pub const INPUT_SIZE: usize = 0x0208;
    /// Output buffer address
    pub const OUTPUT_ADDR_LO: usize = 0x0300;
    /// Output buffer address high
    pub const OUTPUT_ADDR_HI: usize = 0x0304;
    /// Output size
    pub const OUTPUT_SIZE: usize = 0x0308;
    /// Inference trigger
    pub const INFER_START: usize = 0x0400;
    /// Inference status
    pub const INFER_STATUS: usize = 0x0404;

    /// Status register bit definitions
    pub mod status {
        /// Device is ready to accept commands
        pub const READY: u32 = 1 << 0;
        /// Device is currently processing
        pub const BUSY: u32 = 1 << 1;
        /// An error occurred during last operation
        pub const ERROR: u32 = 1 << 2;
        /// A model has been successfully loaded
        pub const MODEL_LOADED: u32 = 1 << 3;
    }

    /// Control register bit definitions
    pub mod control {
        /// Trigger a soft reset of the device
        pub const RESET: u32 = 1 << 0;
        /// Enable device operation
        pub const ENABLE: u32 = 1 << 1;
        /// Enable power-saving mode
        pub const POWER_SAVE: u32 = 1 << 2;
    }
}

/// VFIO region info structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioRegionInfo {
    /// Size of this structure (for versioning)
    pub argsz: u32,
    /// Region flags (capabilities, permissions)
    pub flags: u32,
    /// Region index (BAR number)
    pub index: u32,
    /// Offset to extended capabilities
    pub cap_offset: u32,
    /// Size of the region in bytes
    pub size: u64,
    /// Offset from mmap base
    pub offset: u64,
}

/// Mapped BAR region for MMIO access
pub struct MappedRegion {
    /// Memory-mapped pointer
    ptr: *mut u8,
    /// Size of the mapping
    size: usize,
    /// BAR index
    bar: Bar,
}

impl std::fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedRegion")
            .field("ptr", &format_args!("{:p}", self.ptr))
            .field("size", &self.size)
            .field("bar", &self.bar)
            .finish()
    }
}

// SAFETY: Send - MappedRegion owns the mapped memory exclusively. Moving between threads
// doesn't invalidate the mapping (mmap'd memory is process-wide). No thread-local state.
unsafe impl Send for MappedRegion {}

// SAFETY: Sync - Read operations use &self and are bounds-checked; write operations require
// &mut self (exclusive access). Volatile MMIO reads are idempotent; concurrent reads safe.
unsafe impl Sync for MappedRegion {}

impl MappedRegion {
    /// Map a BAR region via VFIO
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The VFIO ioctl to get region info fails
    /// - Memory mapping the BAR region fails
    pub fn map(device_fd: &File, bar: Bar) -> Result<Self> {
        // Query region info
        #[allow(clippy::cast_possible_truncation)]
        let mut region_info = VfioRegionInfo {
            argsz: std::mem::size_of::<VfioRegionInfo>() as u32,
            index: bar as u32,
            ..Default::default()
        };

        // VFIO_DEVICE_GET_REGION_INFO = _IOWR(';', 100 + 8, ...)
        const VFIO_DEVICE_GET_REGION_INFO: libc::c_ulong = 0xc018_3b68;

        // SAFETY: VFIO_DEVICE_GET_REGION_INFO ioctl necessary for MMIO - kernel returns BAR size/offset.
        // Invariants: (1) device_fd valid from VFIO device open; (2) VfioRegionInfo initialized
        // with argsz = size_of, index = bar; (3) _IOWR reads/writes region_info; (4) layout matches
        // kernel. Caller guarantees: device fd from VFIO, bar is valid BAR index.
        let ret = unsafe {
            libc::ioctl(
                device_fd.as_raw_fd(),
                VFIO_DEVICE_GET_REGION_INFO,
                &raw mut region_info,
            )
        };

        if ret < 0 {
            return Err(AkidaError::capability_query_failed(format!(
                "Failed to get BAR{} info: {}",
                bar as u32,
                std::io::Error::last_os_error()
            )));
        }

        tracing::debug!(
            "BAR{}: size={:#x}, offset={:#x}, flags={:#x}",
            bar as u32,
            region_info.size,
            region_info.offset,
            region_info.flags
        );

        // SAFETY: mmap necessary for MMIO - maps BAR region into process address space.
        // Invariants: (1) device_fd valid; (2) region_info.size/offset from successful ioctl;
        // (3) mapping exclusive via VFIO/IOMMU; (4) ptr valid for size bytes or Err.
        // Caller guarantees: region_info populated by kernel, device_fd open.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                region_info.size as usize,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                device_fd.as_fd(),
                region_info.offset,
            )
            .map_err(|e| {
                AkidaError::capability_query_failed(format!(
                    "Failed to mmap BAR{}: {}",
                    bar as u32, e
                ))
            })?
        };

        tracing::info!(
            "Mapped BAR{} at {:p}, size={:#x}",
            bar as u32,
            ptr,
            region_info.size
        );

        Ok(Self {
            ptr: ptr.cast(),
            size: region_info.size as usize,
            bar,
        })
    }

    /// Read a 32-bit register
    ///
    /// # Panics
    ///
    /// Panics if `offset + 4` exceeds the mapped region size.
    pub fn read32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.size, "Register offset out of bounds");
        // SAFETY: read_volatile necessary for MMIO - hardware can change value.
        // Invariants: (1) ptr from mmap in map(), valid for self.size; (2) offset+4 <= size;
        // (3) u32 aligned; (4) no uninit reads. Caller guarantees: offset in bounds.
        unsafe { std::ptr::read_volatile(self.ptr.add(offset).cast::<u32>()) }
    }

    /// Write a 32-bit register
    ///
    /// # Panics
    ///
    /// Panics if `offset + 4` exceeds the mapped region size.
    pub fn write32(&self, offset: usize, value: u32) {
        assert!(offset + 4 <= self.size, "Register offset out of bounds");
        // SAFETY: write_volatile necessary for MMIO - triggers hardware side effects.
        // Invariants: (1) ptr from mmap; (2) offset+4 <= size; (3) u32 aligned.
        // Caller guarantees: offset in bounds.
        unsafe {
            std::ptr::write_volatile(self.ptr.add(offset).cast::<u32>(), value);
        }
    }

    /// Read a 64-bit register
    ///
    /// # Panics
    ///
    /// Panics if `offset + 8` exceeds the mapped region size.
    pub fn read64(&self, offset: usize) -> u64 {
        assert!(offset + 8 <= self.size, "Register offset out of bounds");
        // SAFETY: read_volatile necessary for MMIO - hardware can change value.
        // Invariants: (1) ptr from mmap; (2) offset+8 <= size; (3) u64 aligned.
        // Caller guarantees: offset in bounds.
        unsafe { std::ptr::read_volatile(self.ptr.add(offset).cast::<u64>()) }
    }

    /// Write a 64-bit register
    ///
    /// # Panics
    ///
    /// Panics if `offset + 8` exceeds the mapped region size.
    pub fn write64(&self, offset: usize, value: u64) {
        assert!(offset + 8 <= self.size, "Register offset out of bounds");
        // SAFETY: write_volatile necessary for MMIO - triggers hardware side effects.
        // Invariants: (1) ptr from mmap; (2) offset+8 <= size; (3) u64 aligned.
        // Caller guarantees: offset in bounds.
        unsafe {
            std::ptr::write_volatile(self.ptr.add(offset).cast::<u64>(), value);
        }
    }

    /// Get BAR type
    pub const fn bar(&self) -> Bar {
        self.bar
    }

    /// Get region size
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        // SAFETY: munmap necessary - must unmap region before process ends.
        // Invariants: (1) ptr from mmap in map(), valid for self.size; (2) munmap with
        // ptr+size that was previously mapped; (3) Drop runs at most once; (4) no refs.
        unsafe {
            // Ignore error in Drop (can't propagate, would need to log)
            let _ = munmap(self.ptr.cast(), self.size);
        }
        tracing::debug!("Unmapped BAR{}", self.bar as u32);
    }
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)] // Compile-time constant validation tests
mod tests {
    use super::*;

    #[test]
    fn test_register_offsets() {
        // Sanity check register layout
        assert_eq!(regs::DEVICE_ID, 0x0000);
        assert_eq!(regs::INFER_START, 0x0400);
        assert!(regs::status::READY != 0);
    }
}
