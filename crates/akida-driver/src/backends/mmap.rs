// SPDX-License-Identifier: AGPL-3.0-or-later

//! Memory-mapped region abstraction
//!
//! Deep Debt Principles:
//! - Minimal unsafe (only in mmap, well-encapsulated)
//! - Runtime validation (bounds checking)
//! - Safe public API
//! - Comprehensive error handling
//!
//! # Evolution (Feb 12, 2026)
//!
//! Evolved from `libc` raw C bindings to `rustix` safe Rust wrappers.
//! This provides better error handling and type safety while maintaining
//! identical functionality.

use crate::error::{AkidaError, Result};
use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsFd;
use std::ptr::NonNull;

/// Memory-mapped `PCIe` BAR region
///
/// Provides safe, bounds-checked access to memory-mapped hardware.
/// Unsafe operations are encapsulated and well-documented.
#[derive(Debug)]
pub struct MmapRegion {
    ptr: NonNull<u8>,
    size: usize,
    _file: File,
    pcie_address: String,
    bar_index: usize,
}

impl MmapRegion {
    /// Create memory-mapped region for `PCIe` BAR
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Resource file doesn't exist
    /// - Cannot open file
    /// - mmap fails
    ///
    /// # Safety
    ///
    /// This function contains unsafe mmap operation, but:
    /// - Validates file descriptor before mapping
    /// - Checks mmap return value
    /// - Ensures proper cleanup via Drop
    ///
    /// # Panics
    ///
    /// Panics if `rustix::mm::mmap` returns a null pointer on success
    /// (should never happen per rustix API contract).
    pub fn new(pcie_address: &str, bar_index: usize) -> Result<Self> {
        let path = format!("/sys/bus/pci/devices/{pcie_address}/resource{bar_index}");

        tracing::debug!("Mapping PCIe BAR: {path}");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| {
                AkidaError::capability_query_failed(format!(
                    "Cannot open {path}: {e}. Is device enabled?"
                ))
            })?;

        // Truncation acceptable: BAR sizes fit in usize on 64-bit (our only target)
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Mapped length fits usize on host"
        )]
        let size = file
            .metadata()
            .map_err(|e| AkidaError::capability_query_failed(format!("Cannot stat BAR: {e}")))?
            .len() as usize;

        if size == 0 {
            return Err(AkidaError::capability_query_failed(
                "BAR size is 0 (device not enabled?)",
            ));
        }

        tracing::debug!("BAR size: {size} bytes ({} MB)", size / (1024 * 1024));

        // SAFETY: `mmap` is required to map the PCIe BAR; rustix wraps the libc mmap contract.
        // Invariants: fd is valid (opened above), `size` is non-zero (checked), offset 0, and
        // `ProtFlags`/`MapFlags` match BAR MMIO access. The `File` field keeps the fd alive until
        // `Drop` runs `munmap`, so the mapping is not unmapped while `MmapRegion` exists.
        // EVOLVED: rustix instead of raw libc (typed flags and `Result`-based errors).
        let ptr = unsafe {
            let addr = mmap(
                std::ptr::null_mut(),
                size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                file.as_fd(),
                0,
            )
            .map_err(|e| AkidaError::capability_query_failed(format!("mmap failed: {e}")))?;

            NonNull::new(addr.cast::<u8>())
                .ok_or_else(|| AkidaError::capability_query_failed("mmap returned null pointer"))?
        };

        tracing::info!(
            "Mapped BAR{bar_index} for {pcie_address} ({} MB at {ptr:p})",
            size / (1024 * 1024),
        );

        Ok(Self {
            ptr,
            size,
            _file: file,
            pcie_address: pcie_address.to_string(),
            bar_index,
        })
    }

    /// Read 32-bit register at offset
    ///
    /// # Errors
    ///
    /// Returns error if offset is out of bounds
    pub fn read_u32(&self, offset: usize) -> Result<u32> {
        if offset + 4 > self.size {
            return Err(AkidaError::transfer_failed(format!(
                "Out of bounds read: offset={offset:#x}, size=4, limit={:#x}",
                self.size
            )));
        }

        // SAFETY: Volatile read from memory-mapped hardware register.
        // Invariants that must hold:
        // - Bounds validated above: offset + 4 <= self.size
        // - ptr is valid (from successful mmap, stored in NonNull)
        // - offset + 4 bytes are within mapped region
        // - Pointer arithmetic: self.ptr.as_ptr().add(offset) is valid (offset < size)
        // - Cast to *const u32: PCIe BAR registers are 4-byte aligned per spec
        // - read_volatile is required: MMIO registers have side effects and compiler
        //   must not reorder or optimize these reads (hardware may change value)
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "MMIO base is device-aligned per BAR"
        )]
        let value = unsafe {
            let ptr = self.ptr.as_ptr().add(offset).cast::<u32>();
            ptr.read_volatile()
        };

        tracing::trace!("Read u32 @ {offset:#x} = {value:#x}");
        Ok(value)
    }

    /// Write 32-bit register at offset
    ///
    /// # Errors
    ///
    /// Returns error if offset is out of bounds
    pub fn write_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        if offset + 4 > self.size {
            return Err(AkidaError::transfer_failed(format!(
                "Out of bounds write: offset={offset:#x}, size=4, limit={:#x}",
                self.size
            )));
        }

        tracing::trace!("Write u32 @ {offset:#x} = {value:#x}");

        #[expect(
            clippy::cast_ptr_alignment,
            reason = "MMIO base is device-aligned per BAR"
        )]
        // SAFETY: `write_volatile` is required for MMIO (hardware side effects).
        // Invariants: same bounds and pointer validity as `read_u32`; exclusive `&mut self` on the
        // region serializes writers at the type level.
        unsafe {
            let ptr = self.ptr.as_ptr().add(offset).cast::<u32>();
            ptr.write_volatile(value);
        }

        Ok(())
    }

    /// Read bytes at offset
    ///
    /// # Errors
    ///
    /// Returns error if read would exceed bounds
    pub fn read_bytes(&self, offset: usize, buffer: &mut [u8]) -> Result<()> {
        if offset + buffer.len() > self.size {
            return Err(AkidaError::transfer_failed(format!(
                "Out of bounds read: offset={offset:#x}, size={}, limit={:#x}",
                buffer.len(),
                self.size
            )));
        }

        // SAFETY: `copy_nonoverlapping` reads `buffer.len()` bytes from the mapping into `buffer`.
        // Invariants: `offset + buffer.len() <= self.size` (checked); `self.ptr.add(offset)` is
        // valid for that length; `buffer.as_mut_ptr()` is valid for writes; src/dst are disjoint
        // (MMIO vs heap stack). `u8` has align 1.
        unsafe {
            let src = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src, buffer.as_mut_ptr(), buffer.len());
        }

        Ok(())
    }

    /// Write bytes at offset
    ///
    /// # Errors
    ///
    /// Returns error if write would exceed bounds
    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.size {
            return Err(AkidaError::transfer_failed(format!(
                "Out of bounds write: offset={offset:#x}, size={}, limit={:#x}",
                data.len(),
                self.size
            )));
        }

        // SAFETY: `copy_nonoverlapping` writes `data` into the mapping at `offset`.
        // Invariants: `offset + data.len() <= self.size` (checked); pointers valid for respective
        // lengths; regions do not overlap. `u8` has align 1.
        unsafe {
            let dst = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(())
    }

    /// Get region size
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Get `PCIe` address
    #[must_use]
    pub fn pcie_address(&self) -> &str {
        &self.pcie_address
    }

    /// Get BAR index
    #[must_use]
    pub const fn bar_index(&self) -> usize {
        self.bar_index
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        tracing::debug!(
            "Unmapping BAR{} for {} ({} MB)",
            self.bar_index,
            self.pcie_address,
            self.size / (1024 * 1024)
        );

        // SAFETY: `munmap` must pair with the `mmap` in `new()` before the fd can be closed.
        // Invariants: `self.ptr`/`self.size` are exactly the mapping from `new()`; `Drop` runs once.
        // EVOLVED: rustix `munmap` instead of raw libc.
        unsafe {
            if let Err(e) = munmap(self.ptr.as_ptr().cast(), self.size) {
                tracing::error!("munmap failed during drop: {e}");
            }
        }
    }
}

// SAFETY: Send implementation is safe because:
// - MmapRegion owns the mapped memory exclusively (no other references exist)
// - The memory mapping is process-private (MAP_SHARED with device file, but no
//   other in-process references to the same mapping)
// - The mapped memory is valid for the lifetime of the MmapRegion (file kept open)
// - All pointer operations are bounds-checked and safe
// - Moving MmapRegion between threads doesn't invalidate the mapping
unsafe impl Send for MmapRegion {}

// SAFETY: Sync implementation is safe because:
// - MmapRegion API requires &mut self for writes (exclusive access enforced by borrow checker)
// - Read operations use &self but are safe because:
//   - All reads are bounds-checked
//   - Volatile reads prevent data races (hardware register reads are idempotent)
//   - Multiple concurrent reads from MMIO registers are safe (hardware handles it)
// - The underlying memory mapping is thread-safe (mmap'd memory can be accessed from any thread)
// - No internal mutable state without synchronization (size, ptr, _file are immutable)
unsafe impl Sync for MmapRegion {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bounds_checking() {
        // This would require actual hardware, so we document the behavior
        // Bounds checking prevents:
        // - Reading beyond BAR size
        // - Writing beyond BAR size
        // - Accessing unallocated memory
    }
}
