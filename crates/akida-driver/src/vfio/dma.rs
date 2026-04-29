// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(unsafe_code, reason = "DMA buffers require mmap, mlock, and IOMMU mapping")]

//! DMA buffer management for VFIO-based NPU access.
//!
//! Page-aligned, mlock'd buffers with IOMMU DMA mapping for zero-copy
//! data transfer between host and AKD1000.

use super::ioctls::{
    self, VfioDmaMap, VfioDmaUnmap, ioctl_vfio_iommu_map_dma, ioctl_vfio_iommu_unmap_dma,
};
use crate::error::{AkidaError, Result};
use rustix::mm::{mlock, munlock};
use std::os::unix::io::RawFd;

/// DMA buffer for fast data transfer
#[derive(Debug)]
pub struct DmaBuffer {
    /// Virtual address (user-space)
    vaddr: *mut u8,
    /// IOVA (device-visible address)
    iova: u64,
    /// Size in bytes
    size: usize,
    /// Container fd for cleanup
    container_fd: RawFd,
}

impl DmaBuffer {
    /// Create a new DMA buffer
    pub(super) fn new(container_fd: RawFd, size: usize, iova: u64) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|e| AkidaError::transfer_failed(format!("Invalid DMA buffer layout: {e}")))?;

        // SAFETY: Raw alloc_zeroed necessary for page-aligned DMA buffer (4096). Invariants:
        // (1) Layout from from_size_align, size>0, align 4096 power-of-two; (2) returns valid
        // ptr for layout.size() bytes or null on OOM; (3) dealloc in Drop with same layout.
        let vaddr = unsafe { std::alloc::alloc_zeroed(layout) };

        if vaddr.is_null() {
            return Err(AkidaError::transfer_failed("Failed to allocate DMA buffer"));
        }

        // SAFETY: mlock necessary for VFIO DMA (prevents swap, ensures physical pages).
        // Invariants: (1) vaddr from alloc_zeroed, valid for size bytes; (2) size matches
        // layout.size(); (3) region [vaddr, vaddr+size) entirely within allocation.
        if let Err(e) = unsafe { mlock(vaddr.cast(), size) } {
            // SAFETY: vaddr allocated above with layout; cleanup on error path before return.
            unsafe { std::alloc::dealloc(vaddr, layout) };
            return Err(AkidaError::transfer_failed(format!(
                "Failed to lock DMA memory: {e}"
            )));
        }

        // Truncation safe: struct sizes fit in u32
        #[expect(
            clippy::cast_possible_truncation,
            reason = "DMA size rounded to page alignment"
        )]
        let dma_map = VfioDmaMap {
            argsz: std::mem::size_of::<VfioDmaMap>() as u32,
            flags: ioctls::VFIO_DMA_MAP_FLAG_READ | ioctls::VFIO_DMA_MAP_FLAG_WRITE,
            vaddr: vaddr as u64,
            iova,
            size: size as u64,
        };

        tracing::debug!(
            "DMA map attempt: vaddr={:#x}, iova={:#x}, size={:#x}, flags={:#x}",
            dma_map.vaddr,
            dma_map.iova,
            dma_map.size,
            dma_map.flags
        );

        // SAFETY: VFIO_IOMMU_MAP_DMA ioctl — kernel maps user buffer to IOVA.
        // Invariants: (1) container_fd valid from VFIO container open; (2) dma_map has argsz,
        // vaddr/iova/size from our allocation; (3) _IOW ioctl reads dma_map; (4) layout matches
        // kernel.
        if let Err(e) = ioctl_vfio_iommu_map_dma(container_fd, &dma_map) {
            tracing::warn!("DMA map failed: {e}");
            // SAFETY: vaddr was allocated above with this exact layout and mlock'd
            // successfully, so munlock and dealloc are valid cleanup operations
            unsafe {
                let _ = munlock(vaddr.cast(), size);
                std::alloc::dealloc(vaddr, layout);
            };
            return Err(e);
        }

        tracing::debug!("Created DMA buffer: vaddr={vaddr:p}, iova={iova:#x}, size={size:#x}");

        Ok(Self {
            vaddr,
            iova,
            size,
            container_fd,
        })
    }

    /// Get slice view of buffer for reading
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: (1) vaddr from alloc in new(), valid for size; (2) we own the allocation;
        // (3) &self ensures no concurrent mutation; (4) size unchanged since allocation.
        unsafe { std::slice::from_raw_parts(self.vaddr, self.size) }
    }

    /// Get mutable slice view of buffer for writing
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: (1) vaddr valid for size; (2) &mut self gives exclusive access;
        // (3) no aliasing; (4) size and alignment correct for [u8].
        unsafe { std::slice::from_raw_parts_mut(self.vaddr, self.size) }
    }

    /// Get IOVA (device address)
    pub const fn iova(&self) -> u64 {
        self.iova
    }

    /// Get size
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        // SAFETY: munlock necessary - vaddr was mlock'd in new(); must unlock before dealloc.
        unsafe {
            let _ = munlock(self.vaddr.cast(), self.size);
        };

        let dma_unmap = VfioDmaUnmap {
            argsz: std::mem::size_of::<VfioDmaUnmap>() as u32,
            flags: 0,
            iova: self.iova,
            size: self.size as u64,
        };

        // SAFETY: VFIO_IOMMU_UNMAP_DMA — kernel unmaps IOVA before dealloc.
        // Invariants: (1) container_fd valid; (2) dma_unmap has iova/size from our mapping;
        // (3) layout matches kernel VfioDmaUnmap; (4) called before dealloc.
        ioctl_vfio_iommu_unmap_dma(self.container_fd, &dma_unmap);

        // Layout is infallible here: size is from alloc in new(), 4096 is a power-of-two.
        // Allow in Drop because we cannot propagate errors.
        #[expect(
            clippy::expect_used,
            reason = "Invariant: layout size matches allocation"
        )]
        let layout = std::alloc::Layout::from_size_align(self.size, 4096)
            .expect("Layout valid: size from alloc in new(), 4096 is power-of-two");
        // SAFETY: dealloc necessary; must match alloc_zeroed in new(). Invariants: (1) vaddr
        // from alloc in new(); (2) layout matches; (3) munlock already called; (4) no refs.
        unsafe { std::alloc::dealloc(self.vaddr, layout) };

        tracing::debug!("Freed DMA buffer at iova={:#x}", self.iova);
    }
}

// SAFETY: `Send` — the allocation and VFIO DMA mapping are owned by this struct; `vaddr` is not
// shared as a raw pointer elsewhere. Moving the struct to another thread does not invalidate the
// kernel mapping or the locked pages (`mlock` applies to the address range in the process).
unsafe impl Send for DmaBuffer {}

// SAFETY: `Sync` — concurrent `&` access only goes through `as_slice()` (immutable) or device DMA
// with kernel-coherent semantics; mutation uses `&mut self` / exclusive slice. The IOMMU mapping
// and buffer size are fixed for the lifetime of the value.
unsafe impl Sync for DmaBuffer {}
