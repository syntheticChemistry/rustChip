// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(unsafe_code, reason = "VFIO ioctls require raw syscalls via libc::ioctl")]

//! VFIO ioctl numbers, layouts, and low-level ioctl wrappers.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::redundant_pub_crate)]

use std::ffi::CStr;
use std::os::raw::c_ulong;
use std::os::unix::io::RawFd;

use crate::error::{AkidaError, Result};

/// Helper to create ioctl number: _IO(type, nr) = (type << 8) | nr
const fn io(ty: u8, nr: u8) -> c_ulong {
    ((ty as c_ulong) << 8) | (nr as c_ulong)
}

pub const VFIO_TYPE: u8 = b';';
pub const VFIO_BASE: u8 = 100;

// VFIO container ioctls
pub const VFIO_GET_API_VERSION: c_ulong = io(VFIO_TYPE, VFIO_BASE);
pub const VFIO_CHECK_EXTENSION: c_ulong = io(VFIO_TYPE, VFIO_BASE + 1);
pub const VFIO_SET_IOMMU: c_ulong = io(VFIO_TYPE, VFIO_BASE + 2);

// VFIO group ioctls
pub const VFIO_GROUP_GET_STATUS: c_ulong = io(VFIO_TYPE, VFIO_BASE + 3);
pub const VFIO_GROUP_SET_CONTAINER: c_ulong = io(VFIO_TYPE, VFIO_BASE + 4);
pub const VFIO_GROUP_GET_DEVICE_FD: c_ulong = io(VFIO_TYPE, VFIO_BASE + 6);

// VFIO device ioctls
pub const VFIO_DEVICE_GET_INFO: c_ulong = io(VFIO_TYPE, VFIO_BASE + 7);
#[expect(
    dead_code,
    reason = "VFIO ioctl constant reserved for future region queries"
)]
pub const VFIO_DEVICE_GET_REGION_INFO: c_ulong = io(VFIO_TYPE, VFIO_BASE + 8);
#[expect(dead_code, reason = "Reserved for future IRQ support")]
pub const VFIO_DEVICE_GET_IRQ_INFO: c_ulong = io(VFIO_TYPE, VFIO_BASE + 9);
#[expect(dead_code, reason = "Reserved for future IRQ support")]
pub const VFIO_DEVICE_SET_IRQS: c_ulong = io(VFIO_TYPE, VFIO_BASE + 10);
pub const VFIO_DEVICE_RESET: c_ulong = io(VFIO_TYPE, VFIO_BASE + 11);

// IOMMU DMA mapping
pub const VFIO_IOMMU_MAP_DMA: c_ulong = io(VFIO_TYPE, VFIO_BASE + 13);
pub const VFIO_IOMMU_UNMAP_DMA: c_ulong = io(VFIO_TYPE, VFIO_BASE + 14);

// API version
pub const VFIO_API_VERSION: i32 = 0;

// IOMMU types
#[expect(dead_code, reason = "IOMMU type constant for Type1 v1 compatibility")]
pub const VFIO_TYPE1_IOMMU: u32 = 1;
pub const VFIO_TYPE1V2_IOMMU: u32 = 3;

// Group status flags
pub const VFIO_GROUP_FLAGS_VIABLE: u32 = 1 << 0;
#[expect(dead_code, reason = "Group status flag reserved for status checks")]
pub const VFIO_GROUP_FLAGS_CONTAINER_SET: u32 = 1 << 1;

// DMA map flags
pub const VFIO_DMA_MAP_FLAG_READ: u32 = 1 << 0;
pub const VFIO_DMA_MAP_FLAG_WRITE: u32 = 1 << 1;

/// VFIO device info structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioDeviceInfo {
    pub argsz: u32,
    pub flags: u32,
    pub num_regions: u32,
    pub num_irqs: u32,
}

/// VFIO region info structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioRegionInfo {
    pub argsz: u32,
    pub flags: u32,
    pub index: u32,
    pub cap_offset: u32,
    pub size: u64,
    pub offset: u64,
}

/// VFIO group status structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioGroupStatus {
    pub argsz: u32,
    pub flags: u32,
}

/// VFIO DMA map structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioDmaMap {
    pub argsz: u32,
    pub flags: u32,
    pub vaddr: u64,
    pub iova: u64,
    pub size: u64,
}

/// VFIO DMA unmap structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct VfioDmaUnmap {
    pub argsz: u32,
    pub flags: u32,
    pub iova: u64,
    pub size: u64,
}

/// # Safety
/// `container_fd` must be a valid VFIO container fd.
pub(crate) unsafe fn vfio_get_api_version(container_fd: RawFd) -> i32 {
    unsafe { libc::ioctl(container_fd, VFIO_GET_API_VERSION as _) }
}

/// # Safety
/// `container_fd` must be a valid VFIO container fd.
pub(crate) unsafe fn vfio_check_extension(container_fd: RawFd, extension: u32) -> i32 {
    unsafe { libc::ioctl(container_fd, VFIO_CHECK_EXTENSION as _, extension) }
}

/// # Safety
/// `container_fd` must be a valid VFIO container fd.
pub(crate) unsafe fn vfio_set_iommu(container_fd: RawFd, iommu_type: u32) -> i32 {
    unsafe { libc::ioctl(container_fd, VFIO_SET_IOMMU as _, iommu_type) }
}

/// # Safety
/// `group_fd` must be a valid VFIO group fd; `status` must point to valid `VfioGroupStatus`.
pub(crate) unsafe fn vfio_group_get_status(group_fd: RawFd, status: *mut VfioGroupStatus) -> i32 {
    unsafe { libc::ioctl(group_fd, VFIO_GROUP_GET_STATUS as _, status) }
}

/// # Safety
/// `group_fd` must be valid; `container_fd_ptr` must point to a valid `RawFd`.
pub(crate) unsafe fn vfio_group_set_container(
    group_fd: RawFd,
    container_fd_ptr: *const RawFd,
) -> i32 {
    unsafe { libc::ioctl(group_fd, VFIO_GROUP_SET_CONTAINER as _, container_fd_ptr) }
}

/// # Safety
/// `group_fd` must be valid; `pcie_address` must be a valid null-terminated C string.
pub(crate) unsafe fn vfio_group_get_device_fd(
    group_fd: RawFd,
    pcie_address: *const libc::c_char,
) -> i32 {
    unsafe { libc::ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD as _, pcie_address) }
}

/// # Safety
/// `device_fd` must be valid; `info` must point to valid `VfioDeviceInfo`.
pub(crate) unsafe fn vfio_device_get_info(device_fd: RawFd, info: *mut VfioDeviceInfo) -> i32 {
    unsafe { libc::ioctl(device_fd, VFIO_DEVICE_GET_INFO as _, info) }
}

/// # Safety
/// `container_fd` must be valid; `map` must point to valid `VfioDmaMap`.
pub(crate) unsafe fn vfio_iommu_map_dma(container_fd: RawFd, map: *const VfioDmaMap) -> i32 {
    unsafe { libc::ioctl(container_fd, VFIO_IOMMU_MAP_DMA as _, map) }
}

/// # Safety
/// `container_fd` must be valid; `unmap` must point to valid `VfioDmaUnmap`.
pub(crate) unsafe fn vfio_iommu_unmap_dma(container_fd: RawFd, unmap: *const VfioDmaUnmap) -> i32 {
    unsafe { libc::ioctl(container_fd, VFIO_IOMMU_UNMAP_DMA as _, unmap) }
}

pub(crate) fn ioctl_vfio_get_api_version(container_fd: RawFd) -> i32 {
    // SAFETY: VFIO_GET_API_VERSION ioctl — see VFIO backend init.
    unsafe { vfio_get_api_version(container_fd) }
}

pub(crate) fn ioctl_vfio_check_type1v2(container_fd: RawFd) -> Result<()> {
    // SAFETY: VFIO_CHECK_EXTENSION for Type1v2 — see VFIO backend init.
    let has_type1 = unsafe { vfio_check_extension(container_fd, VFIO_TYPE1V2_IOMMU) };
    if has_type1 != 1 {
        return Err(AkidaError::capability_query_failed(
            "VFIO Type1v2 IOMMU not supported",
        ));
    }
    Ok(())
}

pub(crate) fn ioctl_vfio_group_get_status(group_fd: RawFd) -> Result<VfioGroupStatus> {
    let mut group_status = VfioGroupStatus {
        argsz: std::mem::size_of::<VfioGroupStatus>() as u32,
        flags: 0,
    };
    // SAFETY: VFIO_GROUP_GET_STATUS — group fd from open.
    let ret = unsafe { vfio_group_get_status(group_fd, &raw mut group_status) };
    if ret < 0 || (group_status.flags & VFIO_GROUP_FLAGS_VIABLE) == 0 {
        return Err(AkidaError::capability_query_failed(
            "VFIO group not viable (all devices must be bound to vfio-pci)",
        ));
    }
    Ok(group_status)
}

pub(crate) fn ioctl_vfio_group_set_container(group_fd: RawFd, container_fd: RawFd) -> Result<()> {
    // SAFETY: VFIO_GROUP_SET_CONTAINER — attaches group to container.
    let ret = unsafe { vfio_group_set_container(group_fd, std::ptr::from_ref(&container_fd)) };
    if ret < 0 {
        return Err(AkidaError::capability_query_failed(format!(
            "Failed to set container: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

pub(crate) fn ioctl_vfio_set_iommu(container_fd: RawFd) -> Result<()> {
    // SAFETY: VFIO_SET_IOMMU — enables Type1v2 IOMMU.
    let ret = unsafe { vfio_set_iommu(container_fd, VFIO_TYPE1V2_IOMMU) };
    if ret < 0 {
        return Err(AkidaError::capability_query_failed(format!(
            "Failed to set IOMMU: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

pub(crate) fn ioctl_vfio_group_get_device_fd(group_fd: RawFd, pcie_address: &CStr) -> Result<i32> {
    // SAFETY: VFIO_GROUP_GET_DEVICE_FD — pcie_address is CString.
    let device_fd = unsafe { vfio_group_get_device_fd(group_fd, pcie_address.as_ptr()) };
    if device_fd < 0 {
        return Err(AkidaError::capability_query_failed(format!(
            "Failed to get device fd: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(device_fd)
}

pub(crate) fn ioctl_vfio_device_get_info(device_fd: RawFd) -> Result<VfioDeviceInfo> {
    let mut device_info = VfioDeviceInfo {
        argsz: std::mem::size_of::<VfioDeviceInfo>() as u32,
        ..Default::default()
    };
    // SAFETY: VFIO_DEVICE_GET_INFO — device fd from VFIO_GROUP_GET_DEVICE_FD.
    let ret = unsafe { vfio_device_get_info(device_fd, &raw mut device_info) };
    if ret < 0 {
        return Err(AkidaError::capability_query_failed(format!(
            "Failed to get device info: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(device_info)
}

/// Map a user buffer to IOVA via `VFIO_IOMMU_MAP_DMA`.
pub(crate) fn ioctl_vfio_iommu_map_dma(container_fd: RawFd, map: &VfioDmaMap) -> Result<()> {
    let ret = unsafe { vfio_iommu_map_dma(container_fd, std::ptr::from_ref(map)) };
    if ret < 0 {
        return Err(crate::error::AkidaError::transfer_failed(format!(
            "Failed to map DMA: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// Unmap IOVA via `VFIO_IOMMU_UNMAP_DMA` (best-effort; used from `Drop`).
pub(crate) fn ioctl_vfio_iommu_unmap_dma(container_fd: RawFd, unmap: &VfioDmaUnmap) {
    unsafe {
        let _ = vfio_iommu_unmap_dma(container_fd, std::ptr::from_ref(unmap));
    }
}

/// Issue `VFIO_DEVICE_RESET` to reset the device through the VFIO subsystem.
///
/// Not all devices support this — returns `Ok(())` on success, error on failure.
pub(crate) fn ioctl_vfio_device_reset(device_fd: RawFd) -> Result<()> {
    // SAFETY: VFIO_DEVICE_RESET — device fd from VFIO_GROUP_GET_DEVICE_FD, no args.
    let ret = unsafe { libc::ioctl(device_fd, VFIO_DEVICE_RESET as _) };
    if ret < 0 {
        return Err(AkidaError::hardware_error(format!(
            "VFIO device reset failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}
