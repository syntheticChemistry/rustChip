// SPDX-License-Identifier: AGPL-3.0-or-later

//! VFIO container and IOMMU group setup.

use std::fs::OpenOptions;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::fs::File;

use crate::error::{AkidaError, Result};

use super::ioctls::{
    self, ioctl_vfio_device_get_info, ioctl_vfio_group_get_device_fd, ioctl_vfio_group_get_status,
    ioctl_vfio_group_set_container, ioctl_vfio_set_iommu, ioctl_vfio_check_type1v2,
    ioctl_vfio_get_api_version, VfioDeviceInfo,
};

/// Find IOMMU group number for a `PCIe` device (sysfs).
pub(super) fn find_iommu_group(pcie_address: &str) -> Result<u32> {
    let iommu_group_path = format!("/sys/bus/pci/devices/{pcie_address}/iommu_group");

    let link = std::fs::read_link(&iommu_group_path).map_err(|e| {
        AkidaError::capability_query_failed(format!(
            "Cannot read IOMMU group for {pcie_address}: {e}. Is IOMMU enabled?"
        ))
    })?;

    let group_name = link
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| AkidaError::capability_query_failed("Invalid IOMMU group path"))?;

    group_name.parse::<u32>().map_err(|e| {
        AkidaError::capability_query_failed(format!("Invalid IOMMU group number: {e}"))
    })
}

/// Opened `/dev/vfio/vfio` with API version validated.
pub(super) struct VfioContainer {
    pub file: File,
}

impl VfioContainer {
    pub(super) fn open_and_validate() -> Result<Self> {
        let container = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/vfio/vfio")
            .map_err(|e| {
                AkidaError::capability_query_failed(format!("Cannot open /dev/vfio/vfio: {e}"))
            })?;

        let api_version = ioctl_vfio_get_api_version(container.as_raw_fd());
        if api_version != ioctls::VFIO_API_VERSION {
            return Err(AkidaError::capability_query_failed(format!(
                "Unsupported VFIO API version: {api_version}"
            )));
        }

        ioctl_vfio_check_type1v2(container.as_raw_fd())?;

        Ok(Self { file: container })
    }

    pub(super) fn set_iommu(&self) -> Result<()> {
        ioctl_vfio_set_iommu(self.file.as_raw_fd())
    }

    pub(super) fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// Opened `/dev/vfio/{group}` after viability check and attachment to a container.
pub(super) struct VfioGroup {
    pub file: File,
}

impl VfioGroup {
    pub(super) fn open_attach_and_validate(
        iommu_group: u32,
        container: &VfioContainer,
    ) -> Result<Self> {
        let group_path = format!("/dev/vfio/{iommu_group}");
        let group = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&group_path)
            .map_err(|e| {
                AkidaError::capability_query_failed(format!("Cannot open {group_path}: {e}"))
            })?;

        ioctl_vfio_group_get_status(group.as_raw_fd())?;
        ioctl_vfio_group_set_container(group.as_raw_fd(), container.as_raw_fd())?;
        container.set_iommu()?;

        Ok(Self { file: group })
    }

    /// Obtain VFIO device fd for the given `PCIe` address string.
    pub(super) fn open_device(&self, pcie_address: &str) -> Result<File> {
        let pcie_address_cstr = std::ffi::CString::new(pcie_address).map_err(|e| {
            AkidaError::capability_query_failed(format!("Invalid PCIe address: {e}"))
        })?;

        let device_fd = ioctl_vfio_group_get_device_fd(self.file.as_raw_fd(), &pcie_address_cstr)?;

        // SAFETY: `device_fd` from VFIO ioctl; ownership transferred.
        let device = unsafe { File::from_raw_fd(device_fd) };
        Ok(device)
    }
}

/// Query VFIO device info via ioctl.
pub(super) fn query_device_info(device: &File) -> Result<VfioDeviceInfo> {
    ioctl_vfio_device_get_info(device.as_raw_fd())
}
