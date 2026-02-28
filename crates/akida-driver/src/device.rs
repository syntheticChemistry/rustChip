//! Akida device handle and operations
//!
//! # Evolution (Feb 12, 2026)
//!
//! Evolved from `libc` constants to `rustix::fs::OFlags` for pure Rust.

use rustix::fs::OFlags;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

use crate::discovery::DeviceInfo;
use crate::error::{AkidaError, Result};
use crate::io::IoHandle;

/// Akida device handle
///
/// Represents an open connection to an Akida neuromorphic processor.
/// Provides safe, high-level access to device operations.
#[derive(Debug)]
pub struct AkidaDevice {
    /// Device information
    info: DeviceInfo,

    /// Underlying device handle
    handle: DeviceHandle,

    /// I/O operations handler
    io: IoHandle,
}

/// Low-level device file handle
#[derive(Debug)]
pub struct DeviceHandle {
    file: File,
}

impl AkidaDevice {
    /// Open an Akida device
    ///
    /// # Errors
    ///
    /// Returns error if device cannot be opened or is not accessible.
    pub fn open(info: &DeviceInfo) -> Result<Self> {
        tracing::debug!("Opening device {}: {}", info.index, info.path.display());

        let handle = DeviceHandle::open(&info.path, info.index)?;
        let io = IoHandle::new(handle.as_raw_fd());

        tracing::info!("Opened device {}: {}", info.index, info.path.display());

        Ok(Self {
            info: info.clone(),
            handle,
            io,
        })
    }

    /// Get device index
    #[must_use]
    pub const fn index(&self) -> usize {
        self.info.index
    }

    /// Get device path
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.info.path
    }

    /// Get device information
    #[must_use]
    pub const fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Write data to device
    ///
    /// Performs a DMA transfer from host memory to device SRAM.
    ///
    /// # Errors
    ///
    /// Returns error if transfer fails or times out.
    pub fn write(&mut self, data: &[u8]) -> Result<usize> {
        self.io.write(data)
    }

    /// Read data from device
    ///
    /// Performs a DMA transfer from device SRAM to host memory.
    ///
    /// # Errors
    ///
    /// Returns error if transfer fails or times out.
    pub fn read(&mut self, buffer: &mut [u8]) -> Result<usize> {
        self.io.read(buffer)
    }

    /// Get raw file descriptor (for advanced use)
    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.handle.as_raw_fd()
    }
}

impl DeviceHandle {
    /// Open device file
    fn open(path: &Path, _index: usize) -> Result<Self> {
        if !path.exists() {
            return Err(AkidaError::device_not_found(path));
        }

        // SAFETY: OFlags::NONBLOCK.bits() is always a valid i32 value (flag bits are small positive values)
        #[allow(clippy::cast_possible_wrap)]
        let nonblock_flag = OFlags::NONBLOCK.bits() as i32;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(nonblock_flag) // Match Python SDK behavior, using rustix
            .open(path)?;

        Ok(Self { file })
    }
}

impl AsRawFd for DeviceHandle {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl Drop for AkidaDevice {
    fn drop(&mut self) {
        tracing::info!(
            "Closing device {}: {}",
            self.info.index,
            self.info.path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::DeviceManager;

    #[test]
    fn test_device_open() {
        let Ok(manager) = DeviceManager::discover() else {
            println!("ℹ️  Skipping test (no hardware)");
            return;
        };

        let device = manager.open_first();
        assert!(device.is_ok());

        let device = device.unwrap();
        println!("✅ Opened device {}", device.index());
    }
}
