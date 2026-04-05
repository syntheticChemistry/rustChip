// SPDX-License-Identifier: AGPL-3.0-or-later

//! Low-level I/O operations for Akida devices
//!
//! Handles direct read/write operations to device files with proper
//! error handling and tracing.
//!
//! Uses `rustix` safe fd borrowing — no unsafe code.

use crate::error::{AkidaError, Result};
use rustix::fd::BorrowedFd;
use rustix::io::{read, write};

/// I/O operations handler (borrowed fd — safe, no lifetime footgun)
///
/// The caller must ensure the underlying file descriptor outlives this handle.
/// Prefer constructing from `&File` or `&OwnedFd` via the `From` impls.
#[derive(Debug, Clone, Copy)]
pub struct IoHandle<'fd> {
    fd: BorrowedFd<'fd>,
}

impl<'fd> IoHandle<'fd> {
    /// Create from any type that can lend a `BorrowedFd`.
    pub fn new<F: std::os::unix::io::AsFd>(source: &'fd F) -> Self {
        Self { fd: source.as_fd() }
    }

    /// Read data from device.
    ///
    /// # Errors
    ///
    /// Returns error if read operation fails.
    pub fn read(self, buffer: &mut [u8]) -> Result<usize> {
        read(self.fd, buffer).map_err(|e| AkidaError::transfer_failed(format!("Read failed: {e}")))
    }

    /// Write data to device.
    ///
    /// # Errors
    ///
    /// Returns error if write operation fails.
    pub fn write(self, data: &[u8]) -> Result<usize> {
        write(self.fd, data).map_err(|e| AkidaError::transfer_failed(format!("Write failed: {e}")))
    }
}
