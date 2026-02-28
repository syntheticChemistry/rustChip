//! Low-level I/O operations for Akida devices
//!
//! Handles direct read/write operations to device files with proper
//! error handling and tracing.
//!
//! Deep Debt: Minimal unsafe, well-documented.
//!
//! # Evolution (Feb 12, 2026)
//!
//! Evolved from `nix` to `rustix` for pure Rust syscall wrappers.

use crate::error::{AkidaError, Result};
use rustix::fd::BorrowedFd;
use rustix::io::{read, write};
use std::os::unix::io::RawFd;

/// I/O operations handler
///
/// Wraps a file descriptor for read/write operations.
/// Does not own the file descriptor - the caller retains ownership.
#[derive(Debug)]
pub struct IoHandle {
    fd: RawFd,
}

impl IoHandle {
    /// Create new I/O handler for a file descriptor
    #[must_use]
    pub const fn new(fd: RawFd) -> Self {
        Self { fd }
    }

    /// Read data from device
    ///
    /// # Errors
    ///
    /// Returns error if read operation fails.
    pub fn read(&self, buffer: &mut [u8]) -> Result<usize> {
        // SAFETY: fd is valid for the lifetime of this IoHandle (caller's responsibility)
        let borrowed = unsafe { BorrowedFd::borrow_raw(self.fd) };
        read(borrowed, buffer).map_err(|e| AkidaError::transfer_failed(format!("Read failed: {e}")))
    }

    /// Write data to device
    ///
    /// # Errors
    ///
    /// Returns error if write operation fails.
    pub fn write(&self, data: &[u8]) -> Result<usize> {
        // SAFETY: fd is valid for the lifetime of this IoHandle (caller's responsibility)
        let borrowed = unsafe { BorrowedFd::borrow_raw(self.fd) };
        write(borrowed, data).map_err(|e| AkidaError::transfer_failed(format!("Write failed: {e}")))
    }
}
