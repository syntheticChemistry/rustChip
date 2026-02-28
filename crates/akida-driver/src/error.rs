//! Error types for Akida driver operations

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for Akida operations
pub type Result<T> = std::result::Result<T, AkidaError>;

/// Errors that can occur during Akida operations
#[derive(Debug, Error)]
pub enum AkidaError {
    /// Device not found at the expected path
    #[error("Device not found: {path}")]
    DeviceNotFound {
        /// Path that was checked
        path: PathBuf,
    },

    /// No Akida devices detected on the system
    #[error("No Akida devices detected")]
    NoDevicesFound,

    /// Device index out of range
    #[error("Device index {index} out of range (have {count} devices)")]
    InvalidIndex {
        /// Requested index
        index: usize,
        /// Number of available devices
        count: usize,
    },

    /// I/O error during device communication
    #[error("I/O error: {source}")]
    Io {
        /// Underlying I/O error
        #[from]
        source: std::io::Error,
    },

    /// Data transfer failed
    #[error("Transfer failed: {reason}")]
    TransferFailed {
        /// Reason for failure
        reason: String,
    },

    /// Device capability query failed
    #[error("Failed to query device capabilities: {reason}")]
    CapabilityQueryFailed {
        /// Reason for failure
        reason: String,
    },

    /// Device is in an invalid state
    #[error("Device in invalid state: {state}")]
    InvalidState {
        /// Current state description
        state: String,
    },

    /// Operation timeout
    #[error("Operation timeout after {duration_ms}ms")]
    Timeout {
        /// Timeout duration in milliseconds
        duration_ms: u64,
    },

    /// Hardware-level error from device
    #[error("Hardware error: {reason}")]
    HardwareError {
        /// Reason for failure
        reason: String,
    },
}

impl AkidaError {
    /// Create a device not found error
    pub fn device_not_found(path: impl Into<PathBuf>) -> Self {
        Self::DeviceNotFound { path: path.into() }
    }

    /// Create a transfer failed error
    pub fn transfer_failed(reason: impl Into<String>) -> Self {
        Self::TransferFailed {
            reason: reason.into(),
        }
    }

    /// Create a capability query failed error
    pub fn capability_query_failed(reason: impl Into<String>) -> Self {
        Self::CapabilityQueryFailed {
            reason: reason.into(),
        }
    }

    /// Create an invalid state error
    pub fn invalid_state(state: impl Into<String>) -> Self {
        Self::InvalidState {
            state: state.into(),
        }
    }

    /// Create a hardware error
    pub fn hardware_error(reason: impl Into<String>) -> Self {
        Self::HardwareError {
            reason: reason.into(),
        }
    }
}
