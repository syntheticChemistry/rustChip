//! Error types for Akida model operations

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for Akida model operations
pub type Result<T> = std::result::Result<T, AkidaModelError>;

/// Errors that can occur during model parsing and loading
#[derive(Debug, Error)]
pub enum AkidaModelError {
    /// File not found or cannot be read
    #[error("Model file not found: {path}")]
    FileNotFound {
        /// Path that was attempted
        path: PathBuf,
    },

    /// Invalid `FlatBuffers` magic bytes
    #[error("Invalid FlatBuffers header: expected magic bytes \\x80D\\x04\\x10")]
    InvalidHeader,

    /// Unsupported model version
    #[error("Unsupported model version: {version} (expected 2.18.x)")]
    UnsupportedVersion {
        /// Version string from model
        version: String,
    },

    /// Model parsing failed
    #[error("Failed to parse model: {reason}")]
    ParseError {
        /// Reason for failure
        reason: String,
    },

    /// Invalid layer configuration
    #[error("Invalid layer: {reason}")]
    InvalidLayer {
        /// Reason for failure
        reason: String,
    },

    /// I/O error
    #[error("I/O error: {source}")]
    Io {
        /// Underlying I/O error
        #[from]
        source: std::io::Error,
    },

    /// Model loading failed
    #[error("Model loading failed: {reason}")]
    LoadingFailed {
        /// Reason for failure
        reason: String,
    },
}

impl AkidaModelError {
    /// Create a parse error
    pub fn parse_error(reason: impl Into<String>) -> Self {
        Self::ParseError {
            reason: reason.into(),
        }
    }

    /// Create an invalid layer error
    pub fn invalid_layer(reason: impl Into<String>) -> Self {
        Self::InvalidLayer {
            reason: reason.into(),
        }
    }

    /// Create a loading error
    pub fn loading_failed(reason: impl Into<String>) -> Self {
        Self::LoadingFailed {
            reason: reason.into(),
        }
    }
}
