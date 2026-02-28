//! FlatBuffer program format for Akida models.
//!
//! Reverse-engineered from `.fbz` model files and C++ engine symbols.
//! Source: BEYOND_SDK.md Discovery 7.
//!
//! ## Format summary
//!
//! A compiled model binary splits into two regions:
//!
//! | Region | Content | Typical size | Varies with weights? |
//! |--------|---------|-------------|----------------------|
//! | `program_info` | NP routing, register writes, structure | 332 B | **No** |
//! | `program_data` | Layer metadata, activation params | 396 B | Only initial values |
//!
//! Weights are NOT stored in either region. They are DMA'd to NP SRAM
//! separately via `set_variable()` or `program_external()`.
//!
//! ## `program_external()` — raw injection
//!
//! ```text
//! program_external(self, bytes, int) -> None
//!   "Program a device using a serialized program info bytes object,
//!    and the address, as it is seen from akida on the device, of
//!    corresponding program data that must have been written beforehand."
//! ```
//!
//! This bypasses the SDK compilation pipeline entirely. Given the
//! FlatBuffer format, we can construct programs directly.

/// Magic bytes at the start of a valid Akida program binary.
pub const FLATBUFFER_MAGIC: &[u8] = b"FBUF";

/// FlatBuffer root table offset location (byte 0 of the binary).
/// On the AKD1000 reference model: root offset = 0x148 (328).
pub const FB_ROOT_OFFSET: usize = 0;

/// SDK version string embedded in `program_info`.
/// Found at byte offset 236 in all captured AKD1000 programs.
pub const SDK_VERSION_OFFSET: usize = 236;

/// Observed SDK version in tested programs.
pub const SDK_VERSION_STR: &str = "2.19.1";

/// Typical sizes from hardware measurements (Discovery 7).
pub mod typical_sizes {
    /// `program_info` for a minimal ESN readout model (50→128→1).
    pub const PROGRAM_INFO_BYTES: usize = 332;
    /// `program_data` for the same model.
    pub const PROGRAM_DATA_BYTES: usize = 396;
    /// Total for the minimal model.
    pub const TOTAL_BYTES: usize = PROGRAM_INFO_BYTES + PROGRAM_DATA_BYTES;
}

/// `.fbz` file format (FlatBuffer + Snappy compression).
pub mod fbz {
    /// Magic bytes identifying a `.fbz` file.
    pub const MAGIC: &[u8] = b"AKIDA";
    /// Compression algorithm used.
    pub const COMPRESSION: &str = "snappy";
    /// Extension.
    pub const EXTENSION: &str = ".fbz";
}

/// Contents of a loaded program binary.
#[derive(Debug, Clone)]
pub struct ProgramBinary {
    /// Raw bytes of the compiled program.
    pub bytes: Vec<u8>,
    /// Split point between program_info and program_data.
    pub info_end: usize,
}

impl ProgramBinary {
    /// Create from raw bytes with an explicit split point.
    #[must_use]
    pub fn new(bytes: Vec<u8>, info_end: usize) -> Self {
        Self { bytes, info_end }
    }

    /// `program_info` slice — NP routing and register writes.
    #[must_use]
    pub fn program_info(&self) -> &[u8] {
        &self.bytes[..self.info_end]
    }

    /// `program_data` slice — layer metadata and activation parameters.
    #[must_use]
    pub fn program_data(&self) -> &[u8] {
        &self.bytes[self.info_end..]
    }

    /// Total program size in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True if the binary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}
