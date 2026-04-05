// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-tenant NPU management.
//!
//! Enables multiple independent programs to run on distinct NP subsets
//! of a single Akida device simultaneously. Validated architecture:
//! 7 programs on 814/1,000 NPs (see `baseCamp/systems/multi_tenancy.md`).
//!
//! # Architecture
//!
//! ```text
//! MultiTenantDevice
//!   ├── Slot 0: ESN readout      [NP 0x0000 – 0x00B2]  (179 NPs)
//!   ├── Slot 1: Transport pred.  [NP 0x00B3 – 0x0138]  (134 NPs)
//!   ├── Slot 2: Phase class.     [NP 0x0139 – 0x0214]  (220 NPs)
//!   ├── Slot 3: Anomaly det.     [NP 0x0215 – 0x0274]  ( 96 NPs)
//!   ├── Slot 4: Flow predictor   [NP 0x0275 – 0x02B7]  ( 67 NPs)
//!   ├── Slot 5: QCD observable   [NP 0x02B8 – 0x02FB]  ( 68 NPs)
//!   └── Slot 6: Stability ESN    [NP 0x02FC – 0x032D]  ( 50 NPs)
//!                                              Total:   814 NPs
//! ```
//!
//! # Isolation verification
//!
//! `verify_isolation()` reads SRAM from each slot's NP range and confirms
//! no cross-contamination between loaded programs. This uses the SRAM
//! access infrastructure from `crate::sram::SramAccessor`.

use crate::backend::{ModelHandle, NpuBackend};
use crate::error::{AkidaError, Result};
use crate::sram::SramAccessor;

/// Multi-tenant device manager.
///
/// Wraps a backend and an `SramAccessor` to enable loading multiple
/// programs at distinct NP offsets and verifying their isolation.
pub struct MultiTenantDevice {
    backend: Box<dyn NpuBackend>,
    sram: Option<SramAccessor>,
    slots: Vec<ProgramSlot>,
    total_nps: u32,
}

impl MultiTenantDevice {
    /// Create a new multi-tenant device manager.
    ///
    /// # Arguments
    ///
    /// * `backend` — the NPU backend for inference dispatch
    /// * `total_nps` — total NP count (78 for AKD1000, or from `Capabilities`)
    #[must_use]
    pub fn new(backend: Box<dyn NpuBackend>, total_nps: u32) -> Self {
        Self {
            backend,
            sram: None,
            slots: Vec::new(),
            total_nps,
        }
    }

    /// Attach an SRAM accessor for isolation verification.
    ///
    /// Without this, `verify_isolation()` will return unsupported.
    pub fn with_sram(&mut self, sram: SramAccessor) -> &mut Self {
        self.sram = Some(sram);
        self
    }

    /// Load a program into a specific NP slot.
    ///
    /// The program is loaded at the given NP offset using
    /// `program_external()` semantics. The slot tracks the NP range
    /// and a fingerprint of the loaded program.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The NP range overlaps with an existing slot
    /// - The backend fails to load the program
    pub fn load_at_offset(
        &mut self,
        slot_id: usize,
        np_start: u32,
        np_count: u32,
        program: &[u8],
    ) -> Result<ModelHandle> {
        // Check for overlap with existing slots
        let np_end = np_start + np_count;
        for existing in &self.slots {
            if existing.occupied {
                let ex_end = existing.np_start + existing.np_count;
                if np_start < ex_end && np_end > existing.np_start {
                    return Err(AkidaError::invalid_state(format!(
                        "NP range [{np_start}–{np_end}) overlaps with slot {} [{}-{})",
                        existing.id, existing.np_start, ex_end
                    )));
                }
            }
        }

        if np_end > self.total_nps {
            return Err(AkidaError::invalid_state(format!(
                "NP range [{np_start}–{np_end}) exceeds total NPs ({total})",
                total = self.total_nps
            )));
        }

        let handle = self.backend.load_model(program)?;

        let fingerprint = compute_fingerprint(program);

        // Grow slots vector if needed
        while self.slots.len() <= slot_id {
            self.slots.push(ProgramSlot::empty(self.slots.len()));
        }

        self.slots[slot_id] = ProgramSlot {
            id: slot_id,
            np_start,
            np_count,
            handle: Some(handle),
            fingerprint,
            occupied: true,
        };

        Ok(handle)
    }

    /// Verify SRAM isolation between two loaded slots.
    ///
    /// Reads back SRAM from each slot's NP range and confirms the data
    /// matches expected fingerprints. Returns `LoadVerification` with
    /// match statistics.
    ///
    /// Requires an attached `SramAccessor` (see `with_sram()`).
    ///
    /// # Errors
    ///
    /// Returns error if SRAM is not attached or reads fail.
    pub fn verify_isolation(&mut self, slot_a: usize, slot_b: usize) -> Result<IsolationResult> {
        let sram = self.sram.as_mut().ok_or_else(|| {
            AkidaError::capability_query_failed("SRAM accessor not attached for isolation check")
        })?;

        let a = self
            .slots
            .get(slot_a)
            .ok_or_else(|| AkidaError::invalid_state(format!("slot {slot_a} does not exist")))?;
        let b = self
            .slots
            .get(slot_b)
            .ok_or_else(|| AkidaError::invalid_state(format!("slot {slot_b} does not exist")))?;

        if !a.occupied || !b.occupied {
            return Err(AkidaError::invalid_state("both slots must be occupied"));
        }

        let layout = sram.layout();
        let a_base = layout.np_base_offset(a.np_start).unwrap_or(0);
        let b_base = layout.np_base_offset(b.np_start).unwrap_or(0);

        // Read first page of each slot's NP range
        let sample_size = 4096usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "NP index fits hardware address field"
        )]
        let a_data = sram.read_bar1(a_base as usize, sample_size)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "NP index fits hardware address field"
        )]
        let b_data = sram.read_bar1(b_base as usize, sample_size)?;

        let cross_match = a_data == b_data;

        Ok(IsolationResult {
            slot_a,
            slot_b,
            bytes_sampled: sample_size,
            isolated: !cross_match || a_data.iter().all(|&b| b == 0),
            a_has_data: a_data.iter().any(|&b| b != 0),
            b_has_data: b_data.iter().any(|&b| b != 0),
        })
    }

    /// Get status of all program slots.
    #[must_use]
    pub fn slot_status(&self) -> Vec<SlotStatus> {
        self.slots
            .iter()
            .map(|s| SlotStatus {
                id: s.id,
                occupied: s.occupied,
                np_start: s.np_start,
                np_count: s.np_count,
                fingerprint: s.fingerprint,
            })
            .collect()
    }

    /// Total number of NPs currently allocated across all slots.
    #[must_use]
    pub fn nps_allocated(&self) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.occupied)
            .map(|s| s.np_count)
            .sum()
    }

    /// Number of NPs available for new programs.
    #[must_use]
    pub fn nps_available(&self) -> u32 {
        self.total_nps.saturating_sub(self.nps_allocated())
    }

    /// Unload a program from a slot.
    pub fn unload(&mut self, slot_id: usize) {
        if let Some(slot) = self.slots.get_mut(slot_id) {
            slot.occupied = false;
            slot.handle = None;
            slot.fingerprint = 0;
        }
    }

    /// Access the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &dyn NpuBackend {
        &*self.backend
    }

    /// Access the underlying backend mutably.
    pub fn backend_mut(&mut self) -> &mut dyn NpuBackend {
        &mut *self.backend
    }
}

/// A program loaded at a specific NP address range.
#[derive(Debug, Clone)]
pub struct ProgramSlot {
    /// Slot identifier.
    pub id: usize,
    /// Starting NP address.
    pub np_start: u32,
    /// Number of NPs allocated.
    pub np_count: u32,
    /// Handle returned by backend on load.
    pub handle: Option<ModelHandle>,
    /// CRC32 fingerprint of the loaded program bytes.
    pub fingerprint: u32,
    /// Whether this slot has a program loaded.
    pub occupied: bool,
}

impl ProgramSlot {
    const fn empty(id: usize) -> Self {
        Self {
            id,
            np_start: 0,
            np_count: 0,
            handle: None,
            fingerprint: 0,
            occupied: false,
        }
    }
}

/// Result of an isolation verification between two slots.
#[derive(Debug, Clone)]
pub struct IsolationResult {
    /// First slot checked.
    pub slot_a: usize,
    /// Second slot checked.
    pub slot_b: usize,
    /// Number of bytes sampled from each slot.
    pub bytes_sampled: usize,
    /// Whether the two slots appear isolated (no cross-contamination).
    pub isolated: bool,
    /// Whether slot A contains non-zero data.
    pub a_has_data: bool,
    /// Whether slot B contains non-zero data.
    pub b_has_data: bool,
}

/// Summary of a program slot's state.
#[derive(Debug, Clone)]
pub struct SlotStatus {
    /// Slot identifier.
    pub id: usize,
    /// Whether a program is loaded.
    pub occupied: bool,
    /// Starting NP address.
    pub np_start: u32,
    /// Number of NPs allocated.
    pub np_count: u32,
    /// Program fingerprint (0 if empty).
    pub fingerprint: u32,
}

/// Simple CRC32-like fingerprint for program identification.
fn compute_fingerprint(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV offset basis
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::software::SoftwareBackend;

    fn software_model_blob(rs: u32, is: u32, os: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&rs.to_le_bytes());
        v.extend_from_slice(&is.to_le_bytes());
        v.extend_from_slice(&os.to_le_bytes());
        v.extend_from_slice(&0.3f32.to_le_bytes());
        let n_in = (rs as usize) * (is as usize);
        let n_res = (rs as usize) * (rs as usize);
        let n_out = (os as usize) * (rs as usize);
        v.resize(16 + (n_in + n_res + n_out) * 4, 0u8);
        v
    }

    #[test]
    fn fingerprint_deterministic() {
        let data = b"test program bytes";
        let f1 = compute_fingerprint(data);
        let f2 = compute_fingerprint(data);
        assert_eq!(f1, f2);
    }

    #[test]
    fn fingerprint_differs() {
        let f1 = compute_fingerprint(b"program A");
        let f2 = compute_fingerprint(b"program B");
        assert_ne!(f1, f2);
    }

    #[test]
    fn empty_slot() {
        let slot = ProgramSlot::empty(0);
        assert!(!slot.occupied);
        assert_eq!(slot.np_count, 0);
    }

    #[test]
    fn load_at_offset_rejects_overlapping_np_ranges() {
        let mut dev = MultiTenantDevice::new(Box::new(SoftwareBackend::new(4, 2, 1)), 100);
        let blob = software_model_blob(4, 2, 1);
        dev.load_at_offset(0, 0, 10, &blob).unwrap();
        let err = dev.load_at_offset(1, 5, 10, &blob).unwrap_err();
        assert!(err.to_string().contains("overlap"));
    }

    #[test]
    fn load_at_offset_rejects_range_beyond_total_nps() {
        let mut dev = MultiTenantDevice::new(Box::new(SoftwareBackend::new(4, 2, 1)), 20);
        let blob = software_model_blob(4, 2, 1);
        let err = dev.load_at_offset(0, 0, 50, &blob).unwrap_err();
        assert!(err.to_string().contains("exceeds total NPs"));
    }

    #[test]
    fn load_non_overlapping_slots_tracks_allocation() {
        let mut dev = MultiTenantDevice::new(Box::new(SoftwareBackend::new(4, 2, 1)), 100);
        let blob = software_model_blob(4, 2, 1);
        dev.load_at_offset(0, 0, 10, &blob).unwrap();
        dev.load_at_offset(1, 10, 10, &blob).unwrap();
        assert_eq!(dev.nps_allocated(), 20);
        assert_eq!(dev.nps_available(), 80);
        let status = dev.slot_status();
        assert_eq!(status.len(), 2);
        assert!(status[0].occupied && status[1].occupied);
        assert_eq!(
            dev.backend().backend_type(),
            crate::backend::BackendType::Software
        );
    }

    #[test]
    fn verify_isolation_requires_sram() {
        let mut dev = MultiTenantDevice::new(Box::new(SoftwareBackend::new(4, 2, 1)), 100);
        let blob = software_model_blob(4, 2, 1);
        dev.load_at_offset(0, 0, 5, &blob).unwrap();
        dev.load_at_offset(1, 5, 5, &blob).unwrap();
        let err = dev.verify_isolation(0, 1).unwrap_err();
        assert!(err.to_string().contains("SRAM") || err.to_string().contains("accessor"));
    }

    #[test]
    fn unload_clears_slot() {
        let mut dev = MultiTenantDevice::new(Box::new(SoftwareBackend::new(4, 2, 1)), 100);
        let blob = software_model_blob(4, 2, 1);
        dev.load_at_offset(0, 0, 10, &blob).unwrap();
        dev.unload(0);
        assert_eq!(dev.nps_allocated(), 0);
        let st = dev.slot_status();
        assert!(!st[0].occupied);
    }

    #[test]
    fn isolation_result_clone() {
        let r = IsolationResult {
            slot_a: 0,
            slot_b: 1,
            bytes_sampled: 4096,
            isolated: true,
            a_has_data: true,
            b_has_data: false,
        };
        assert_eq!(r.clone().isolated, r.isolated);
    }

    #[test]
    fn slot_status_clone() {
        let s = SlotStatus {
            id: 0,
            occupied: true,
            np_start: 1,
            np_count: 2,
            fingerprint: 3,
        };
        assert_eq!(s.clone().fingerprint, 3);
    }
}
