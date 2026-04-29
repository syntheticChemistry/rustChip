// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(any(test, feature = "test-mocks"))]

//! Minimal deterministic NPU mock for integration tests.
//!
//! Unlike [`SoftwareBackend`](crate::backends::software::SoftwareBackend) (full
//! f32 ESN simulation), `SyntheticNpuBackend` is a trivial pass-through that
//! returns input as output. Use it when test code needs *any* backend but
//! doesn't care about numerical correctness — e.g., testing model loading
//! pipelines, backend selection logic, or multi-tenancy slot management.
//!
//! Ported from toadStool's `SyntheticNpuBackend`.

use crate::backend::{BackendType, ModelHandle, NpuBackend};
use crate::capabilities::{
    BatchCapabilities, Capabilities, ChipVersion, PcieConfig, WeightMutationSupport,
};
use crate::error::Result;
use std::sync::atomic::{AtomicU32, Ordering};

/// Deterministic NPU backend for CI and integration tests (no hardware required).
#[derive(Debug)]
pub struct SyntheticNpuBackend {
    caps: Capabilities,
    model_counter: AtomicU32,
}

impl SyntheticNpuBackend {
    /// AKD1000-like capability profile matching toadStool's coverage mock.
    #[must_use]
    pub fn coverage_default() -> Self {
        let caps = Capabilities {
            chip_version: ChipVersion::Akd1000,
            npu_count: 80,
            memory_mb: 10,
            pcie: PcieConfig::new(3, 8),
            power_mw: None,
            temperature_c: None,
            mesh: None,
            clock_mode: None,
            batch: Some(BatchCapabilities {
                max_batch: 8,
                optimal_batch: 8,
                optimal_speedup: 2.35,
            }),
            weight_mutation: WeightMutationSupport::None,
        };
        Self {
            caps,
            model_counter: AtomicU32::new(0),
        }
    }
}

impl NpuBackend for SyntheticNpuBackend {
    fn init(_device_id: &str) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::coverage_default())
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load_model(&mut self, _model: &[u8]) -> Result<ModelHandle> {
        let id = self.model_counter.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(ModelHandle::new(id))
    }

    fn load_reservoir(&mut self, _w_in: &[f32], _w_res: &[f32]) -> Result<()> {
        Ok(())
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        Ok(input.to_vec())
    }

    fn measure_power(&self) -> Result<f32> {
        Ok(1500.0)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Userspace
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_default_has_80_nps() {
        let b = SyntheticNpuBackend::coverage_default();
        assert_eq!(b.caps.npu_count, 80);
    }

    #[test]
    fn init_returns_coverage_default() {
        let b = SyntheticNpuBackend::init("any").unwrap();
        assert!(b.is_ready());
        assert_eq!(b.capabilities().npu_count, 80);
    }

    #[test]
    fn infer_passes_through_input() {
        let mut b = SyntheticNpuBackend::coverage_default();
        let input = vec![1.0, 2.0, 3.0];
        let out = b.infer(&input).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn load_model_increments_handle() {
        let mut b = SyntheticNpuBackend::coverage_default();
        let h1 = b.load_model(&[0u8; 16]).unwrap();
        let h2 = b.load_model(&[0u8; 16]).unwrap();
        assert_eq!(h1.id(), 1);
        assert_eq!(h2.id(), 2);
    }

    #[test]
    fn measure_power_returns_fixed_value() {
        let b = SyntheticNpuBackend::coverage_default();
        assert!((b.measure_power().unwrap() - 1500.0).abs() < f32::EPSILON);
    }
}
