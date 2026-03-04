// SPDX-License-Identifier: AGPL-3.0-or-later

//! ESN determinism tests — verify identical results across multiple runs.
//!
//! These tests exercise the SoftwareBackend and HybridEsn to confirm
//! that re-running with the same inputs produces bitwise-identical outputs.

use akida_driver::{NpuBackend, SoftwareBackend, SubstrateSelector};

fn make_test_input(size: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..size)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            #[allow(clippy::cast_precision_loss)]
            let v = (s % 10000) as f32 / 10000.0;
            v * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn test_software_backend_determinism() {
    let rs = 64;
    let is = 4;
    let os = 1;

    let run = || {
        let mut backend = SoftwareBackend::new(rs, is, os);
        let w_in: Vec<f32> = make_test_input(rs * is, 0xAAAA);
        let w_res: Vec<f32> = make_test_input(rs * rs, 0xBBBB);
        let w_out: Vec<f32> = make_test_input(os * rs, 0x1111);
        backend.load_weights(&w_in, &w_res, &w_out).unwrap();

        let mut outputs = Vec::new();
        for step in 0..20 {
            let input = make_test_input(is, step as u64);
            let out = backend.infer(&input).unwrap();
            outputs.extend(out);
        }
        outputs
    };

    let run1 = run();
    let run2 = run();

    assert_eq!(run1.len(), run2.len(), "Output length mismatch");
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert!(
            (a - b).abs() < f32::EPSILON,
            "Determinism violation at index {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_hybrid_esn_determinism() {
    let rs = 32;
    let is = 4;
    let leak_rate = 0.3;
    let w_in: Vec<f32> = make_test_input(rs * is, 0xF1F1);
    let w_res: Vec<f32> = make_test_input(rs * rs, 0xF2F2);
    let w_out: Vec<f32> = make_test_input(rs, 0xF3F3);

    let run = || -> Vec<f32> {
        let esn = akida_driver::HybridEsn::from_weights(&w_in, &w_res, &w_out, leak_rate).unwrap();
        let mut selector = SubstrateSelector::from_esn(esn);
        let mut outputs = Vec::new();
        for step in 0..20 {
            let input = make_test_input(is, step as u64);
            let out = selector.esn_step(&input).unwrap();
            outputs.extend(out);
        }
        outputs
    };

    let run1 = run();
    let run2 = run();

    assert_eq!(run1.len(), run2.len());
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert!(
            (a - b).abs() < f32::EPSILON,
            "HybridEsn determinism violation at index {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_software_backend_non_degenerate() {
    let rs = 64;
    let is = 4;
    let os = 1;
    let mut backend = SoftwareBackend::new(rs, is, os);
    let w_in: Vec<f32> = make_test_input(rs * is, 0xCCCC);
    let w_res: Vec<f32> = make_test_input(rs * rs, 0xDDDD);
    let w_out: Vec<f32> = make_test_input(os * rs, 0xEEEE);
    backend.load_weights(&w_in, &w_res, &w_out).unwrap();

    let mut nonzero_count = 0;
    let mut last_out = vec![0.0f32; 1];

    for step in 0..50 {
        let input = make_test_input(is, step as u64);
        let out = backend.infer(&input).unwrap();
        if out.iter().any(|&x| x.abs() > 1e-6) {
            nonzero_count += 1;
        }
        last_out = out;
    }

    assert!(
        nonzero_count > 10,
        "Reservoir appears degenerate: only {nonzero_count}/50 non-zero outputs"
    );
    assert!(
        last_out.iter().all(|x| x.is_finite()),
        "Output contains NaN or Inf"
    );
}
