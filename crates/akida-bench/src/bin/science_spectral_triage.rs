// SPDX-License-Identifier: AGPL-3.0-or-later

//! Science: LC-MS Spectral Triage — NPU Gatekeeper
//!
//! Standalone proof of the microsecond gatekeeper pattern from wetSpring.
//! Generates synthetic LC-MS-like spectral peaks, runs NPU triage to
//! decide which peaks need expensive library search. Shows the throughput
//! multiplier from cheap NPU pre-filtering.
//!
//! This is derivative of wetSpring (scyBorg lineage applies).
//! For the full spectral pipeline: https://github.com/syntheticChemistry/wetSpring
//!
//! ```bash
//! cargo run --bin science_spectral_triage
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, pack_software_model};
use std::time::Instant;

const RS: usize = 16;
const IS: usize = 8;
const OS: usize = 2;
const LEAK: f32 = 1.0;
const N_SPECTRA: usize = 50_000;
const LIBRARY_SEARCH_MS: f64 = 0.3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Science: LC-MS Spectral Triage — NPU Gatekeeper");
    println!("  Domain:  Chemistry — Mass spectrometry peak classification");
    println!("  Spring:  wetSpring (syntheticChemistry/wetSpring)");
    println!("  Pattern: Microsecond Gatekeeper — NPU pre-filter");
    println!("  Claim:   14x throughput via background rejection");
    println!("  Outputs: interesting (needs library search) / background (skip)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Build triage model ──────────────────────────────────────────────
    // Readout weights trained on synthetic spectral feature distribution.
    // In production, wetSpring trains on real LC-MS data; here we embed
    // a pre-computed linear separator for the known synthetic distribution.
    //
    // With w_res = 0, the ESN reduces to: output = W_out * tanh(W_in * input)
    // This is a single-layer nonlinear classifier — sufficient for triage.
    //
    // Feature order (normalized to [-1,1]):
    //   0: m/z centroid, 1: intensity, 2: RT, 3: peak width,
    //   4: isotope spacing, 5: charge state, 6: S/N ratio, 7: shape score
    //
    // Discriminating features: intensity (1), S/N (6), shape (7), width (3)
    // Interesting peaks: high intensity, high S/N, high shape, narrow width

    // W_in: identity mapping — feature j goes to reservoir unit j
    let mut w_in = vec![0.0f32; RS * IS];
    for j in 0..IS {
        w_in[j * IS + j] = 1.0;
    }
    let w_res = vec![0.0f32; RS * RS];

    // W_out: class 0 = "interesting", class 1 = "background"
    let mut w_out = vec![0.0f32; OS * RS];
    // "interesting" channel responds positively to high intensity, S/N, shape
    w_out[0 * RS + 1] = 1.0;     // intensity
    w_out[0 * RS + 6] = 1.5;     // S/N (strongest signal)
    w_out[0 * RS + 7] = 0.8;     // shape score
    w_out[0 * RS + 3] = -0.5;    // narrow width → interesting
    // "background" channel: anti-correlated
    w_out[1 * RS + 1] = -0.8;
    w_out[1 * RS + 6] = -1.2;
    w_out[1 * RS + 7] = -0.6;
    w_out[1 * RS + 3] = 0.4;

    let model_blob = pack_software_model(RS, IS, OS, LEAK, &w_in, &w_res, &w_out);

    let mut npu = SoftwareBackend::init("0")?;
    npu.load_model(&model_blob)?;
    let backend_label = format!("{}", npu.backend_type());

    println!("  Backend          : {backend_label}");
    println!("  Features         : m/z, intensity, RT, width, isotope, charge, S/N, shape");
    println!("  Architecture     : InputConv({IS},1,1) → FC → FC({OS})");
    println!("  Spectra          : {N_SPECTRA}");
    println!("  Library cost     : {LIBRARY_SEARCH_MS} ms/peak (simulated)");
    println!();

    // Normalization ranges
    let norm = [
        (100.0f32, 1000.0),  // m/z
        (0.0, 20000.0),      // intensity
        (2.0, 30.0),         // RT
        (0.0, 0.6),          // width
        (0.0, 2.0),          // isotope spacing
        (1.0, 3.0),          // charge
        (0.0, 65.0),         // S/N
        (0.0, 1.0),          // shape score
    ];

    let mut rng_data = Rng(0xDA55_5AEC);
    let interesting_rate = 0.042f32;

    let mut triage_interesting = 0u64;
    let mut triage_background = 0u64;
    let mut ground_truth_interesting = 0u64;
    let mut true_positive = 0u64;
    let mut false_negative = 0u64;
    let mut npu_time = std::time::Duration::ZERO;

    println!("── Spectral Triage ────────────────────────────────────────────");
    println!();

    for i in 0..N_SPECTRA {
        let is_interesting = {
            let roll = (rng_data.next() + 1.0) / 2.0;
            roll < interesting_rate
        };
        if is_interesting {
            ground_truth_interesting += 1;
        }

        let raw = gen_spectral_features(is_interesting, &mut rng_data);

        let input: Vec<f32> = raw.iter().enumerate().map(|(k, &v)| {
            let (lo, hi) = norm[k];
            (v - lo) / (hi - lo) * 2.0 - 1.0
        }).collect();

        npu.reset_state();
        let start = Instant::now();
        let output = npu.infer(&input)?;
        npu_time += start.elapsed();

        let triaged_as_interesting = output[0] > output[1];

        if triaged_as_interesting {
            triage_interesting += 1;
            if is_interesting { true_positive += 1; }
        } else {
            triage_background += 1;
            if is_interesting { false_negative += 1; }
        }

        if i < 5 || i == N_SPECTRA / 4 || i == N_SPECTRA / 2
            || i == 3 * N_SPECTRA / 4 || i == N_SPECTRA - 1
        {
            let label = if triaged_as_interesting { "→ LIBRARY" } else { "  skip" };
            let truth = if is_interesting { " (HIT)" } else { "" };
            println!(
                "  [{:>6}] m/z={:6.1} int={:8.0} S/N={:5.1} {label}{truth}",
                i, raw[0], raw[1], raw[6]
            );
        }
    }

    let avg_npu_us = npu_time.as_micros() as f64 / N_SPECTRA as f64;
    let npu_throughput = N_SPECTRA as f64 / npu_time.as_secs_f64();
    let background_pct = triage_background as f64 / N_SPECTRA as f64 * 100.0;

    let time_without_triage_s = N_SPECTRA as f64 * LIBRARY_SEARCH_MS / 1000.0;
    let time_with_triage_s = npu_time.as_secs_f64()
        + triage_interesting as f64 * LIBRARY_SEARCH_MS / 1000.0;
    let speedup = time_without_triage_s / time_with_triage_s;

    let throughput_without = N_SPECTRA as f64 / time_without_triage_s;
    let throughput_with = N_SPECTRA as f64 / time_with_triage_s;

    let sensitivity = if ground_truth_interesting > 0 {
        true_positive as f64 / ground_truth_interesting as f64 * 100.0
    } else { 0.0 };

    println!();
    println!("── Results ────────────────────────────────────────────────────");
    println!("  NPU triage");
    println!("    Avg latency     : {avg_npu_us:.1} µs/spectrum [{backend_label}]");
    println!("    NPU throughput  : {npu_throughput:.0} spectra/sec");
    println!("    Background      : {triage_background} / {N_SPECTRA} ({background_pct:.1}%)");
    println!("    Sent to library : {triage_interesting}");
    println!();
    println!("  Accuracy (on synthetic data)");
    println!("    Ground truth interesting : {ground_truth_interesting}");
    println!("    True positives           : {true_positive}");
    println!("    False negatives (missed) : {false_negative}");
    println!("    Sensitivity              : {sensitivity:.1}%");
    println!();
    println!("  Pipeline throughput");
    println!("    Without triage  : {throughput_without:.0} spectra/sec ({time_without_triage_s:.1}s)");
    println!("    With NPU triage : {throughput_with:.0} spectra/sec ({time_with_triage_s:.1}s)");
    println!("    Speedup         : {speedup:.1}x");

    println!();
    if npu.backend_type().is_hardware() {
        println!("  Hardware gatekeeper — silicon measurements.");
    } else {
        println!("  Software validation — hardware achieves ~45k spectra/s triage.");
        println!("  wetSpring measured 14x throughput from 95.8% background rejection");
        println!("  on real LC-MS data with trained weights.");
    }

    println!();
    println!("  The chemistry question: can a microsecond NPU decision eliminate");
    println!("  most expensive library searches and multiply throughput?");
    println!("  wetSpring proved it can.");
    println!();
    println!("  Full science: https://github.com/syntheticChemistry/wetSpring");
    println!("  Exploration:  whitePaper/explorations/SPRINGS_ON_SILICON.md#microsecond-gatekeeper");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn gen_spectral_features(interesting: bool, rng: &mut Rng) -> Vec<f32> {
    let mz = 100.0 + rng.next().abs() * 900.0;
    let intensity = if interesting {
        5000.0 + rng.next().abs() * 15000.0
    } else {
        50.0 + rng.next().abs() * 500.0
    };
    let rt = 2.0 + rng.next().abs() * 28.0;
    let width = if interesting {
        0.02 + rng.next().abs() * 0.05
    } else {
        0.1 + rng.next().abs() * 0.5
    };
    let isotope_spacing = if interesting {
        1.003 + rng.next() * 0.002
    } else {
        rng.next().abs() * 2.0
    };
    let charge = if interesting { 2.0 } else { 1.0 + rng.next().abs() };
    let sn_ratio = if interesting {
        15.0 + rng.next().abs() * 50.0
    } else {
        1.0 + rng.next().abs() * 5.0
    };
    let shape_score = if interesting {
        0.8 + rng.next().abs() * 0.2
    } else {
        0.1 + rng.next().abs() * 0.6
    };
    vec![mz, intensity, rt, width, isotope_spacing, charge, sn_ratio, shape_score]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}
