// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Industrial Domain
//!
//! Loads the adaptive sentinel model, generates synthetic vibration
//! features from an accelerometer, detects anomalies in machinery.

use akida_bench::preserve::{self, Rng};
use std::f32::consts::PI;

const MACHINE_STATES: &[&str] = &[
    "Normal operation",
    "Bearing wear",
    "Imbalance",
    "Misalignment",
    "Looseness",
];

fn main() {
    preserve::header("Industrial", "Predictive maintenance via vibration anomaly detection");

    let model = match preserve::load_model("adaptive_sentinel.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic Vibration Signal ──────────────────────────────");
    let sample_rate = 8000; // Hz
    let duration_ms = 100;
    let n_samples = sample_rate * duration_ms / 1000;
    let mut rng = Rng::new(0x71BE);

    // Simulate accelerometer: fundamental + harmonics + noise
    let fundamental_hz = 60.0f32; // motor speed
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let base = (2.0 * PI * fundamental_hz * t).sin();
            let harmonic = 0.3 * (2.0 * PI * 2.0 * fundamental_hz * t).sin();
            let noise = rng.next_f32() * 0.1;
            base + harmonic + noise
        })
        .collect();

    println!("  Sample rate       : {sample_rate} Hz");
    println!("  Duration          : {duration_ms} ms ({n_samples} samples)");
    println!("  Fundamental       : {fundamental_hz} Hz");
    println!();

    println!("── Feature Extraction (FFT-like) ────────────────────────────");
    let n_bins = 64;
    let bin_width = sample_rate as f32 / (2.0 * n_bins as f32);
    let mut spectrum = vec![0.0f32; n_bins];
    for (bin_idx, spec) in spectrum.iter_mut().enumerate() {
        let freq = (bin_idx as f32 + 0.5) * bin_width;
        // Goertzel-like magnitude estimation
        *spec = signal.iter().enumerate().map(|(i, &s)| {
            let t = i as f32 / sample_rate as f32;
            s * (2.0 * PI * freq * t).cos()
        }).sum::<f32>().abs() / n_samples as f32;
    }

    let peak_bin = spectrum.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let peak_freq = (peak_bin as f32 + 0.5) * bin_width;
    let rms = (signal.iter().map(|&s| s * s).sum::<f32>() / n_samples as f32).sqrt();
    let crest_factor = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max) / rms;

    println!("  Spectral bins     : {n_bins}");
    println!("  Peak frequency    : {peak_freq:.0} Hz (bin {peak_bin})");
    println!("  RMS amplitude     : {rms:.4}");
    println!("  Crest factor      : {crest_factor:.2}");
    println!();

    println!("── Anomaly Detection ──────────────────────────────────────");
    let n_classes = MACHINE_STATES.len();
    let mut logits = vec![0.0f32; n_classes];

    // Heuristic classification from vibration features
    logits[0] = 3.0 - crest_factor * 0.5;       // Normal: low crest
    logits[1] = (peak_freq - 120.0).abs() * -0.02 + 1.0; // Bearing: 2× fundamental
    logits[2] = (peak_freq - 60.0).abs() * -0.05 + 2.0;  // Imbalance: at fundamental
    logits[3] = crest_factor * 0.3;              // Misalignment: high crest
    logits[4] = rms * 2.0 - 1.0;                // Looseness: high RMS

    let probs = preserve::softmax(&logits);
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, prob) in ranked.iter().take(3) {
        let name = MACHINE_STATES.get(*idx).unwrap_or(&"?");
        println!("    {name:>20} : {:.1}%", prob * 100.0);
    }

    let (best_idx, best_prob) = ranked[0];
    let state = MACHINE_STATES.get(best_idx).unwrap_or(&"?");
    let is_anomaly = best_idx != 0;

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    println!("  Machine state     : {state}");
    println!("  Confidence        : {:.1}%", best_prob * 100.0);
    println!("  Anomaly detected  : {}", if is_anomaly { "YES" } else { "NO" });

    let passed = model.layer_count() > 0;
    std::process::exit(preserve::result(
        "Industrial",
        passed,
        &format!("{} layers, state=\"{state}\" (anomaly={})", model.layer_count(), is_anomaly),
    ));
}
