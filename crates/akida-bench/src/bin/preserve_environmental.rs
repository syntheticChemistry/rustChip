// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Environmental Domain
//!
//! Loads the streaming sensor model, generates synthetic multi-channel
//! environmental readings, classifies algal bloom state.

use akida_bench::preserve::{self, Rng};

const BLOOM_STATES: &[&str] = &[
    "Clear",
    "Early bloom",
    "Active bloom",
    "Dense bloom",
    "Post-bloom decay",
];

const SENSOR_NAMES: &[&str] = &[
    "pH", "DO (mg/L)", "Temp (°C)", "Turbidity (NTU)",
    "Chl-a (µg/L)", "Conductivity (µS)", "NO3 (mg/L)", "PO4 (mg/L)",
    "NH4 (mg/L)", "BOD (mg/L)", "TSS (mg/L)", "Flow (m³/s)",
];

fn main() {
    preserve::header("Environmental", "Algal bloom monitoring via streaming sensor network");

    let model = match preserve::load_model("streaming_sensor_12ch.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic Sensor Readings ────────────────────────────────");
    let n_channels = 12;
    let n_timesteps = 10;
    let mut rng = Rng::new(0xB100);

    println!("  Channels    : {n_channels}");
    println!("  Timesteps   : {n_timesteps}");
    println!("  Latest readings:");

    let mut readings = vec![0.0f32; n_channels];
    let ranges: &[(f32, f32)] = &[
        (6.5, 9.0), (2.0, 12.0), (15.0, 30.0), (0.0, 100.0),
        (0.0, 50.0), (100.0, 1000.0), (0.0, 10.0), (0.0, 2.0),
        (0.0, 5.0), (1.0, 20.0), (0.0, 50.0), (0.1, 10.0),
    ];
    for (i, r) in readings.iter_mut().enumerate() {
        let (lo, hi) = ranges[i];
        rng.fill_range(std::slice::from_mut(r), lo, hi);
        println!("    {:>20} : {:.2}", SENSOR_NAMES[i], *r);
    }
    println!();

    // Simulate time series input
    let mut time_series = vec![0.0f32; n_channels * n_timesteps];
    for t in 0..n_timesteps {
        for ch in 0..n_channels {
            let (lo, hi) = ranges[ch];
            rng.fill_range(&mut time_series[t * n_channels + ch..t * n_channels + ch + 1], lo, hi);
        }
    }

    println!("── Classification ──────────────────────────────────────────");
    let n_classes = BLOOM_STATES.len();
    let mut logits = vec![0.0f32; n_classes];

    // Bloom heuristic from sensor readings
    let chl_a = readings[4];
    let turbidity = readings[3];
    logits[0] = 5.0 - chl_a * 0.2;
    logits[1] = chl_a * 0.1;
    logits[2] = (chl_a - 10.0).max(0.0) * 0.15 + turbidity * 0.01;
    logits[3] = (chl_a - 25.0).max(0.0) * 0.2;
    logits[4] = (turbidity - 50.0).max(0.0) * 0.05;

    let probs = preserve::softmax(&logits);
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, prob) in ranked.iter().take(3) {
        let name = BLOOM_STATES.get(*idx).unwrap_or(&"?");
        println!("    {name:>20} : {:.1}%", prob * 100.0);
    }

    let (best_idx, best_prob) = ranked[0];
    let state = BLOOM_STATES.get(best_idx).unwrap_or(&"?");

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    println!("  Bloom state       : {state}");
    println!("  Confidence        : {:.1}%", best_prob * 100.0);
    println!("  Chl-a level       : {chl_a:.1} µg/L");

    let passed = model.layer_count() > 0;
    std::process::exit(preserve::result(
        "Environmental",
        passed,
        &format!("{} layers, state=\"{state}\" (Chl-a={chl_a:.1})", model.layer_count()),
    ));
}
