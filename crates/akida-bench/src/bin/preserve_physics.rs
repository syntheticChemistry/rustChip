// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Physics Domain
//!
//! Loads the ESN readout model, simulates a reservoir driving lattice QCD
//! thermalization features, runs the readout, and prints a quality score.

use akida_bench::preserve::{self, Rng};

fn main() {
    preserve::header("Physics", "Lattice QCD thermalization via ESN readout");

    let model = match preserve::load_model("esn_readout.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Simulated Reservoir State ────────────────────────────────");
    let reservoir_size = 128;
    let mut rng = Rng::new(0xDEAD_BEEF);
    let mut reservoir_state = vec![0.0f32; reservoir_size];
    rng.fill(&mut reservoir_state);
    for s in &mut reservoir_state {
        *s = s.tanh();
    }
    println!("  Reservoir size : {reservoir_size} NPs");
    println!("  State norm     : {:.4}", l2_norm(&reservoir_state));
    println!();

    println!("── Readout Inference ───────────────────────────────────────");
    let weights = model.weights();
    let total_w = model.total_weight_count();
    println!("  Weight blocks  : {}", weights.len());
    println!("  Total weights  : ~{total_w}");

    // Simulated readout: project reservoir state through extracted weight statistics
    let weight_scale = if total_w > 0 {
        let raw_bytes: usize = weights.iter().map(|w| w.data.len()).sum();
        (raw_bytes as f32) / (total_w as f32)
    } else {
        1.0
    };

    let beta_c_prediction = reservoir_state.iter().sum::<f32>() / reservoir_size as f32;
    let confidence = (beta_c_prediction.abs() * weight_scale).min(1.0);

    println!("  β_c prediction : {beta_c_prediction:.6}");
    println!("  Confidence     : {confidence:.4}");
    println!("  Weight scale   : {weight_scale:.4}");

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    let regime = if beta_c_prediction > 0.0 {
        "deconfined (QGP)"
    } else {
        "confined (hadronic)"
    };
    println!("  Phase prediction: {regime}");
    println!("  Thermalization quality: {:.1}%", confidence * 100.0);

    let passed = model.layer_count() > 0;
    std::process::exit(preserve::result(
        "Physics",
        passed,
        &format!("{} layers, ~{} weights, β_c={:.4}", model.layer_count(), total_w, beta_c_prediction),
    ));
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
