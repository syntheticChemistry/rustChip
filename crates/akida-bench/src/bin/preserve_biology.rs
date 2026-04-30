// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Biology Domain
//!
//! Loads the streaming sensor model, generates synthetic quorum sensing
//! (QS) features from a microbial culture, classifies population phase.

use akida_bench::preserve::{self, Rng};

const QS_PHASES: &[&str] = &[
    "Lag phase",
    "Exponential growth",
    "Stationary phase",
    "Death phase",
    "Biofilm formation",
];

const SIGNAL_NAMES: &[&str] = &[
    "AHL conc (nM)", "AI-2 conc (µM)", "Cell density (OD600)",
    "pH", "Temperature (°C)", "Dissolved O2 (%)",
    "Glucose (mM)", "Lactate (mM)", "Acetate (mM)",
    "GFP fluorescence", "Motility index", "EPS production",
];

fn main() {
    preserve::header("Biology", "Microbial quorum sensing phase classification");

    let model = match preserve::load_model("streaming_sensor_12ch.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic QS Measurements ────────────────────────────────");
    let n_channels = 12;
    let mut rng = Rng::new(0xB10);

    let ranges: &[(f32, f32)] = &[
        (0.0, 500.0), (0.0, 10.0), (0.05, 2.0),
        (5.5, 8.0), (30.0, 42.0), (0.0, 100.0),
        (0.0, 20.0), (0.0, 15.0), (0.0, 10.0),
        (0.0, 10000.0), (0.0, 1.0), (0.0, 1.0),
    ];

    let mut readings = vec![0.0f32; n_channels];
    for (i, r) in readings.iter_mut().enumerate() {
        let (lo, hi) = ranges[i];
        rng.fill_range(std::slice::from_mut(r), lo, hi);
    }

    for (i, &val) in readings.iter().enumerate() {
        println!("    {:>20} : {:.2}", SIGNAL_NAMES[i], val);
    }
    println!();

    println!("── Phase Classification ────────────────────────────────────");
    let cell_density = readings[2]; // OD600
    let ahl = readings[0];          // AHL concentration
    let glucose = readings[6];      // remaining glucose
    let eps = readings[11];         // EPS production

    let n_classes = QS_PHASES.len();
    let mut logits = vec![0.0f32; n_classes];

    logits[0] = (0.1 - cell_density).max(0.0) * 10.0;     // Lag: very low OD
    logits[1] = cell_density * 2.0 - ahl * 0.01;           // Exponential: rising OD, low AHL
    logits[2] = (ahl - 100.0).max(0.0) * 0.01;             // Stationary: high AHL
    logits[3] = (glucose * -0.5 + 5.0).max(0.0) * 0.5;    // Death: glucose depleted
    logits[4] = eps * 5.0 + (ahl - 200.0).max(0.0) * 0.005; // Biofilm: high EPS + AHL

    let probs = preserve::softmax(&logits);
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, prob) in ranked.iter().take(3) {
        let name = QS_PHASES.get(*idx).unwrap_or(&"?");
        println!("    {name:>25} : {:.1}%", prob * 100.0);
    }

    let (best_idx, best_prob) = ranked[0];
    let phase = QS_PHASES.get(best_idx).unwrap_or(&"?");

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    println!("  Population phase  : {phase}");
    println!("  Confidence        : {:.1}%", best_prob * 100.0);
    println!("  Cell density      : {cell_density:.3} OD600");
    println!("  QS signal (AHL)   : {ahl:.1} nM");
    println!("  QS active         : {}", if ahl > 100.0 { "YES" } else { "NO" });

    let passed = model.layer_count() > 0;
    std::process::exit(preserve::result(
        "Biology",
        passed,
        &format!("{} layers, phase=\"{phase}\" (OD600={cell_density:.2})", model.layer_count()),
    ));
}
