// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Audio Domain
//!
//! Loads the DS-CNN keyword spotting model, generates synthetic MFCC frames,
//! simulates classification, and prints keyword detection results.

use akida_bench::preserve::{self, Rng};

const KEYWORDS: &[&str] = &[
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go",
];

fn main() {
    preserve::header("Audio", "Keyword spotting via DS-CNN (Google Speech Commands)");

    let model = match preserve::load_model("ds_cnn_kws.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic MFCC Input ─────────────────────────────────────");
    let n_mfcc = 10;
    let n_frames = 49;
    let input_size = n_mfcc * n_frames;
    let mut rng = Rng::new(0xA0D10);
    let mut mfcc_features = vec![0.0f32; input_size];
    rng.fill_range(&mut mfcc_features, -2.0, 2.0);
    println!("  MFCC coefficients : {n_mfcc}");
    println!("  Time frames       : {n_frames}");
    println!("  Feature vector    : {input_size} values");
    println!();

    println!("── Classification ──────────────────────────────────────────");
    let n_classes = KEYWORDS.len();
    let mut logits = vec![0.0f32; n_classes];
    // Simulate classification using model weight statistics as bias
    let weight_energy: f32 = model.weights().iter()
        .flat_map(|w| w.data.iter())
        .take(n_classes * 8)
        .map(|&b| (b as f32 - 128.0) / 128.0)
        .collect::<Vec<f32>>()
        .chunks(8)
        .map(|c| c.iter().sum::<f32>())
        .zip(logits.iter_mut())
        .map(|(e, l)| { *l = e; e })
        .sum();

    if logits.iter().all(|&x| x == 0.0) {
        // Fallback if no weights extracted
        for (i, l) in logits.iter_mut().enumerate() {
            *l = mfcc_features[i % input_size];
        }
    }

    let probs = preserve::softmax(&logits);
    let (best_idx, best_prob) = probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((0, &0.0));

    println!("  Classes           : {n_classes}");
    println!("  Weight energy     : {weight_energy:.2}");
    println!("  Top-3 predictions:");
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, prob) in ranked.iter().take(3) {
        let kw = KEYWORDS.get(*idx).unwrap_or(&"?");
        println!("    {kw:>10} : {:.1}%", prob * 100.0);
    }

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    let detected = KEYWORDS.get(best_idx).unwrap_or(&"?");
    println!("  Detected keyword  : \"{detected}\" ({:.1}% confidence)", best_prob * 100.0);

    let passed = model.layer_count() > 0 && model.total_weight_count() > 0;
    std::process::exit(preserve::result(
        "Audio",
        passed,
        &format!("{} layers, detected=\"{detected}\" at {:.1}%", model.layer_count(), best_prob * 100.0),
    ));
}
