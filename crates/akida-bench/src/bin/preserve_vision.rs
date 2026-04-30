// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Vision Domain
//!
//! Loads the AkidaNet PlantVillage model, generates synthetic image features,
//! simulates classification, and prints plant disease detection results.

use akida_bench::preserve::{self, Rng};

const DISEASES: &[&str] = &[
    "Healthy",
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Cercospora Leaf Spot",
    "Common Rust",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Powdery Mildew",
    "Bacterial Spot",
    "Target Spot",
    "Tomato Yellow Leaf Curl",
    "Mosaic Virus",
    "Septoria Leaf Spot",
];

fn main() {
    preserve::header("Vision", "Plant disease classification via AkidaNet (PlantVillage)");

    let model = match preserve::load_model("akidanet_plantvillage.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic Image Features ────────────────────────────────");
    let spatial = 7;
    let channels = 256;
    let feature_size = spatial * spatial * channels;
    let mut rng = Rng::new(0xF10CA);
    let mut features = vec![0.0f32; feature_size];
    rng.fill_range(&mut features, 0.0, 1.0);
    println!("  Feature map       : {spatial}×{spatial}×{channels}");
    println!("  Feature vector    : {feature_size} values");
    println!();

    println!("── Classification ──────────────────────────────────────────");
    let n_classes = DISEASES.len();
    let mut logits = vec![0.0f32; n_classes];

    // Use model weight statistics to simulate classification
    let weight_bytes: Vec<u8> = model.weights().iter()
        .flat_map(|w| w.data.iter().copied())
        .collect();

    for (i, l) in logits.iter_mut().enumerate() {
        let start = (i * 64) % weight_bytes.len().max(1);
        let end = (start + 64).min(weight_bytes.len());
        *l = weight_bytes[start..end].iter()
            .map(|&b| (b as f32 - 128.0) / 256.0)
            .sum::<f32>();
    }

    let probs = preserve::softmax(&logits);
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Classes           : {n_classes}");
    println!("  Top-3 predictions:");
    for (idx, prob) in ranked.iter().take(3) {
        let name = DISEASES.get(*idx).unwrap_or(&"?");
        println!("    {name:>25} : {:.1}%", prob * 100.0);
    }

    let (best_idx, best_prob) = ranked[0];
    let diagnosis = DISEASES.get(best_idx).unwrap_or(&"?");

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    println!("  Diagnosis         : {diagnosis}");
    println!("  Confidence        : {:.1}%", best_prob * 100.0);
    if best_idx == 0 {
        println!("  Action            : No treatment needed");
    } else {
        println!("  Action            : Recommend inspection for {diagnosis}");
    }

    let passed = model.layer_count() > 0 && model.total_weight_count() > 0;
    std::process::exit(preserve::result(
        "Vision",
        passed,
        &format!("{} layers, diagnosis=\"{diagnosis}\" at {:.1}%", model.layer_count(), best_prob * 100.0),
    ));
}
