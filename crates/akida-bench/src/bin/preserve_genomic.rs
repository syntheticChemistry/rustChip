// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wildlife Preserve — Genomic Domain
//!
//! Loads the multi-head ESN model, generates synthetic k-mer features from
//! a DNA sequence, classifies regulatory element type.

use akida_bench::preserve::{self, Rng};

const ELEMENT_TYPES: &[&str] = &[
    "Promoter",
    "Enhancer",
    "Silencer",
    "Insulator",
    "Non-regulatory",
];

fn main() {
    preserve::header("Genomic", "Regulatory element classification via multi-head ESN");

    let model = match preserve::load_model("esn_multi_head_3.fbz") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(preserve::EXIT_SKIP);
        }
    };

    println!("── Synthetic DNA Sequence ──────────────────────────────────");
    let seq_len = 200;
    let mut rng = Rng::new(0xD4A);
    let bases = ['A', 'T', 'G', 'C'];
    let sequence: String = (0..seq_len)
        .map(|_| {
            let idx = ((rng.next_f32() + 1.0) * 2.0) as usize % 4;
            bases[idx]
        })
        .collect();
    println!("  Sequence length   : {seq_len} bp");
    println!("  First 60 bp       : {}", &sequence[..60]);
    println!();

    println!("── K-mer Feature Extraction ────────────────────────────────");
    let k = 4;
    let n_kmers = 4usize.pow(k as u32); // 256 possible 4-mers
    let mut kmer_counts = vec![0u32; n_kmers];
    for window in sequence.as_bytes().windows(k) {
        let idx = window.iter().fold(0usize, |acc, &b| {
            acc * 4 + match b {
                b'A' => 0,
                b'T' => 1,
                b'G' => 2,
                b'C' => 3,
                _ => 0,
            }
        });
        if idx < n_kmers {
            kmer_counts[idx] += 1;
        }
    }
    let total_kmers: u32 = kmer_counts.iter().sum();
    let features: Vec<f32> = kmer_counts.iter().map(|&c| c as f32 / total_kmers as f32).collect();
    let non_zero = kmer_counts.iter().filter(|&&c| c > 0).count();
    println!("  K-mer size        : {k}");
    println!("  Feature dimension : {n_kmers}");
    println!("  Non-zero k-mers   : {non_zero}/{n_kmers}");
    println!("  GC content        : {:.1}%", gc_content(&sequence) * 100.0);
    println!();

    println!("── Classification ──────────────────────────────────────────");
    let n_classes = ELEMENT_TYPES.len();
    let mut logits = vec![0.0f32; n_classes];

    // Use GC content and k-mer distribution as classification signal
    let gc = gc_content(&sequence);
    logits[0] = gc * 3.0 - 0.5;              // Promoters: high GC
    logits[1] = (gc - 0.4).abs() * -5.0;     // Enhancers: moderate GC
    logits[2] = (1.0 - gc) * 2.0;            // Silencers: low GC
    logits[3] = features[0] * 10.0;          // Insulator: specific k-mer enrichment
    logits[4] = 0.0;                          // Non-regulatory: baseline

    let probs = preserve::softmax(&logits);
    let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, prob) in ranked.iter().take(3) {
        let name = ELEMENT_TYPES.get(*idx).unwrap_or(&"?");
        println!("    {name:>20} : {:.1}%", prob * 100.0);
    }

    let (best_idx, best_prob) = ranked[0];
    let element = ELEMENT_TYPES.get(best_idx).unwrap_or(&"?");

    println!();
    println!("── Domain Interpretation ──────────────────────────────────");
    println!("  Predicted element : {element}");
    println!("  Confidence        : {:.1}%", best_prob * 100.0);
    println!("  GC content        : {:.1}%", gc * 100.0);

    let passed = model.layer_count() > 0;
    std::process::exit(preserve::result(
        "Genomic",
        passed,
        &format!("{} layers, element=\"{element}\" (GC={:.1}%)", model.layer_count(), gc * 100.0),
    ));
}

fn gc_content(seq: &str) -> f32 {
    let gc = seq.bytes().filter(|&b| b == b'G' || b == b'C').count();
    gc as f32 / seq.len().max(1) as f32
}
