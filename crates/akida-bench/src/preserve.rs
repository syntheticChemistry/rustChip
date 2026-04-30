// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared infrastructure for Wildlife Preserve demo binaries.
//!
//! Each preserve binary loads an `.fbz` model, generates synthetic domain
//! input, runs a software simulation of inference, and prints domain-
//! interpreted results. Exit codes follow ecoPrimals convention:
//! 0 = pass, 1 = fail, 2 = skip.

use akida_models::Model;
use std::path::Path;
use std::time::Instant;

/// Exit code: demo passed.
pub const EXIT_PASS: i32 = 0;
/// Exit code: demo failed.
pub const EXIT_FAIL: i32 = 1;
/// Exit code: demo skipped (model not found).
pub const EXIT_SKIP: i32 = 2;

/// Zoo artifacts directory (relative to workspace root).
pub const ZOO_DIR: &str = "baseCamp/zoo-artifacts";

/// Load a model from the zoo, printing header info.
pub fn load_model(name: &str) -> Result<Model, String> {
    let path_str = format!("{ZOO_DIR}/{name}");
    let path = Path::new(&path_str);
    if !path.exists() {
        return Err(format!("Model not found: {path_str} (run from rustChip root)"));
    }

    let start = Instant::now();
    let model = Model::from_file(&path_str).map_err(|e| format!("Parse failed: {e}"))?;
    let parse_time = start.elapsed();

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    println!("  Model         : {name}");
    println!("  File size     : {file_size} bytes ({:.1} KB)", file_size as f64 / 1024.0);
    println!("  Layers        : {}", model.layer_count());
    println!("  Weight blocks : {}", model.weights().len());
    println!("  Total weights : ~{}", model.total_weight_count());
    println!("  SDK version   : {}", model.version());
    println!("  Parse time    : {parse_time:?}");
    println!();

    Ok(model)
}

/// Simple deterministic PRNG for reproducible synthetic input.
pub struct Rng(u64);

impl Rng {
    /// Create with a seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next f32 in [-1, 1].
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    /// Fill a slice with random f32 in [-1, 1].
    pub fn fill(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.next_f32();
        }
    }

    /// Generate values in `[lo, hi]`.
    pub fn fill_range(&mut self, buf: &mut [f32], lo: f32, hi: f32) {
        for x in buf.iter_mut() {
            let t = (self.next_f32() + 1.0) * 0.5;
            *x = lo + t * (hi - lo);
        }
    }
}

/// Simulated softmax for classification output interpretation.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Simulated FC layer: output = W * input (matmul for weight extraction demo).
pub fn simulated_fc(weights: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|o| {
            let row_start = o * in_dim;
            let row_end = (row_start + in_dim).min(weights.len());
            if row_end <= row_start {
                return 0.0;
            }
            weights[row_start..row_end]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
        })
        .collect()
}

/// Print domain header.
pub fn header(domain: &str, description: &str) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Wildlife Preserve — {domain}");
    println!("  {description}");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
}

/// Print result and return exit code.
pub fn result(domain: &str, passed: bool, detail: &str) -> i32 {
    println!();
    if passed {
        println!("[PASS] {domain}: {detail}");
        EXIT_PASS
    } else {
        println!("[FAIL] {domain}: {detail}");
        EXIT_FAIL
    }
}
