// SPDX-License-Identifier: AGPL-3.0-or-later

//! Science: Crop Stress Classifier with Online Adaptation
//!
//! Standalone proof of the online adaptation pattern from airSpring.
//! Generates synthetic multi-sensor crop stress readings, runs streaming
//! NPU classification, and demonstrates (1+1)-ES seasonal weight evolution
//! directly on NPU SRAM.
//!
//! This is derivative of airSpring (scyBorg lineage applies).
//! For the full agricultural IoT pipeline: https://github.com/syntheticChemistry/airSpring
//!
//! ```bash
//! cargo run --bin science_crop_classifier
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, pack_software_model};
use std::time::Instant;

const RS: usize = 16;
const IS: usize = 4;
const OS: usize = 4;
const LEAK: f32 = 1.0;
const SAMPLES_PER_SEASON: usize = 500;
const N_SEASONS: usize = 4;
const MUTATION_SIGMA: f32 = 0.15;
const EVOLUTION_INTERVAL: usize = 50;
const VALIDATION_BATCH: usize = 30;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Science: Crop Stress Classifier with Online Adaptation");
    println!("  Domain:  Agriculture — Edge IoT sensor classification");
    println!("  Spring:  airSpring (syntheticChemistry/airSpring)");
    println!("  Pattern: Online Adaptation — (1+1)-ES weight evolution");
    println!("  Claim:   Edge NPU with seasonal weight evolution via SRAM");
    println!("  Outputs: healthy / water-stress / heat-stress / nutrient-deficient");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let season_names = ["Spring", "Summer", "Autumn", "Winter"];
    let _class_labels = ["healthy", "water-stress", "heat-stress", "nutrient-def"];

    // ── Build initial model ─────────────────────────────────────────────
    // Sensors (normalized to [-1,1]):
    //   0: soil moisture, 1: leaf temperature, 2: NDVI, 3: humidity
    //
    // Class signatures:
    //   healthy:      high moisture(0), moderate temp(1), high NDVI(2), high humidity(3)
    //   water-stress: low moisture(0), high temp(1), low NDVI(2), low humidity(3)
    //   heat-stress:  moderate moisture(0), very high temp(1), low NDVI(2)
    //   nutrient-def: moderate moisture(0), low temp(1), very low NDVI(2)

    let mut w_in = vec![0.0f32; RS * IS];
    for j in 0..IS {
        w_in[j * IS + j] = 1.5;
    }
    let w_res = vec![0.0f32; RS * RS];

    let mut w_out = vec![0.0f32; OS * RS];

    // Weights partially tuned for spring. Random perturbation simulates a
    // model trained on limited data — good but not perfect. Seasonal drift
    // degrades accuracy further, motivating online (1+1)-ES adaptation.
    let mut rng_w = Rng(0xCB0F_5EED);

    w_out[0 * RS + 0] = 1.0 + rng_w.next() * 0.5;
    w_out[0 * RS + 1] = -0.3 + rng_w.next() * 0.3;
    w_out[0 * RS + 2] = 1.2 + rng_w.next() * 0.5;
    w_out[0 * RS + 3] = 0.6 + rng_w.next() * 0.3;

    w_out[1 * RS + 0] = -1.0 + rng_w.next() * 0.5;
    w_out[1 * RS + 1] = 0.7 + rng_w.next() * 0.3;
    w_out[1 * RS + 2] = -0.6 + rng_w.next() * 0.3;
    w_out[1 * RS + 3] = -0.8 + rng_w.next() * 0.3;

    w_out[2 * RS + 0] = rng_w.next() * 0.3;
    w_out[2 * RS + 1] = 1.3 + rng_w.next() * 0.5;
    w_out[2 * RS + 2] = -0.7 + rng_w.next() * 0.3;
    w_out[2 * RS + 3] = rng_w.next() * 0.3;

    w_out[3 * RS + 0] = 0.2 + rng_w.next() * 0.2;
    w_out[3 * RS + 1] = -0.8 + rng_w.next() * 0.3;
    w_out[3 * RS + 2] = -1.3 + rng_w.next() * 0.5;
    w_out[3 * RS + 3] = rng_w.next() * 0.2;

    let norm = [
        (0.05f32, 0.5),   // soil moisture
        (15.0, 45.0),      // leaf temperature
        (0.1, 0.9),        // NDVI
        (0.15, 0.85),      // humidity
    ];

    let model_blob = pack_software_model(RS, IS, OS, LEAK, &w_in, &w_res, &w_out);
    let mut npu = SoftwareBackend::init("0")?;
    npu.load_model(&model_blob)?;
    let backend_label = format!("{}", npu.backend_type());

    println!("  Backend        : {backend_label}");
    println!("  Sensors        : soil_moisture, leaf_temp, NDVI, humidity");
    println!("  Architecture   : InputConv({IS},1,1) → FC → FC({OS})");
    println!("  Seasons        : {N_SEASONS} × {SAMPLES_PER_SEASON} samples");
    println!("  Mutation σ     : {MUTATION_SIGMA}");
    println!("  Evolution every: {EVOLUTION_INTERVAL} samples (validation batch: {VALIDATION_BATCH})");
    println!();

    let mut rng_data = Rng(0xF1E1_D00D);
    let mut total_npu_time = std::time::Duration::ZERO;
    let mut total_samples = 0u64;
    let mut evolutions_accepted = 0u64;
    let mut evolutions_total = 0u64;

    for season in 0..N_SEASONS {
        println!("── Season: {} ────────────────────────────────────────────", season_names[season]);

        let stress_bias: [f32; 4] = match season {
            0 => [0.60, 0.10, 0.10, 0.20],
            1 => [0.25, 0.35, 0.30, 0.10],
            2 => [0.45, 0.10, 0.10, 0.35],
            3 => [0.65, 0.05, 0.20, 0.10],
            _ => unreachable!(),
        };

        let mut season_correct = 0u64;
        let season_start = Instant::now();

        for i in 0..SAMPLES_PER_SEASON {
            let gt_class = sample_class(&stress_bias, &mut rng_data);
            let raw = gen_crop_reading(gt_class, season, &mut rng_data);

            let input: Vec<f32> = raw.iter().enumerate().map(|(k, &v)| {
                let (lo, hi) = norm[k];
                (v - lo) / (hi - lo) * 2.0 - 1.0
            }).collect();

            npu.reset_state();
            let start = Instant::now();
            let output = npu.infer(&input)?;
            total_npu_time += start.elapsed();
            total_samples += 1;

            let pred_class = argmax(&output);
            if pred_class == gt_class { season_correct += 1; }

            // ── (1+1)-ES evolution ──────────────────────────────────
            if i > 0 && i % EVOLUTION_INTERVAL == 0 {
                evolutions_total += 1;

                let mut w_out_mutant = w_out.clone();
                for w in &mut w_out_mutant {
                    *w += rng_data.next() * MUTATION_SIGMA;
                }

                let mutant_blob = pack_software_model(
                    RS, IS, OS, LEAK, &w_in, &w_res, &w_out_mutant,
                );
                let mut mutant_npu = SoftwareBackend::init("0")?;
                mutant_npu.load_model(&mutant_blob)?;

                let mut parent_score = 0i32;
                let mut mutant_score = 0i32;

                for _ in 0..VALIDATION_BATCH {
                    let val_class = sample_class(&stress_bias, &mut rng_data);
                    let val_raw = gen_crop_reading(val_class, season, &mut rng_data);
                    let val_input: Vec<f32> = val_raw.iter().enumerate().map(|(k, &v)| {
                        let (lo, hi) = norm[k];
                        (v - lo) / (hi - lo) * 2.0 - 1.0
                    }).collect();

                    npu.reset_state();
                    let p_out = npu.infer(&val_input)?;
                    mutant_npu.reset_state();
                    let m_out = mutant_npu.infer(&val_input)?;

                    if argmax(&p_out) == val_class { parent_score += 1; }
                    if argmax(&m_out) == val_class { mutant_score += 1; }
                }

                if mutant_score > parent_score {
                    w_out.clone_from(&w_out_mutant);
                    let new_blob = pack_software_model(
                        RS, IS, OS, LEAK, &w_in, &w_res, &w_out,
                    );
                    npu.load_model(&new_blob)?;
                    evolutions_accepted += 1;
                }
            }
        }

        let season_elapsed = season_start.elapsed();
        let season_accuracy = season_correct as f64 / SAMPLES_PER_SEASON as f64 * 100.0;

        println!("    Samples     : {SAMPLES_PER_SEASON}");
        println!("    Accuracy    : {season_accuracy:.1}%");
        println!("    Stress bias : healthy={:.0}% water={:.0}% heat={:.0}% nutr={:.0}%",
            stress_bias[0] * 100.0, stress_bias[1] * 100.0,
            stress_bias[2] * 100.0, stress_bias[3] * 100.0);
        println!("    Season time : {:.1} ms", season_elapsed.as_secs_f64() * 1000.0);
        println!();
    }

    let avg_us = total_npu_time.as_micros() as f64 / total_samples as f64;
    let throughput = total_samples as f64 / total_npu_time.as_secs_f64();

    println!("── Results ────────────────────────────────────────────────────");
    println!("  Total samples      : {total_samples}");
    println!("  Avg NPU latency    : {avg_us:.1} µs/sample [{backend_label}]");
    println!("  Throughput          : {throughput:.0} samples/sec");
    println!();
    println!("  Online Adaptation ((1+1)-ES):");
    println!("    Evolution trials  : {evolutions_total}");
    println!("    Accepted mutants  : {evolutions_accepted}");
    let accept_pct = if evolutions_total > 0 {
        evolutions_accepted as f64 / evolutions_total as f64 * 100.0
    } else { 0.0 };
    println!("    Acceptance rate   : {accept_pct:.1}%");

    println!();
    if npu.backend_type().is_hardware() {
        println!("  Hardware edge NPU — SRAM weight mutation in microseconds.");
    } else {
        println!("  Software validation — hardware achieves ~48 µs/sample.");
        println!("  On AKD1000, weight mutation is a direct SRAM write (µs, not ms).");
        println!("  airSpring deploys seasonal (1+1)-ES on real agricultural data.");
    }

    println!();
    println!("  The agriculture question: can an edge NPU adapt to seasonal");
    println!("  drift via online evolution, maintaining accuracy without");
    println!("  retraining from scratch? airSpring proved it can.");
    println!();
    println!("  Full science: https://github.com/syntheticChemistry/airSpring");
    println!("  Exploration:  whitePaper/explorations/SPRINGS_ON_SILICON.md#online-adaptation");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn sample_class(bias: &[f32; 4], rng: &mut Rng) -> usize {
    let roll = (rng.next() + 1.0) / 2.0;
    let mut acc = 0.0;
    for (i, &b) in bias.iter().enumerate() {
        acc += b;
        if roll < acc { return i; }
    }
    3
}

fn gen_crop_reading(class: usize, season: usize, rng: &mut Rng) -> Vec<f32> {
    // Seasonal shifts move class boundaries. The model is tuned for spring;
    // summer drift degrades accuracy, motivating (1+1)-ES adaptation.
    let temp_shift = [0.0f32, 5.0, 2.0, -2.5][season];
    let moist_shift = [0.0f32, -0.06, -0.02, 0.02][season];

    // Sensor noise is bidirectional — class boundaries overlap at edges.
    // This is realistic: soil moisture 0.25 could be mildly dry (healthy)
    // or moderately recovered (water-stress).
    match class {
        0 => vec![
            (0.33 + moist_shift + rng.next() * 0.08).clamp(0.05, 0.5),
            23.0 + temp_shift + rng.next() * 4.0,
            0.65 + rng.next() * 0.12,
            0.58 + rng.next() * 0.12,
        ],
        1 => vec![
            (0.13 + moist_shift + rng.next() * 0.08).clamp(0.05, 0.5),
            32.0 + temp_shift + rng.next() * 5.0,
            0.30 + rng.next() * 0.12,
            0.23 + rng.next() * 0.10,
        ],
        2 => vec![
            (0.22 + moist_shift + rng.next() * 0.08).clamp(0.05, 0.5),
            37.0 + temp_shift + rng.next() * 5.0,
            0.27 + rng.next() * 0.12,
            0.40 + rng.next() * 0.12,
        ],
        3 => vec![
            (0.25 + moist_shift + rng.next() * 0.08).clamp(0.05, 0.5),
            20.0 + temp_shift + rng.next() * 4.0,
            0.18 + rng.next() * 0.10,
            0.44 + rng.next() * 0.12,
        ],
        _ => unreachable!(),
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}
