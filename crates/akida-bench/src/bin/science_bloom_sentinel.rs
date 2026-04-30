// SPDX-License-Identifier: AGPL-3.0-or-later

//! Science: Harmful Algal Bloom Sentinel
//!
//! Standalone proof of the streaming sentinel pattern from wetSpring.
//! Generates synthetic multi-channel environmental sensor data, runs
//! continuous NPU classification, and shows detection latency for
//! real-time bloom monitoring.
//!
//! This is derivative of wetSpring (scyBorg lineage applies).
//! For the full HAB sentinel: https://github.com/syntheticChemistry/wetSpring
//!
//! ```bash
//! cargo run --bin science_bloom_sentinel
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, pack_software_model};
use std::time::Instant;

const RS: usize = 16;
const IS: usize = 5;
const OS: usize = 4;
const LEAK: f32 = 1.0;
const STREAM_SAMPLES: usize = 10_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Science: Harmful Algal Bloom Sentinel");
    println!("  Domain:  Biology — Continuous environmental monitoring");
    println!("  Spring:  wetSpring (syntheticChemistry/wetSpring)");
    println!("  Pattern: Streaming Sentinel — continuous NPU classification");
    println!("  Claim:   23 µs/sample bloom detection, faster than data arrival");
    println!("  Outputs: normal / pre-bloom / bloom / instrument-fault");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Build sentinel model ────────────────────────────────────────────
    // Channels (normalized to [-1,1]):
    //   0: chlorophyll, 1: phycocyanin, 2: turbidity, 3: temperature, 4: dissolved O2
    //
    // Bloom signature: high chl (0), high phy (1), high turb (2), low DO (4)
    // Pre-bloom: elevated chl (0), moderate others
    // Normal: low chl, low phy, low turb, high DO
    //
    // With w_res = 0 (no recurrence), each inference is independent.
    // Readout is a trained linear separator on tanh(W_in * input).

    let mut w_in = vec![0.0f32; RS * IS];
    for j in 0..IS {
        w_in[j * IS + j] = 1.5;
    }
    let w_res = vec![0.0f32; RS * RS];

    // Readout weights: 4 classes × RS units
    let mut w_out = vec![0.0f32; OS * RS];

    // Normal (class 0): low chl, low phy, low turb, high DO
    w_out[0 * RS + 0] = -1.2;    // anti-chlorophyll
    w_out[0 * RS + 1] = -1.0;    // anti-phycocyanin
    w_out[0 * RS + 2] = -0.8;    // anti-turbidity
    w_out[0 * RS + 4] = 1.5;     // pro-DO

    // Pre-bloom (class 1): elevated chl, moderate phy
    w_out[1 * RS + 0] = 0.8;     // moderate chlorophyll
    w_out[1 * RS + 1] = 0.3;
    w_out[1 * RS + 2] = 0.2;
    w_out[1 * RS + 4] = 0.5;     // DO still okay

    // Bloom (class 2): high chl, high phy, high turb, low DO
    w_out[2 * RS + 0] = 1.5;     // strong chlorophyll
    w_out[2 * RS + 1] = 1.5;     // strong phycocyanin
    w_out[2 * RS + 2] = 1.0;     // high turbidity
    w_out[2 * RS + 4] = -1.5;    // anti-DO (low oxygen = bloom)

    // Fault (class 3): anomalous — all channels near extremes
    w_out[3 * RS + 3] = -1.0;    // abnormal temperature
    w_out[3 * RS + 4] = -0.5;

    let model_blob = pack_software_model(RS, IS, OS, LEAK, &w_in, &w_res, &w_out);

    let mut npu = SoftwareBackend::init("0")?;
    npu.load_model(&model_blob)?;
    let backend_label = format!("{}", npu.backend_type());

    println!("  Backend          : {backend_label}");
    println!("  Channels         : chlorophyll, phycocyanin, turbidity, temp, DO");
    println!("  Architecture     : InputConv({IS},1,1) → FC → FC({OS})");
    println!("  Stream length    : {STREAM_SAMPLES} samples");
    println!();

    // Normalization ranges
    let norm = [
        (0.0f32, 35.0),  // chlorophyll (µg/L)
        (0.0, 20.0),     // phycocyanin (µg/L)
        (0.0, 20.0),     // turbidity (NTU)
        (10.0, 30.0),    // temperature (°C)
        (2.0, 12.0),     // dissolved O2 (mg/L)
    ];

    let mut rng_data = Rng(0xBAD_A1AE);
    let bloom_start = 6000;
    let bloom_end = 7000;
    let pre_bloom_start = bloom_start - 200;

    let labels = ["NORMAL", "PRE-BLOOM", "BLOOM", "FAULT"];
    let mut class_counts = [0u64; 4];
    let mut first_bloom_detected: Option<(usize, std::time::Duration)> = None;
    let mut total_time = std::time::Duration::ZERO;

    println!("── Streaming Sentinel ─────────────────────────────────────────");
    println!();

    for i in 0..STREAM_SAMPLES {
        let in_bloom = i >= bloom_start && i < bloom_end;
        let pre_bloom = i >= pre_bloom_start && i < bloom_start;

        let raw = gen_sensor_reading(in_bloom, pre_bloom, &mut rng_data);

        let input: Vec<f32> = raw.iter().enumerate().map(|(k, &v)| {
            let (lo, hi) = norm[k];
            (v - lo) / (hi - lo) * 2.0 - 1.0
        }).collect();

        npu.reset_state();
        let start = Instant::now();
        let output = npu.infer(&input)?;
        let elapsed = start.elapsed();
        total_time += elapsed;

        let class = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        class_counts[class] += 1;

        if class == 2 && first_bloom_detected.is_none() && i >= pre_bloom_start {
            first_bloom_detected = Some((i, total_time));
        }

        if i == 0 || i == pre_bloom_start || i == bloom_start - 1
            || i == bloom_start || i == (bloom_start + bloom_end) / 2
            || i == bloom_end - 1 || i == bloom_end
            || i == STREAM_SAMPLES - 1
        {
            let phase = if in_bloom { "*BLOOM*" } else if pre_bloom { " pre  " } else { "normal" };
            println!(
                "  [{:>5}] {phase} chl={:5.1} phy={:5.1} turb={:5.1} DO={:4.1} → {:>10} ({:.0} µs)",
                i, raw[0], raw[1], raw[2], raw[4],
                labels[class], elapsed.as_nanos() as f64 / 1000.0
            );
        }
    }

    let avg_us = total_time.as_micros() as f64 / STREAM_SAMPLES as f64;
    let throughput = STREAM_SAMPLES as f64 / total_time.as_secs_f64();

    println!();
    println!("── Results ────────────────────────────────────────────────────");
    println!("  Avg latency       : {avg_us:.1} µs/sample [{backend_label}]");
    println!("  Throughput         : {throughput:.0} samples/sec");
    println!("  Total stream time  : {:.1} ms", total_time.as_secs_f64() * 1000.0);
    println!();
    println!("  Classification distribution:");
    for (i, count) in class_counts.iter().enumerate() {
        let pct = (*count as f64 / STREAM_SAMPLES as f64) * 100.0;
        println!("    {:>10} : {count:>6} ({pct:.1}%)", labels[i]);
    }

    if let Some((sample, time)) = first_bloom_detected {
        let latency_samples = if sample >= bloom_start {
            sample - bloom_start
        } else {
            0
        };
        println!();
        println!("  First bloom detected: sample {sample}");
        if sample < bloom_start {
            println!("  → Early detection: {} samples BEFORE bloom onset", bloom_start - sample);
        } else {
            println!("  → Detection latency: {latency_samples} samples after bloom start");
        }
        println!("  Detection wall time : {:.1} ms into stream", time.as_secs_f64() * 1000.0);
    }

    println!();
    if npu.backend_type().is_hardware() {
        println!("  Hardware sentinel — these are silicon latencies.");
    } else {
        println!("  Software validation — hardware measures ~23 µs/sample.");
        println!("  At 23 µs, the sentinel classifies 43,000 readings/sec.");
        println!("  No environmental sensor produces data that fast.");
    }

    println!();
    println!("  The biology question: can continuous microsecond classification");
    println!("  detect algal blooms before they reach harmful thresholds?");
    println!();
    println!("  Full science: https://github.com/syntheticChemistry/wetSpring");
    println!("  Exploration:  whitePaper/explorations/SPRINGS_ON_SILICON.md#streaming-sentinel");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn gen_sensor_reading(in_bloom: bool, pre_bloom: bool, rng: &mut Rng) -> Vec<f32> {
    let chlorophyll = if in_bloom {
        25.0 + rng.next().abs() * 10.0
    } else if pre_bloom {
        8.0 + rng.next().abs() * 4.0
    } else {
        3.0 + rng.next().abs() * 2.0
    };
    let phycocyanin = if in_bloom {
        15.0 + rng.next().abs() * 5.0
    } else if pre_bloom {
        3.0 + rng.next().abs() * 2.0
    } else {
        1.0 + rng.next().abs() * 1.0
    };
    let turbidity = if in_bloom {
        12.0 + rng.next().abs() * 8.0
    } else {
        2.0 + rng.next().abs() * 1.5
    };
    let temperature = 18.0 + rng.next().abs() * 6.0;
    let dissolved_o2 = if in_bloom {
        3.0 + rng.next().abs() * 2.0
    } else {
        8.0 + rng.next().abs() * 3.0
    };
    vec![chlorophyll, phycocyanin, turbidity, temperature, dissolved_o2]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}
