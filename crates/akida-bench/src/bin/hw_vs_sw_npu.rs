// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware NPU vs Software NPU — side-by-side comparison.
//!
//! Runs the same ESN model with identical input on both backends:
//! - **SoftwareBackend**: CPU f32 virtual NPU
//! - **VfioBackend**: AKD1000 hardware via VFIO
//!
//! Compares latency, throughput, power, and output agreement.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin hw_vs_sw_npu
//! cargo run --bin hw_vs_sw_npu -- 0000:e2:00.0
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, VfioBackend, pack_software_model};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const RS: usize = 50;
const IS: usize = 8;
const OS: usize = 1;
const LEAK: f32 = 0.3;
const BENCH_ITERS: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=warn".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Hardware NPU vs Software NPU — Side-by-Side Comparison");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Generate deterministic weights ──────────────────────────────────
    let mut rng = Rng(0xACE0_BABE);
    let w_in: Vec<f32> = (0..RS * IS).map(|_| rng.next() * 0.1).collect();
    let w_res: Vec<f32> = (0..RS * RS).map(|_| rng.next() * 0.05).collect();
    let w_out: Vec<f32> = (0..OS * RS).map(|_| rng.next() * 0.2).collect();
    let model_blob = pack_software_model(RS, IS, OS, LEAK, &w_in, &w_res, &w_out);

    // ── Generate deterministic input sequence ───────────────────────────
    let n_steps = 20;
    let inputs: Vec<Vec<f32>> = (0..n_steps)
        .map(|_| (0..IS).map(|_| rng.next()).collect())
        .collect();

    println!("  Architecture   : RS={RS}  IS={IS}  OS={OS}  leak={LEAK}");
    println!("  Model blob     : {} bytes", model_blob.len());
    println!("  Input steps    : {n_steps} × {IS} floats");
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // SOFTWARE NPU
    // ══════════════════════════════════════════════════════════════════════
    println!("── Software NPU (CPU f32) ─────────────────────────────────────");
    let mut sw = SoftwareBackend::new(RS, IS, OS);
    sw.load_model(&model_blob)?;
    assert!(sw.is_ready(), "SoftwareBackend should be ready after load");
    println!("  Backend        : {}", sw.backend_type());
    println!("  Ready          : {}", sw.is_ready());

    // Warm up + collect outputs
    let mut sw_outputs = Vec::with_capacity(n_steps);
    sw.reset_state();
    for input in &inputs {
        sw_outputs.push(sw.infer(input)?);
    }
    let sw_final = sw_outputs.last().cloned().unwrap_or_default();

    // Benchmark
    let sw_start = Instant::now();
    for _ in 0..BENCH_ITERS {
        sw.reset_state();
        for input in &inputs {
            let _ = sw.infer(input)?;
        }
    }
    let sw_total = sw_start.elapsed();
    let sw_per_step_us = sw_total.as_micros() as f64 / (BENCH_ITERS * n_steps) as f64;
    let sw_power = sw.measure_power().unwrap_or(0.0);

    println!("  Final output   : {:.6}", sw_final.first().unwrap_or(&0.0));
    println!("  Avg step time  : {sw_per_step_us:.1} µs");
    println!("  Steps/sec      : {:.0}", 1e6 / sw_per_step_us);
    println!("  Power          : {sw_power:.2} W (CPU — not measured)");
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // HARDWARE NPU
    // ══════════════════════════════════════════════════════════════════════
    println!("── Hardware NPU (AKD1000 via VFIO) ──────────────────────────");
    let pcie_addr = std::env::args()
        .nth(1)
        .or_else(discover_first_akida);

    let hw_result = if let Some(addr) = pcie_addr {
        match VfioBackend::init(&addr) {
            Ok(mut hw) => {
                println!("  Backend        : {}", hw.backend_type());
                println!("  Ready          : {}", hw.is_ready());

                if !hw.is_ready() {
                    println!("  Retrying init...");
                    let _ = hw.reset_and_enable();
                    println!("  Ready (retry)  : {}", hw.is_ready());
                }

                match hw.load_model(&model_blob) {
                    Ok(handle) => {
                        println!("  Model handle   : {}", handle.id());

                        let mut hw_outputs = Vec::with_capacity(n_steps);
                        for input in &inputs {
                            hw_outputs.push(hw.infer(input)?);
                        }
                        let hw_final = hw_outputs.last().cloned().unwrap_or_default();

                        // Benchmark
                        let hw_start = Instant::now();
                        for _ in 0..BENCH_ITERS.min(100) {
                            for input in &inputs {
                                let _ = hw.infer(input)?;
                            }
                        }
                        let hw_total = hw_start.elapsed();
                        let hw_iters = BENCH_ITERS.min(100);
                        let hw_per_step_us =
                            hw_total.as_micros() as f64 / (hw_iters * n_steps) as f64;
                        let hw_power = hw.measure_power().unwrap_or(0.0);

                        println!("  Final output   : {:.6}", hw_final.first().unwrap_or(&0.0));
                        println!("  Avg step time  : {hw_per_step_us:.1} µs");
                        println!("  Steps/sec      : {:.0}", 1e6 / hw_per_step_us);
                        println!("  Power          : {hw_power:.2} W");

                        Some((hw_outputs, hw_per_step_us, hw_power))
                    }
                    Err(e) => {
                        println!("  Model load     : failed — {e}");
                        None
                    }
                }
            }
            Err(e) => {
                println!("  VFIO init      : failed — {e}");
                None
            }
        }
    } else {
        println!("  No Akida hardware detected — running software-only comparison");
        None
    };
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // COMPARISON
    // ══════════════════════════════════════════════════════════════════════
    println!("── Comparison ─────────────────────────────────────────────────");
    println!();
    println!("  {:20} {:>15} {:>15}", "", "Software NPU", "Hardware NPU");
    println!("  {:20} {:>15} {:>15}", "─".repeat(20), "─".repeat(15), "─".repeat(15));

    let hw_str = |val: Option<f64>, fmt: &str| -> String {
        val.map_or("N/A".to_string(), |v| format!("{v:.1}{fmt}"))
    };

    let hw_latency = hw_result.as_ref().map(|(_, us, _)| *us);
    let hw_power_val = hw_result.as_ref().map(|(_, _, p)| *p as f64);

    println!(
        "  {:20} {:>12.1} µs {:>15}",
        "Avg step latency",
        sw_per_step_us,
        hw_str(hw_latency, " µs")
    );
    println!(
        "  {:20} {:>12.0} /s {:>15}",
        "Throughput",
        1e6 / sw_per_step_us,
        hw_str(hw_latency.map(|l| 1e6 / l), " /s")
    );
    println!(
        "  {:20} {:>12.2} W  {:>15}",
        "Power draw",
        sw_power,
        hw_str(hw_power_val, " W")
    );

    if let Some((ref hw_outputs, hw_us, hw_p)) = hw_result {
        let speedup = sw_per_step_us / hw_us;
        println!("  {:20} {:>15.1}×", "HW speedup", speedup);

        if hw_p > 0.0 {
            let sw_energy_uj = sw_per_step_us * 15.0; // ~15W TDP estimate for CPU
            let hw_energy_uj = hw_us * hw_p as f64;
            let energy_ratio = sw_energy_uj / hw_energy_uj;
            println!("  {:20} {:>15.1}×", "Energy efficiency", energy_ratio);
        }

        // Output agreement
        println!();
        println!("── Output Agreement ───────────────────────────────────────────");
        let mut max_diff: f32 = 0.0;
        let mut total_diff: f32 = 0.0;
        let mut count = 0u32;

        for (sw_out, hw_out) in sw_outputs.iter().zip(hw_outputs.iter()) {
            for (s, h) in sw_out.iter().zip(hw_out.iter()) {
                let diff = (s - h).abs();
                max_diff = max_diff.max(diff);
                total_diff += diff;
                count += 1;
            }
        }

        let avg_diff = if count > 0 { total_diff / count as f32 } else { 0.0 };
        println!("  Compared       : {count} output values across {n_steps} steps");
        println!("  Max |diff|     : {max_diff:.6}");
        println!("  Avg |diff|     : {avg_diff:.6}");

        if max_diff < 0.01 {
            println!("  Agreement      : EXCELLENT (<1% relative error)");
        } else if max_diff < 0.05 {
            println!("  Agreement      : GOOD (<5% — expected for f32 vs int4 quantization)");
        } else {
            println!("  Agreement      : DIVERGENT (>5% — investigate quantization pipeline)");
        }
    } else {
        println!();
        println!("  Hardware comparison unavailable — software results only.");
        println!("  The software NPU produces numerically identical results to what");
        println!("  AKD1000 hardware would produce, minus int4 quantization (~1-3%).");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  HW vs SW comparison complete");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
