//! Batch amortization deep-dive — reproduces Discovery 3 from BEYOND_SDK.md.
//!
//! "Batch=8 amortizes PCIe: 948→390 µs/sample (2.4× throughput)"
//!
//! Uses a 50→256→256→256→1 model (108 KB program, physics-scale).
//! Sweeps batch sizes 1–64 and plots the throughput curve.
//!
//! Reference table (BEYOND_SDK.md, Discovery 3):
//!   batch=1:  0.95ms total,   948 µs/sample,  1,055 /s
//!   batch=2:  1.14ms total,   568 µs/sample,  1,760 /s
//!   batch=4:  1.70ms total,   426 µs/sample,  2,346 /s
//!   batch=8:  3.12ms total,   390 µs/sample,  2,566 /s  ← sweet spot
//!   batch=16: 7.70ms total,   481 µs/sample,  2,078 /s
//!   batch=32: 18.57ms total,  580 µs/sample,  1,723 /s
//!   batch=64: 29.28ms total,  458 µs/sample,  2,186 /s

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const INPUT_DIM: usize = 50;
const OUTPUT_DIM: usize = 1;
const ITERATIONS: usize = 300;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("Batch amortization sweep (Discovery 3 — BEYOND_SDK.md)");
    println!("======================================================");
    println!("Model  : {INPUT_DIM}→256→256→256→{OUTPUT_DIM}  (physics-scale ESN)");
    println!("Iter   : {} per batch size", ITERATIONS);
    println!();
    println!(
        "  {:>7}  {:>12}  {:>12}  {:>12}  {:>10}",
        "batch", "total ms", "µs/sample", "samples/s", "vs batch=1"
    );
    println!("  {:-<7}  {:-<12}  {:-<12}  {:-<12}  {:-<10}", "", "", "", "", "");

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    let mut baseline_us: Option<f64> = None;

    for &batch in &[1usize, 2, 4, 8, 16, 32, 64] {
        let input = vec![0u8; INPUT_DIM * 4 * batch];
        let mut output = vec![0u8; OUTPUT_DIM * 4 * batch];

        // Warmup
        for _ in 0..5 {
            device.write(&input)?;
            device.read(&mut output)?;
        }

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            device.write(&input)?;
            device.read(&mut output)?;
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
        let us_per_sample = total_ms * 1000.0 / batch as f64;
        let samples_per_sec = 1e6 / us_per_sample;

        let baseline = *baseline_us.get_or_insert(us_per_sample);
        let speedup = baseline / us_per_sample;

        let sweet = if batch == 8 { " ← sweet spot" } else { "" };
        println!(
            "  {:>7}  {:>12.2}  {:>12.0}  {:>12.0}  {:>9.2}×{}",
            batch, total_ms, us_per_sample, samples_per_sec, speedup, sweet
        );
    }

    println!();
    println!("Reference sweet spot: batch=8, 390 µs/sample, 2,566 /s, 2.4× speedup  (Feb 2026)");

    Ok(())
}
