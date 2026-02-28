//! Inference latency benchmark — single-sample and batched.
//!
//! Reference measurements (AKD1000, PCIe x1 Gen2, Feb 2026):
//!   Single inference:  54 µs  (18,500 Hz)   — Phase C pure Rust driver
//!   Batch=8:           ~390 µs/sample         — 2.4× throughput
//!   20,700 infer/sec   (batch=8 sustained)
//!
//! The 54 µs is PCIe-dominated: ~650 µs kernel driver round-trip
//! compressed by Phase C's direct ioctl path.
//!
//! Discovery 3 (BEYOND_SDK.md) batch sweet-spot table:
//!   batch=1:  948 µs/sample   1,055 /s
//!   batch=2:  568 µs/sample   1,760 /s
//!   batch=4:  426 µs/sample   2,346 /s
//!   batch=8:  390 µs/sample   2,566 /s  ← sweet spot
//!   batch=16: 481 µs/sample   2,078 /s
//!
//! Usage:
//!   cargo run --bin bench_latency
//!   cargo run --bin bench_latency -- --iterations 2000

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const DEFAULT_ITERATIONS: usize = 1000;
const INPUT_DIM: usize = 50;   // Physics feature vector (plaquette, Polyakov, etc.)
const OUTPUT_DIM: usize = 1;   // ESN readout (β_c prediction, phase classifier, etc.)

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let iterations = parse_arg(&args, "--iterations", DEFAULT_ITERATIONS);

    println!("Inference latency benchmark");
    println!("===========================");
    println!("Model topology : {INPUT_DIM}→256→{OUTPUT_DIM}  (FC, int4 weights)");
    println!("Iterations     : {}", iterations);
    println!();

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    let input_bytes = INPUT_DIM * 4; // f32 features
    let output_bytes = OUTPUT_DIM * 4;
    let input = vec![0u8; input_bytes];
    let mut output = vec![0u8; output_bytes];

    // Warmup
    for _ in 0..20 {
        device.write(&input)?;
        device.read(&mut output)?;
    }

    // Single-sample latency
    let mut latencies_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        device.write(&input)?;
        device.read(&mut output)?;
        latencies_us.push(t0.elapsed().as_micros() as f64);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies_us.iter().sum::<f64>() / iterations as f64;
    let p50 = latencies_us[iterations / 2];
    let p95 = latencies_us[(iterations as f64 * 0.95) as usize];
    let p99 = latencies_us[(iterations as f64 * 0.99) as usize];
    let min = latencies_us[0];
    let max = latencies_us[iterations - 1];

    println!("Single-sample latency");
    println!("---------------------");
    println!("  mean : {:.0} µs  ({:.0} Hz)", mean, 1e6 / mean);
    println!("  min  : {:.0} µs", min);
    println!("  p50  : {:.0} µs", p50);
    println!("  p95  : {:.0} µs", p95);
    println!("  p99  : {:.0} µs", p99);
    println!("  max  : {:.0} µs", max);
    println!();

    // Batch amortization sweep (Discovery 3)
    println!("Batch amortization (PCIe round-trip amortization)");
    println!("-------------------------------------------------");
    println!("  {:>7}  {:>12}  {:>10}  {:>12}", "batch", "µs/sample", "samples/s", "vs batch=1");

    let batch_1_lat = run_batch(&mut device, 1, INPUT_DIM, OUTPUT_DIM, 200)?;

    for &batch in &[1usize, 2, 4, 8, 16, 32] {
        let us_per = run_batch(&mut device, batch, INPUT_DIM, OUTPUT_DIM, 200)?;
        let speedup = batch_1_lat / us_per;
        println!(
            "  {:>7}  {:>12.0}  {:>10.0}  {:>11.2}×",
            batch,
            us_per,
            1e6 / us_per,
            speedup
        );
    }

    println!();
    println!("Reference: 54 µs / 18,500 Hz single  |  390 µs / 2,566 /s at batch=8  (Feb 2026)");

    Ok(())
}

fn run_batch(
    device: &mut akida_driver::AkidaDevice,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    iterations: usize,
) -> Result<f64> {
    let input = vec![0u8; input_dim * 4 * batch];
    let mut output = vec![0u8; output_dim * 4 * batch];

    // Warmup
    for _ in 0..10 {
        device.write(&input)?;
        device.read(&mut output)?;
    }

    let t0 = Instant::now();
    for _ in 0..iterations {
        device.write(&input)?;
        device.read(&mut output)?;
    }
    let elapsed_us = t0.elapsed().as_micros() as f64;
    Ok(elapsed_us / (iterations * batch) as f64)
}

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}
