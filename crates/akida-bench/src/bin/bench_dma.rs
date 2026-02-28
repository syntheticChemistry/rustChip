//! DMA throughput benchmark — sustained read + write to device SRAM.
//!
//! Reference measurement (AKD1000, PCIe x1 Gen2, Feb 2026):
//!   Sustained DMA throughput: 37 MB/s (read + write combined)
//!
//! The AKD1000 is PCIe x1 Gen2 (theoretical 500 MB/s). Measured throughput
//! is 37 MB/s, consistent with the DW eDMA controller overhead and the
//! Akida kernel module's scatter-gather transfer implementation.
//!
//! Usage:
//!   cargo run --bin bench_dma
//!   cargo run --bin bench_dma -- --size-kb 64 --iterations 200

use anyhow::Result;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

const DEFAULT_TRANSFER_KB: usize = 4;
const DEFAULT_ITERATIONS: usize = 500;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let transfer_kb = parse_arg(&args, "--size-kb", DEFAULT_TRANSFER_KB);
    let iterations = parse_arg(&args, "--iterations", DEFAULT_ITERATIONS);
    let transfer_bytes = transfer_kb * 1024;

    println!("DMA throughput benchmark");
    println!("========================");
    println!("Transfer size : {} KB", transfer_kb);
    println!("Iterations    : {}", iterations);
    println!();

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    let payload = vec![0xA5u8; transfer_bytes];

    // Warmup
    for _ in 0..10 {
        device.write(&payload)?;
    }

    // Write benchmark
    let t0 = Instant::now();
    for _ in 0..iterations {
        device.write(&payload)?;
    }
    let write_elapsed = t0.elapsed();

    // Read benchmark
    let mut readback = vec![0u8; transfer_bytes];
    let t0 = Instant::now();
    for _ in 0..iterations {
        device.read(&mut readback)?;
    }
    let read_elapsed = t0.elapsed();

    // Combined (interleaved write + read, simulating real inference DMA)
    let t0 = Instant::now();
    for _ in 0..iterations {
        device.write(&payload)?;
        device.read(&mut readback)?;
    }
    let combined_elapsed = t0.elapsed();

    let total_write_bytes = (iterations * transfer_bytes) as f64;
    let total_rw_bytes = (iterations * transfer_bytes * 2) as f64;

    println!("Results");
    println!("-------");
    print_throughput("Write only ", write_elapsed, total_write_bytes);
    print_throughput("Read only  ", read_elapsed, total_write_bytes);
    print_throughput("Read+Write ", combined_elapsed, total_rw_bytes);

    println!();
    println!("Reference: 37 MB/s sustained (AKD1000, PCIe x1 Gen2, Feb 2026)");

    Ok(())
}

fn print_throughput(label: &str, elapsed: Duration, bytes: f64) {
    let secs = elapsed.as_secs_f64();
    let mb_s = (bytes / 1_048_576.0) / secs;
    let per_transfer_us = (secs / (bytes / 4096.0)) * 1e6;
    println!(
        "  {}: {:.1} MB/s  ({:.0} µs / 4KB transfer)",
        label, mb_s, per_transfer_us
    );
}

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}
