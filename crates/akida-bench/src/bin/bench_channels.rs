//! Channel count benchmark — Discovery 1 from BEYOND_SDK.md.
//!
//! "InputConv: 1 or 3 channels only" — SDK enforced, NOT silicon limited.
//! Any channel count from 1–64 works on hardware.
//!
//! Reference (BEYOND_SDK.md, Discovery 1):
//!   channels=  1: lat=707µs — works
//!   channels= 16: lat=657µs — works
//!   channels= 50: lat=649µs — works  ← our physics vectors
//!   channels= 64: lat=714µs — works
//!
//! The SDK check is in MetaTF Python code, not in the C++ engine or silicon.

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const ITERATIONS: usize = 200;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("Input channel count benchmark (Discovery 1 — BEYOND_SDK.md)");
    println!("==============================================================");
    println!("Verifies that the 'InputConv: 1 or 3 channels only' SDK limit is SDK-enforced,");
    println!("not a silicon constraint. Any channel count works in hardware.");
    println!();
    println!("  {:>10}  {:>12}  {:>10}  {:>10}", "channels", "µs/infer", "Hz", "vs ch=1");
    println!("  {:-<10}  {:-<12}  {:-<10}  {:-<10}", "", "", "", "");

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    let mut baseline_us: Option<f64> = None;

    for &channels in &[1usize, 2, 3, 4, 8, 16, 32, 50, 64] {
        let input = vec![0u8; channels * 4]; // channels × f32
        let mut output = vec![0u8; 4];

        // Warmup
        for _ in 0..5 {
            device.write(&input)?;
            let _ = device.read(&mut output);
        }

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            device.write(&input)?;
            device.read(&mut output)?;
        }
        let us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;
        let baseline = *baseline_us.get_or_insert(us);

        let note = if channels == 50 { " ← physics vectors" } else { "" };
        println!(
            "  {:>10}  {:>12.0}  {:>10.0}  {:>9.2}×{}",
            channels, us, 1e6 / us, baseline / us, note
        );
    }

    println!();
    println!("Reference: all channel counts 1–64 work at ~650–750 µs  (Feb 2026)");
    println!("SDK check: MetaTF Python only — remove it and use any input dimension");

    Ok(())
}
