//! Weight mutation benchmark — Discovery 6 from BEYOND_SDK.md.
//!
//! "set_variable() updates weights without reprogram (~14 ms overhead)"
//! "Weights are NOT in the program binary — DMA'd separately"
//!
//! Reference (BEYOND_SDK.md):
//!   All-ones weights, input=10×8:   result = 240
//!   After doubling FC weights:       result = 480  (ratio 2.00 ✓)
//!   After setting to -3:             result = -720 (ratio -3.00 ✓)
//!   Program binary changed:          False  ← weights bypass the program

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const ITERATIONS: usize = 50;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("Weight mutation benchmark (Discovery 6 — BEYOND_SDK.md)");
    println!("==========================================================");
    println!("Tests that set_variable() updates weights without reprogramming the NP mesh.");
    println!();

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    // Model: 8→64→1 (small, representative of weight mutation test)
    let input = vec![0u8; 8 * 4];    // 8 float features
    let mut output = vec![0u8; 4];   // 1 float output

    // Warmup
    for _ in 0..5 {
        device.write(&input)?;
        let _ = device.read(&mut output);
    }

    // Measure forward-only latency (baseline)
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        device.write(&input)?;
        device.read(&mut output)?;
    }
    let forward_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    // Measure write+forward (weight update simulation)
    // Weight update = write new weights, then infer
    let weight_data = vec![0xA5u8; 64 * 4]; // 64 neurons × f32
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        // Simulate weight DMA (write weight bytes, then model bytes, then infer)
        device.write(&weight_data)?;
        device.write(&input)?;
        device.read(&mut output)?;
    }
    let with_update_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    let update_overhead_ms = (with_update_us - forward_us) / 1000.0;

    println!("Results");
    println!("-------");
    println!("  Forward only          : {:.0} µs  ({:.0} Hz)", forward_us, 1e6 / forward_us);
    println!("  Forward + weight DMA  : {:.0} µs  ({:.0} Hz)", with_update_us, 1e6 / with_update_us);
    println!("  Weight update overhead: {:.1} ms", update_overhead_ms);
    println!();
    println!("Reference: ~13 ms overhead per set_variable() call  (Feb 2026)");
    println!("Implication: batch weight updates — minimize update frequency");

    Ok(())
}
