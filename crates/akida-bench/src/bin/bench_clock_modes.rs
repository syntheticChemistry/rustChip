//! Clock mode benchmark — reproduces Discovery 4 from BEYOND_SDK.md.
//!
//! "3 modes: Performance / Economy / LowPower"
//! "Economy: 19% slower, 18% less power — sweet spot for physics workloads"
//!
//! Reference table (BEYOND_SDK.md, Discovery 4):
//!   Performance:  909 µs,  901 mW  (default)
//!   Economy:     1080 µs,  739 mW  ← sweet spot: 19% slower, 18% less power
//!   LowPower:    8472 µs,  658 mW  (9.3× slower, 27% less power — avoid)
//!
//! The SDK documents only one clock mode. All three modes were discovered
//! via direct sysfs probing (akida_clock_mode attribute).

use akida_driver::ClockMode;
use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const INPUT_DIM: usize = 50;
const OUTPUT_DIM: usize = 1;
const ITERATIONS: usize = 200;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("Clock mode benchmark (Discovery 4 — BEYOND_SDK.md)");
    println!("===================================================");
    println!("Model : {INPUT_DIM}→256→256→256→{OUTPUT_DIM}");
    println!("Iter  : {} per mode", ITERATIONS);
    println!();

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    println!(
        "  {:>12}  {:>12}  {:>12}  {:>12}",
        "mode", "latency µs", "Hz", "vs perf"
    );
    println!("  {:-<12}  {:-<12}  {:-<12}  {:-<12}", "", "", "", "");

    let modes = [
        (ClockMode::Performance, "Performance"),
        (ClockMode::Economy, "Economy"),
        (ClockMode::LowPower, "LowPower"),
    ];

    let input = vec![0u8; INPUT_DIM * 4];
    let mut output = vec![0u8; OUTPUT_DIM * 4];

    let mut perf_us: Option<f64> = None;

    for (mode, name) in &modes {
        // Attempt to set clock mode via sysfs (requires write permission)
        if let Err(e) = set_clock_mode(device.as_raw_fd(), mode) {
            println!(
                "  {:>12}  (cannot set clock mode: {})",
                name, e
            );
            continue;
        }

        // Warmup
        for _ in 0..10 {
            device.write(&input)?;
            device.read(&mut output)?;
        }

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            device.write(&input)?;
            device.read(&mut output)?;
        }
        let us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;
        let baseline = *perf_us.get_or_insert(us);
        let ratio = us / baseline;

        let note = match mode {
            ClockMode::Economy => " ← sweet spot",
            ClockMode::LowPower => " (avoid for latency-sensitive)",
            _ => "",
        };

        println!(
            "  {:>12}  {:>12.0}  {:>12.0}  {:>11.2}×{}",
            name,
            us,
            1e6 / us,
            ratio,
            note
        );
    }

    println!();
    println!("Reference: Perf=909µs/901mW  Eco=1080µs/739mW  LP=8472µs/658mW  (Feb 2026)");
    println!("Economy mode: -19% speed, -18% power — recommended for burst workloads");

    Ok(())
}

fn set_clock_mode(_fd: std::os::unix::io::RawFd, mode: &ClockMode) -> Result<()> {
    // Clock mode is set via the PCIe address sysfs attribute.
    // This requires knowing the PCIe address of the open device.
    // For a full implementation, use DeviceManager to get the address
    // and write to /sys/bus/pci/devices/{addr}/akida_clock_mode.
    let mode_str = match mode {
        ClockMode::Performance => "performance",
        ClockMode::Economy => "economy",
        ClockMode::LowPower => "low_power",
    };
    // Probe sysfs for the device's clock mode attribute
    for entry in std::fs::read_dir("/sys/bus/pci/devices")?.flatten() {
        let path = entry.path().join("akida_clock_mode");
        if path.exists() {
            std::fs::write(&path, mode_str)?;
            return Ok(());
        }
    }
    anyhow::bail!("no akida_clock_mode sysfs attribute found")
}
