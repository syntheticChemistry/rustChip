// SPDX-License-Identifier: AGPL-3.0-or-later

//! SRAM probe — discover and test all accessible memory on Akida devices.
//!
//! This binary reads and (optionally) writes to all SRAM regions accessible
//! through BAR0 (registers) and BAR1 (NP mesh window).
//!
//! ## Modes
//!
//! - **probe** (default): Read-only scan of BAR0 registers and BAR1 SRAM regions
//! - **scan**: Deep scan of BAR1 to find all non-zero data regions
//! - **test**: Write/readback test of BAR1 SRAM (destructive — will overwrite data)
//!
//! ## Usage
//!
//! ```bash
//! # Read-only probe — safe, shows what's accessible
//! cargo run --bin probe_sram
//!
//! # Deep scan — find all non-zero data in first 1 MB of BAR1
//! cargo run --bin probe_sram -- scan
//!
//! # Write/readback test — DESTRUCTIVE, tests SRAM is writable
//! cargo run --bin probe_sram -- test
//! ```

use akida_bench::HardwareProbe;
use akida_chip::sram::Bar1Layout;
use akida_driver::sram::SramAccessor;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=info".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    let mode = std::env::args().nth(1).unwrap_or_else(|| "probe".into());

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Akida SRAM Probe — Direct Memory Access Tool");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let hw = HardwareProbe::detect();
    println!("{}", hw.status_line());
    println!();

    if !hw.is_available() {
        println!("No Akida hardware detected.");
        println!("This tool requires physical device access (PCIe BAR mapping via sysfs).");
        println!();
        println!("Requirements:");
        println!("  1. Akida device installed and visible in lspci");
        println!("  2. Root or appropriate permissions for /sys/bus/pci/devices/*/resource*");
        println!("  3. Device bound to vfio-pci or akida_pcie driver");
        println!();
        println!("Run in software simulation mode to verify tool works:");
        show_layout_info();
        return Ok(());
    }

    let mgr = hw.manager().ok_or("No device manager")?;
    let device = mgr.devices().first().ok_or("No devices")?;
    let pcie_addr = device.pcie_address();

    println!("Using device @ {pcie_addr}");
    println!();

    match mode.as_str() {
        "probe" => run_probe(pcie_addr)?,
        "scan" => run_scan(pcie_addr)?,
        "test" => run_test(pcie_addr)?,
        other => {
            eprintln!("Unknown mode: {other}");
            eprintln!("Usage: probe_sram [probe|scan|test]");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Read-only probe of all BAR0 registers and BAR1 SRAM regions.
fn run_probe(pcie_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sram = SramAccessor::open(pcie_addr)?;

    println!("── BAR0 Register Dump ──────────────────────────────────────");
    println!(
        "BAR0 mapped: {} bytes ({} MB)",
        sram.bar0_size(),
        sram.bar0_size() / (1024 * 1024)
    );
    println!();

    let regs = sram.dump_registers()?;
    println!(
        "  {:>20}  {:>10}  {:>12}  Status",
        "Register", "Offset", "Value"
    );
    println!("  {:─>20}  {:─>10}  {:─>12}  {:─>10}", "", "", "", "");

    for r in &regs {
        if r.readable {
            println!(
                "  {:>20}  {:#010x}  {:#010x}    ✓",
                r.name, r.offset, r.value
            );
        } else {
            println!("  {:>20}  {:#010x}  {:>12}    ✗", r.name, r.offset, "—");
        }
    }
    println!();

    // SRAM config analysis
    match sram.read_sram_config() {
        Ok(cfg) => {
            println!("── SRAM Configuration ─────────────────────────────────────");
            println!("  Region 0:     {:#010x}", cfg.region_0);
            println!("  Region 1:     {:#010x}", cfg.region_1);
            println!("  BAR Address:  {:#010x}", cfg.bar_addr);
            println!();
        }
        Err(e) => println!("  SRAM config read failed: {e}"),
    }

    // Per-NP config probe
    println!("── Per-NP Configuration (first 8 NPs) ─────────────────────");
    for np in 0..8u32 {
        match sram.read_np_config(np, 0) {
            Ok(val) => {
                let val1 = sram.read_np_config(np, 4).unwrap_or(0);
                println!("  NP{np:>2}: base={val:#010x}  +0x04={val1:#010x}");
            }
            Err(_) => println!("  NP{np:>2}: not accessible"),
        }
    }
    println!();

    // BAR1 SRAM probe
    println!("── BAR1 SRAM Probe ────────────────────────────────────────");
    let start = Instant::now();
    match sram.probe_bar1(4) {
        Ok(results) => {
            let elapsed = start.elapsed();
            let total = results.len();
            let readable = results.iter().filter(|r| r.readable).count();
            let has_data = results.iter().filter(|r| r.has_data).count();

            println!("  Probed {total} offsets in {elapsed:?}");
            println!("  Readable: {readable}/{total}");
            println!("  Non-zero: {has_data}/{readable}");
            println!();

            if has_data > 0 {
                println!("  Non-zero regions found:");
                for r in results.iter().filter(|r| r.has_data) {
                    println!(
                        "    {:#012x}  {:#010x}  ({})",
                        r.offset,
                        r.value.unwrap_or(0),
                        r.description
                    );
                }
            } else {
                println!("  All probed regions read as zero (sparse mapping, no model loaded)");
                println!("  Load a model to populate NP SRAM, then re-probe");
            }
        }
        Err(e) => {
            println!("  BAR1 probe failed: {e}");
            println!("  BAR1 may not be accessible (check permissions or driver binding)");
        }
    }

    println!();
    show_layout_info();

    Ok(())
}

/// Deep scan of BAR1 to find all non-zero data.
fn run_scan(pcie_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sram = SramAccessor::open(pcie_addr)?;

    println!("── BAR1 Deep Scan ─────────────────────────────────────────");
    println!("Scanning for non-zero data (stride=4 bytes)...");
    println!();

    // Scan in chunks to show progress
    let chunk_size = 256 * 1024; // 256 KB chunks
    let max_scan = 8 * 1024 * 1024; // Scan up to 8 MB (physical SRAM size)
    let mut total_hits = 0usize;

    for chunk_start in (0..max_scan).step_by(chunk_size) {
        let start = Instant::now();
        match sram.scan_bar1_range(chunk_start, chunk_size, 4) {
            Ok(hits) => {
                let elapsed = start.elapsed();
                if hits.is_empty() {
                    print!(".");
                } else {
                    println!();
                    println!(
                        "  [{:#010x}–{:#010x}] {} hits in {:?}:",
                        chunk_start,
                        chunk_start + chunk_size,
                        hits.len(),
                        elapsed
                    );
                    for &(offset, value) in hits.iter().take(16) {
                        println!("    {offset:#010x}: {value:#010x}");
                    }
                    if hits.len() > 16 {
                        println!("    ... and {} more", hits.len() - 16);
                    }
                    total_hits += hits.len();
                }
            }
            Err(e) => {
                println!();
                println!("  Scan stopped at {chunk_start:#x}: {e}");
                break;
            }
        }
    }

    println!();
    println!();
    println!("Total non-zero words found: {total_hits}");
    println!("Total bytes with data: ~{} KB", total_hits * 4 / 1024);
    println!();

    Ok(())
}

/// Write/readback test of BAR1 SRAM.
fn run_test(pcie_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sram = SramAccessor::open(pcie_addr)?;

    println!("── BAR1 Write/Readback Test ────────────────────────────────");
    println!("WARNING: This test WRITES to SRAM. Any loaded model may be corrupted.");
    println!();

    let test_offsets: &[usize] = &[0x0000, 0x1000, 0x2000, 0x4000, 0x8000];
    let test_pattern: u32 = 0xDEAD_BEEF;
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;
    let mut skip_count = 0usize;

    for &offset in test_offsets {
        print!("  Offset {offset:#010x}: ");

        // Read original value
        let original = match sram.read_bar1_u32(offset) {
            Ok(v) => v,
            Err(e) => {
                println!("read failed ({e})  SKIP");
                skip_count += 1;
                continue;
            }
        };

        // Write test pattern
        if let Err(e) = sram.write_bar1_u32(offset, test_pattern) {
            println!("write failed ({e})  SKIP");
            skip_count += 1;
            continue;
        }

        // Read back
        let readback = match sram.read_bar1_u32(offset) {
            Ok(v) => v,
            Err(e) => {
                println!("readback failed ({e})  FAIL");
                fail_count += 1;
                continue;
            }
        };

        // Restore original
        let _ = sram.write_bar1_u32(offset, original);

        if readback == test_pattern {
            println!("write {test_pattern:#010x}, read {readback:#010x}  PASS");
            pass_count += 1;
        } else {
            println!("write {test_pattern:#010x}, read {readback:#010x}  FAIL");
            fail_count += 1;
        }
    }

    println!();
    println!("Results: {pass_count} pass, {fail_count} fail, {skip_count} skip");

    // Byte-level read/write test
    println!();
    println!("── Byte-level Read/Write Test ──────────────────────────────");

    let test_data = b"rustChip SRAM test - ecoPrimals";
    let offset = 0x0000;

    // Save original
    let original = sram.read_bar1(offset, test_data.len())?;

    // Write test data
    sram.write_bar1(offset, test_data)?;

    // Read back
    let readback = sram.read_bar1(offset, test_data.len())?;

    // Restore
    sram.write_bar1(offset, &original)?;

    if readback == test_data {
        println!(
            "  Byte-level write/readback: PASS ({} bytes)",
            test_data.len()
        );
    } else {
        println!("  Byte-level write/readback: FAIL");
        println!("    Wrote: {:02x?}", &test_data[..test_data.len().min(16)]);
        println!("    Read:  {:02x?}", &readback[..readback.len().min(16)]);
    }

    println!();

    Ok(())
}

fn show_layout_info() {
    let layout = Bar1Layout::akd1000();

    println!("── BAR1 Address Layout (AKD1000) ──────────────────────────");
    println!(
        "  Decode range:     {} GB",
        layout.decode_size / (1024 * 1024 * 1024)
    );
    println!(
        "  Physical SRAM:    {} MB",
        akida_chip::bar::bar1::PHYSICAL_SRAM / (1024 * 1024)
    );
    println!("  NP count:         {}", layout.np_count);
    println!(
        "  Per-NP stride:    {} MB",
        layout.np_stride / (1024 * 1024)
    );
    println!(
        "  Per-NP SRAM:      ~{} KB",
        layout.per_np_sram_bytes / 1024
    );
    println!();
    println!("  SRAM types per NP:");
    println!("    Filter (64b)    — convolution kernels / FC weights");
    println!("    Threshold (51b) — activation thresholds / biases");
    println!("    Event (32b)     — spike events / activations");
    println!("    Status (32b)    — layer status / control");
    println!();
    println!("  First 5 NP base offsets in BAR1:");

    for np in 0..5u32 {
        if let Some(base) = layout.np_base_offset(np) {
            println!("    NP{np}: {base:#012x} ({} MB)", base / (1024 * 1024));
        }
    }

    println!();
}
