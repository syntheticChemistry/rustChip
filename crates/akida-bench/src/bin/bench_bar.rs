// SPDX-License-Identifier: AGPL-3.0-or-later

//! BAR layout probe — Discovery 8 from `BEYOND_SDK.md`.
//!
//! "`PCIe` BAR1 exposes 16 GB address space — full NP mesh decode range"
//!
//! Reference (`BEYOND_SDK.md`, Discovery 8):
//!   BAR0: 0x84000000,       16 MB,  32-bit non-prefetch,  register space
//!   BAR1: 0x4000000000,     16 GB,  64-bit prefetchable,  NP mesh window
//!   BAR3: 0x4400000000,     32 MB,  64-bit prefetchable,  secondary memory
//!   BAR5: 0x7000,           128 B,  I/O ports,            control
//!   BAR6: 0x85000000,       512 KB, Expansion ROM,        firmware
//!
//! With 78 NPs and 16 GB, each NP could address ~200 MB.
//! BAR1 first 64 KB reads as all-zeros (sparse mapping).
//!
//! ## Evolution
//!
//! This benchmark now performs actual MMIO probing of BAR0 control
//! registers in addition to sysfs inspection. BAR1 SRAM probing
//! is available via the dedicated `probe_sram` binary.

use akida_bench::HardwareProbe;
use akida_driver::sram::SramAccessor;
use std::path::Path;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("BAR layout probe (Discovery 8 — BEYOND_SDK.md)");
    println!("================================================");
    println!();

    let hw = HardwareProbe::detect();
    println!("{}", hw.status_line());
    println!();

    let mgr = if let Some(m) = hw.manager() {
        m
    } else {
        println!("No hardware detected — showing expected layout only");
        println!();
        show_expected_layout();
        return Ok(());
    };

    for info in mgr.devices() {
        let addr = info.pcie_address();
        println!("Device: {} @ {}", info.path().display(), addr);
        println!();

        // Phase 1: sysfs BAR layout (existing functionality)
        let resource_path = format!("/sys/bus/pci/devices/{addr}/resource");
        if Path::new(&resource_path).exists() {
            let content = std::fs::read_to_string(&resource_path)?;
            println!("  BAR layout from sysfs:");
            println!(
                "  {:>5}  {:>18}  {:>18}  {:>12}  flags",
                "BAR", "start", "end", "size"
            );
            println!(
                "  {:-<5}  {:-<18}  {:-<18}  {:-<12}  {:-<10}",
                "", "", "", "", ""
            );

            for (i, line) in content.lines().enumerate() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 3 {
                    continue;
                }
                let start = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).unwrap_or(0);
                let end = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap_or(0);
                let flags = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).unwrap_or(0);

                if start == 0 && end == 0 {
                    continue;
                }

                let size = end - start + 1;
                let size_str = humanize_size(size);
                let flag_str = bar_flags(flags);

                println!("  {i:>5}  {start:#018x}  {end:#018x}  {size_str:>12}  {flag_str}");
            }
            println!();
        } else {
            println!("  (sysfs resource file not found)");
        }

        // Phase 2: actual MMIO probing of BAR0 (new capability)
        println!("  BAR0 MMIO probe (control registers):");
        match SramAccessor::open(addr) {
            Ok(sram) => {
                println!("    BAR0 mapped: {} MB", sram.bar0_size() / (1024 * 1024));

                match sram.dump_registers() {
                    Ok(regs) => {
                        for r in &regs {
                            if r.readable {
                                let annotation = annotate_register(&r.name, r.value);
                                println!(
                                    "    {:#010x}  {:#010x}  {}{}",
                                    r.offset, r.value, r.name, annotation
                                );
                            }
                        }
                    }
                    Err(e) => println!("    Register dump failed: {e}"),
                }
            }
            Err(e) => {
                println!("    BAR0 mapping failed: {e}");
                println!("    (need root or vfio-pci binding)");
            }
        }

        println!();

        // Cross-check with expected BAR1 size
        let expected_bar1_gb = 16u64;
        println!("  Expected BAR1: {expected_bar1_gb} GB decode range");
        println!(
            "  With 78 NPs: ~{} MB addressable per NP",
            (expected_bar1_gb * 1024) / 78
        );
        println!();
    }

    println!("Reference: BAR1=16GB (full NP mesh), BAR0=16MB (registers)  (Feb 2026)");
    println!("BAR1 first 64KB reads as all-zeros — sparse NP-mapped layout");
    println!();
    println!("For detailed SRAM probing, use: cargo run --bin probe_sram");

    Ok(())
}

fn annotate_register(name: &str, value: u32) -> String {
    match name {
        "DEVICE_ID" if value == 0x1940_00a1 => " (AKD1000 confirmed)".to_string(),
        "NP_COUNT" => format!(" ({} NPs)", value & 0xFF),
        "DMA_MESH_CONFIG" => " (DMA config word)".to_string(),
        "SRAM_REGION_0" => " (SRAM region 0)".to_string(),
        "SRAM_REGION_1" => " (SRAM region 1)".to_string(),
        _ if name.starts_with("NP_ENABLE") => {
            if value == 1 {
                " (enabled)".to_string()
            } else {
                " (disabled)".to_string()
            }
        }
        _ => String::new(),
    }
}

fn show_expected_layout() {
    println!("  Expected BAR layout (from hardware probing, Feb 2026):");
    println!("  BAR0: 16 MB   32-bit non-prefetch   Register space (MMIO)");
    println!("  BAR1: 16 GB   64-bit prefetchable   NP mesh / SRAM window");
    println!("  BAR3: 32 MB   64-bit prefetchable   Secondary memory");
    println!("  BAR5: 128 B   I/O ports              Control ports");
    println!("  BAR6: 512 KB  Expansion ROM          Firmware");
    println!();
    println!("  For SRAM probing, use: cargo run --bin probe_sram");
}

fn humanize_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}

const fn bar_flags(flags: u64) -> &'static str {
    match flags & 0xF {
        0x0 => "32-bit non-prefetch",
        0x4 => "64-bit non-prefetch",
        0x8 => "32-bit prefetchable",
        0xC => "64-bit prefetchable",
        0x1 => "I/O port",
        _ => "unknown",
    }
}
