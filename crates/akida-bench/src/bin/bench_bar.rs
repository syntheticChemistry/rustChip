//! BAR layout probe — Discovery 8 from BEYOND_SDK.md.
//!
//! "PCIe BAR1 exposes 16 GB address space — full NP mesh decode range"
//!
//! Reference (BEYOND_SDK.md, Discovery 8):
//!   BAR0: 0x84000000,       16 MB,  32-bit non-prefetch,  register space
//!   BAR1: 0x4000000000,     16 GB,  64-bit prefetchable,  NP mesh window
//!   BAR3: 0x4400000000,     32 MB,  64-bit prefetchable,  secondary memory
//!   BAR5: 0x7000,           128 B,  I/O ports,            control
//!   BAR6: 0x85000000,       512 KB, Expansion ROM,        firmware
//!
//! With 78 NPs and 16 GB, each NP could address ~200 MB.
//! BAR1 first 64 KB reads as all-zeros (sparse mapping).

use anyhow::Result;
use std::path::Path;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("BAR layout probe (Discovery 8 — BEYOND_SDK.md)");
    println!("================================================");
    println!();

    let manager = akida_driver::DeviceManager::discover()?;

    for info in manager.devices() {
        let addr = info.pcie_address();
        println!("Device: {} @ {}", info.path().display(), addr);
        println!();

        let resource_path = format!("/sys/bus/pci/devices/{addr}/resource");
        if !Path::new(&resource_path).exists() {
            println!("  (resource file not found — need hardware access)");
            continue;
        }

        let content = std::fs::read_to_string(&resource_path)?;
        println!("  BAR layout from sysfs:");
        println!("  {:>5}  {:>18}  {:>18}  {:>12}  {}", "BAR", "start", "end", "size", "flags");
        println!("  {:-<5}  {:-<18}  {:-<18}  {:-<12}  {:-<10}", "", "", "", "", "");

        for (i, line) in content.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let start = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).unwrap_or(0);
            let end   = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap_or(0);
            let flags = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).unwrap_or(0);

            if start == 0 && end == 0 {
                continue;
            }

            let size = end - start + 1;
            let size_str = humanize_size(size);
            let flag_str = bar_flags(flags);

            println!("  {:>5}  {:#018x}  {:#018x}  {:>12}  {}",
                i, start, end, size_str, flag_str);
        }

        println!();

        // Cross-check with expected BAR1 size
        let expected_bar1_gb = 16u64;
        println!("  Expected BAR1: {} GB decode range", expected_bar1_gb);
        println!("  With 78 NPs: ~{} MB addressable per NP",
            (expected_bar1_gb * 1024) / 78);
        println!();
    }

    println!("Reference: BAR1=16GB (full NP mesh), BAR0=16MB (registers)  (Feb 2026)");
    println!("BAR1 first 64KB reads as all-zeros — sparse NP-mapped layout");

    Ok(())
}

fn humanize_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

fn bar_flags(flags: u64) -> &'static str {
    match flags & 0xF {
        0x0 => "32-bit non-prefetch",
        0x4 => "64-bit non-prefetch",
        0x8 => "32-bit prefetchable",
        0xC => "64-bit prefetchable",
        0x1 => "I/O port",
        _ => "unknown",
    }
}
