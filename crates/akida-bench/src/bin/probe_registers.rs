// SPDX-License-Identifier: AGPL-3.0-or-later

//! BAR0 register probe — empirical register map discovery.
//!
//! Reads all confirmed and suspected register offsets from BAR0,
//! prints values, and identifies the real control/status interface
//! for the AKD1000 VFIO path.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin probe_registers
//! cargo run --bin probe_registers -- 0000:e2:00.0
//! ```

use akida_driver::{NpuBackend, VfioBackend};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=warn".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  AKD1000 BAR0 Register Probe");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let pcie_addr = std::env::args()
        .nth(1)
        .or_else(discover_first_akida)
        .ok_or("No PCIe address given and no Akida device found")?;

    println!("Target: {pcie_addr}");
    let backend = VfioBackend::init(&pcie_addr)?;
    let bar0_size = backend.bar0_size();
    println!("BAR0 size: {bar0_size} bytes ({} MB)", bar0_size / (1024 * 1024));
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // CONFIRMED REGISTERS (from akida-chip probing / BEYOND_SDK.md)
    // ═══════════════════════════════════════════════════════════════════
    println!("── Confirmed Register Reads ────────────────────────────────────");
    println!("  {:>10}  {:>12}  {}", "Offset", "Value", "Description");
    println!("  {:>10}  {:>12}  {}", "──────", "──────────", "───────────");

    let confirmed: &[(usize, &str, Option<u32>)] = &[
        (0x0000, "DEVICE_ID (expect 0x194000a1)", Some(0x194000a1)),
        (0x0004, "VERSION", None),
        (0x0008, "STATUS (mmio.rs, inferred)", None),
        (0x000C, "CONTROL (mmio.rs, inferred)", None),
        (0x0010, "CONTROL (akida-chip, inferred)", None),
        (0x0014, "SRAM_SIZE (mmio.rs)", None),
        (0x0020, "IRQ_STATUS (mmio.rs)", None),
        (0x0024, "IRQ_ENABLE (mmio.rs)", None),
        (0x0100, "MODEL_ADDR_LO", None),
        (0x0104, "MODEL_ADDR_HI", None),
        (0x0108, "MODEL_SIZE", None),
        (0x010C, "MODEL_LOAD", None),
        (0x0200, "eDMA_WRITE_CH0 / INPUT_ADDR_LO", None),
        (0x0204, "INPUT_ADDR_HI", None),
        (0x0208, "INPUT_SIZE", None),
        (0x0300, "eDMA_READ_CH0 / OUTPUT_ADDR_LO", None),
        (0x0304, "OUTPUT_ADDR_HI", None),
        (0x0308, "OUTPUT_SIZE", None),
        (0x0400, "INFER_START", None),
        (0x0404, "INFER_STATUS", None),
        (0x1094, "CONTROL (confirmed @ 0x001094)", Some(0x0000a028)),
        (0x10C0, "NP_COUNT (confirmed, expect 0x5b=91)", Some(0x5b)),
        (0x1410, "SRAM_REGION_0 (confirmed)", Some(0x2000)),
        (0x1418, "SRAM_REGION_1 (confirmed)", Some(0x8000)),
        (0x141C, "SRAM_BAR_ADDR (confirmed)", None),
        (0x1484, "FIRMWARE_VERSION (confirmed)", None),
        (0x1E0C, "NP_ENABLE[0] (confirmed)", Some(0x1)),
        (0x1E10, "NP_ENABLE[1]", None),
        (0x1E14, "NP_ENABLE[2]", None),
        (0x1E18, "NP_ENABLE[3]", None),
        (0x1E1C, "NP_ENABLE[4]", None),
        (0x1E20, "NP_ENABLE[5]", None),
        (0x4010, "DMA_MESH_CONFIG (confirmed)", Some(0x04aa0001)),
    ];

    for &(offset, desc, expected) in confirmed {
        if offset + 4 > bar0_size {
            println!("  {offset:#010x}  OUT OF RANGE  {desc}");
            continue;
        }
        let val = backend.read_bar0_u32(offset);
        let marker = match expected {
            Some(exp) if val == exp => " [MATCH]",
            Some(_) => " [DIFFER]",
            None => "",
        };
        println!("  {offset:#010x}  {val:#010x}   {desc}{marker}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // SEARCH FOR STATUS/CONTROL NEAR CONFIRMED CONTROL (0x1094)
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("── Registers Near Confirmed CONTROL (0x1080–0x1100) ──────────");

    for offset in (0x1080..0x1100).step_by(4) {
        if offset + 4 > bar0_size { break; }
        let val = backend.read_bar0_u32(offset);
        if val != 0 && val != 0xFFFF_FFFF && val != 0xBADF_5040 {
            println!("  {offset:#010x}: {val:#010x}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // FIRST 0x200 bytes — find our "real" registers
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("── First 0x200 bytes of BAR0 (non-zero only) ─────────────────");

    for offset in (0x0000..0x0200).step_by(4) {
        if offset + 4 > bar0_size { break; }
        let val = backend.read_bar0_u32(offset);
        if val != 0 && val != 0xFFFF_FFFF && val != 0xBADF_5040 {
            println!("  {offset:#010x}: {val:#010x}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // WRITE-READBACK TEST ON CONFIRMED CONTROL AT 0x1094
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("── Control Register (0x1094) Write Test ──────────────────────");
    let original = backend.read_bar0_u32(0x1094);
    println!("  Original     : {original:#010x}");

    backend.write_bar0_u32(0x1094, original | 0x1);
    std::thread::sleep(std::time::Duration::from_millis(10));
    let after_bit0 = backend.read_bar0_u32(0x1094);
    println!("  After |= 0x1 : {after_bit0:#010x}");

    backend.write_bar0_u32(0x1094, original);
    std::thread::sleep(std::time::Duration::from_millis(10));
    let restored = backend.read_bar0_u32(0x1094);
    println!("  Restored     : {restored:#010x}");

    let writable = original != after_bit0;
    println!("  Writable     : {writable}");

    // ═══════════════════════════════════════════════════════════════════
    // PER-NP CONFIG BLOCK (0xE000+)
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("── Per-NP Config (0xE000, first 5 NPs × 0x100 stride) ───────");

    for np in 0..5 {
        let base = 0xE000 + np * 0x100;
        if base + 0x20 > bar0_size { break; }
        print!("  NP{np:02}: ");
        for off in (0..0x20).step_by(4) {
            let val = backend.read_bar0_u32(base + off);
            print!("{val:08x} ");
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════
    // FULL BAR0 CENSUS
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("── BAR0 Non-Zero Register Census ─────────────────────────────");

    let mut non_zero = 0u32;
    let mut badf_count = 0u32;
    let mut zero_count = 0u32;
    let mut ffff_count = 0u32;
    let total = bar0_size / 4;

    let start = Instant::now();
    for offset in (0..bar0_size).step_by(4) {
        let val = backend.read_bar0_u32(offset);
        match val {
            0 => zero_count += 1,
            0xFFFF_FFFF => ffff_count += 1,
            0xBADF_5040 => badf_count += 1,
            _ => non_zero += 1,
        }
    }
    let scan_time = start.elapsed();

    println!("  Total dwords   : {total}");
    println!("  Zero           : {zero_count}");
    println!("  0xFFFFFFFF     : {ffff_count}");
    println!("  0xBADF5040     : {badf_count} (\"bad food\" — uninitialized)");
    println!("  Other non-zero : {non_zero}");
    println!("  Scan time      : {scan_time:?}");

    // Print first 200 unique non-zero addresses
    println!();
    println!("── All Non-Zero Registers (first 200) ────────────────────────");
    let mut count = 0u32;
    for offset in (0..bar0_size).step_by(4) {
        let val = backend.read_bar0_u32(offset);
        if val != 0 && val != 0xFFFF_FFFF && val != 0xBADF_5040 {
            println!("  {offset:#010x}: {val:#010x}");
            count += 1;
            if count >= 200 { break; }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Register probe complete — {non_zero} active registers found");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
