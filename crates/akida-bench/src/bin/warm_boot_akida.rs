// SPDX-License-Identifier: AGPL-3.0-or-later

//! Akida warm boot — firmware init via kernel driver, then VFIO takeover.
//!
//! Uses rustChip's internal `glowplug` subsystem (absorbed from coralReef's
//! ember/glowplug). This is fully standalone — no coralReef dependency.
//!
//! ## What it does
//!
//! 1. Detect current driver state
//! 2. If firmware is dead → warm cycle: bind `akida_pcie`, settle, swap to VFIO
//! 3. Verify firmware is alive by probing BAR0 register values
//!
//! ## Usage
//!
//! ```bash
//! # Auto-detect first Akida device
//! cargo run --bin warm_boot_akida
//!
//! # Specify PCIe address
//! cargo run --bin warm_boot_akida -- 0000:e2:00.0
//!
//! # With verbose logging
//! RUST_LOG=debug cargo run --bin warm_boot_akida
//! ```

use akida_driver::glowplug;
use akida_driver::{NpuBackend, VfioBackend};
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=info".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Akida Warm Boot — Sovereign NPU Init");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let pcie_addr = std::env::args()
        .nth(1)
        .or_else(discover_first_akida)
        .ok_or("No PCIe address given and no Akida device found")?;

    println!("Target: {pcie_addr}");
    println!();

    // ── Run sovereign boot ──────────────────────────────────────────────
    let result = glowplug::sovereign_boot(&pcie_addr);

    // ── Print results ───────────────────────────────────────────────────
    println!("── Boot Steps ─────────────────────────────────────────────────");
    for step in &result.steps {
        let icon = match step.status {
            glowplug::StepStatus::Ok => "OK",
            glowplug::StepStatus::Skipped => "SKIP",
            glowplug::StepStatus::Failed => "FAIL",
        };
        println!(
            "  [{icon:>4}] {:24} {:>6}ms  {}",
            step.name,
            step.duration_ms,
            step.detail.as_deref().unwrap_or("")
        );
    }

    println!();
    println!("── Result ─────────────────────────────────────────────────────");
    println!("  Initial driver   : {}", result.initial_driver.as_deref().unwrap_or("none"));
    println!("  Final driver     : {}", result.final_driver.as_deref().unwrap_or("none"));
    println!("  Warm cycle       : {}", result.warm_cycle_performed);
    println!("  Firmware alive   : {}", result.firmware_alive);
    println!("  Success          : {}", result.success);
    println!("  Summary          : {}", result.summary);

    // ── If firmware is alive, do a full register dump ─────────────────
    if result.firmware_alive {
        println!();
        println!("── Post-Boot Register Dump ────────────────────────────────────");
        if let Ok(backend) = VfioBackend::init(&pcie_addr) {
            let device_id = backend.read_bar0_u32(0x0000);
            let version = backend.read_bar0_u32(0x0004);
            let control = backend.read_bar0_u32(0x1094);
            let np_count = backend.read_bar0_u32(0x10C0);
            let sram_r0 = backend.read_bar0_u32(0x1410);
            let dma_mesh = backend.read_bar0_u32(0x4010);

            println!("  DEVICE_ID[0x0000] : {device_id:#010x}");
            println!("  VERSION[0x0004]   : {version:#010x}");
            println!("  CONTROL[0x1094]   : {control:#010x}");
            println!("  NP_COUNT[0x10C0]  : {np_count:#010x}");
            println!("  SRAM_R0[0x1410]   : {sram_r0:#010x}");
            println!("  DMA_MESH[0x4010]  : {dma_mesh:#010x}");
            println!("  is_ready()        : {}", backend.is_ready());
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");

    if result.success {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
