// SPDX-License-Identifier: AGPL-3.0-or-later

//! Query detailed device information
//!
//! Shows all capabilities discovered at runtime.

use akida_driver::{DeviceManager, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("akida_driver=trace")
        .init();

    println!("🧠 Akida Device Information\n");

    let manager = DeviceManager::discover()?;

    for device in manager.devices() {
        let caps = device.capabilities();

        println!("╔════════════════════════════════════════════════════════╗");
        println!(
            "║  Device {} - {}  ║",
            device.index(),
            device.path().display()
        );
        println!("╠════════════════════════════════════════════════════════╣");
        println!("║  Hardware                                              ║");
        println!("║    PCIe Address:  {:37} ║", device.pcie_address());
        println!(
            "║    Chip Version:  {:37} ║",
            format!("{:?}", caps.chip_version)
        );
        println!("║    NPU Count:     {:37} ║", caps.npu_count);
        println!("║    SRAM Memory:   {} MB {:30} ║", caps.memory_mb, "");
        println!("║                                                        ║");
        println!("║  PCIe Configuration                                    ║");
        println!(
            "║    Generation:    Gen{} {:32} ║",
            caps.pcie.generation, ""
        );
        println!("║    Lanes:         x{} {:33} ║", caps.pcie.lanes, "");
        println!(
            "║    Speed:         {:.1} GT/s {:27} ║",
            caps.pcie.speed_gts, ""
        );
        println!(
            "║    Bandwidth:     {:.1} GB/s {:27} ║",
            caps.pcie.bandwidth_gbps, ""
        );

        if let Some(power) = caps.power_mw {
            println!("║                                                        ║");
            println!("║  Power & Thermal                                       ║");
            println!(
                "║    Power:         {:.1} W {:28} ║",
                power as f32 / 1000.0,
                ""
            );
        }

        if let Some(temp) = caps.temperature_c {
            println!("║    Temperature:   {:.1}°C {:28} ║", temp, "");
        }

        println!("╚════════════════════════════════════════════════════════╝");
        println!();
    }

    Ok(())
}
