//! Enumerate all Akida devices on the system
//!
//! This example demonstrates runtime device discovery.

use akida_driver::{DeviceManager, Result};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("akida_driver=debug")
        .init();

    println!("🧠 Akida Device Enumeration\n");

    // Discover devices at runtime
    let manager = DeviceManager::discover()?;

    println!("Found {} device(s):\n", manager.device_count());

    for device in manager.devices() {
        let caps = device.capabilities();

        println!("📟 Device {}:", device.index());
        println!("   Path:       {}", device.path().display());
        println!("   PCIe:       {}", device.pcie_address());
        println!("   Chip:       {:?}", caps.chip_version);
        println!("   NPUs:       {}", caps.npu_count);
        println!("   Memory:     {} MB SRAM", caps.memory_mb);
        println!(
            "   PCIe:       Gen{} x{} ({:.1} GB/s)",
            caps.pcie.generation, caps.pcie.lanes, caps.pcie.bandwidth_gbps
        );
        println!();
    }

    println!("✅ Discovery complete");

    Ok(())
}
