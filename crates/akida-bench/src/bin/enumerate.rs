//! Enumerate all Akida devices on the system and print capabilities.
//!
//! Reference measurements (AKD1000, PCIe x1 Gen2):
//!   Device: BrainChip AKD1000 (1e7c:bca1)
//!   PCIe: Gen2 x1, ~500 MB/s theoretical
//!   NPUs: 80 (5×8×2 mesh, 78 functional)
//!   SRAM: 8 MB on-chip + 256 Mbit LPDDR4
//!   BAR0: 16 MB (registers), BAR1: 16 GB (NP mesh window)

use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let manager = akida_driver::DeviceManager::discover()?;

    println!("Akida device enumeration");
    println!("========================");
    println!("Found {} device(s)\n", manager.device_count());

    for info in manager.devices() {
        let caps = info.capabilities();
        println!("Device {}: {}", info.index(), info.path().display());
        println!("  PCIe address : {}", info.pcie_address());
        println!("  Chip version : {:?}", caps.chip_version);
        println!(
            "  PCIe link    : Gen{} x{}  ({:.1} GB/s theoretical)",
            caps.pcie.generation, caps.pcie.lanes, caps.pcie.bandwidth_gbps
        );
        println!("  NPUs         : {}", caps.npu_count);
        println!("  SRAM         : {} MB", caps.memory_mb);

        if let Some(mesh) = &caps.mesh {
            println!(
                "  NP mesh      : {}×{}×{} ({} functional)",
                mesh.x, mesh.y, mesh.z, mesh.functional_count
            );
        }

        if let Some(clock) = caps.clock_mode {
            println!("  Clock mode   : {:?}", clock);
        }

        if let Some(batch) = &caps.batch {
            println!(
                "  Batch        : max={}, optimal={} ({:.1}× speedup)",
                batch.max_batch, batch.optimal_batch, batch.optimal_speedup
            );
        }

        if let Some(power_mw) = caps.power_mw {
            println!("  Power        : {} mW", power_mw);
        }

        if let Some(temp) = caps.temperature_c {
            println!("  Temperature  : {:.1} °C", temp);
        }

        println!(
            "  Weight mut.  : {:?}",
            caps.weight_mutation
        );
        println!();
    }

    Ok(())
}
