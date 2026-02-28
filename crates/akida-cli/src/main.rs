//! `akida` — command-line interface for BrainChip Akida hardware.
//!
//! ```text
//! USAGE:
//!   akida enumerate                  List all devices and capabilities
//!   akida info <pcie-addr>           Detailed info for one device
//!   akida bind-vfio <pcie-addr>      Bind device to vfio-pci (root)
//!   akida unbind-vfio <pcie-addr>    Unbind from vfio-pci (root)
//!   akida bench [suite]              Run benchmark suite
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "akida", about = "BrainChip Akida hardware CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all Akida devices and their capabilities.
    Enumerate,
    /// Print detailed information for one device.
    Info {
        /// PCIe address (e.g. 0000:a1:00.0) or device index (e.g. 0).
        device: String,
    },
    /// Bind a device to vfio-pci (requires root / CAP_SYS_ADMIN).
    BindVfio {
        /// PCIe address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
    /// Unbind a device from vfio-pci and re-bind to akida_pcie (if loaded).
    UnbindVfio {
        /// PCIe address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
    /// Query the IOMMU group for a device.
    IommuGroup {
        /// PCIe address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Cmd::Enumerate => cmd_enumerate()?,
        Cmd::Info { device } => cmd_info(&device)?,
        Cmd::BindVfio { pcie_addr } => cmd_bind_vfio(&pcie_addr)?,
        Cmd::UnbindVfio { pcie_addr } => cmd_unbind_vfio(&pcie_addr)?,
        Cmd::IommuGroup { pcie_addr } => cmd_iommu_group(&pcie_addr)?,
    }

    Ok(())
}

fn cmd_enumerate() -> Result<()> {
    let mgr = akida_driver::DeviceManager::discover()?;

    println!("Akida devices: {}", mgr.device_count());
    println!();

    for info in mgr.devices() {
        let c = info.capabilities();
        let variant = match c.chip_version {
            akida_driver::ChipVersion::Akd1000 => "AKD1000",
            akida_driver::ChipVersion::Akd1500 => "AKD1500",
            _ => "Unknown",
        };

        println!("[{}] {} @ {}", info.index(), variant, info.pcie_address());
        println!(
            "     PCIe  Gen{} x{}  ({:.1} GB/s theoretical)",
            c.pcie.generation, c.pcie.lanes, c.pcie.bandwidth_gbps
        );
        println!("     NPUs  {}   SRAM  {} MB", c.npu_count, c.memory_mb);

        if let Some(m) = &c.mesh {
            println!("     Mesh  {}×{}×{}  ({} functional)", m.x, m.y, m.z, m.functional_count);
        }
        if let Some(clock) = c.clock_mode {
            println!("     Clock {:?}", clock);
        }
        if let Some(batch) = &c.batch {
            println!(
                "     Batch optimal={}  {:.1}× speedup",
                batch.optimal_batch, batch.optimal_speedup
            );
        }
        if let Some(pw) = c.power_mw {
            println!("     Power {} mW", pw);
        }
        println!("     WeightMut {:?}", c.weight_mutation);
        println!();
    }

    Ok(())
}

fn cmd_info(device: &str) -> Result<()> {
    let mgr = akida_driver::DeviceManager::discover()?;

    // Accept index or PCIe address
    let info = if let Ok(idx) = device.parse::<usize>() {
        mgr.device(idx)?.clone()
    } else {
        mgr.devices()
            .iter()
            .find(|d| d.pcie_address() == device)
            .ok_or_else(|| anyhow::anyhow!("Device not found: {}", device))?
            .clone()
    };

    let c = info.capabilities();
    println!("Device       : {}", info.path().display());
    println!("PCIe address : {}", info.pcie_address());
    println!("Chip version : {:?}", c.chip_version);
    println!("PCIe link    : Gen{} x{} ({:.1} GB/s)", c.pcie.generation, c.pcie.lanes, c.pcie.bandwidth_gbps);
    println!("NPUs         : {}", c.npu_count);
    println!("SRAM         : {} MB", c.memory_mb);

    if let Some(m) = &c.mesh {
        println!(
            "NP mesh      : {}×{}×{} ({} functional, {} disabled)",
            m.x, m.y, m.z, m.functional_count,
            (m.x as u32 * m.y as u32 * m.z as u32).saturating_sub(m.functional_count)
        );
    }
    if let Some(clock) = c.clock_mode {
        println!("Clock mode   : {:?}", clock);
    }
    if let Some(batch) = &c.batch {
        println!(
            "Batch        : max={} optimal={} ({:.1}× speedup)",
            batch.max_batch, batch.optimal_batch, batch.optimal_speedup
        );
    }
    if let Some(pw) = c.power_mw {
        println!("Power        : {} mW", pw);
    }
    if let Some(t) = c.temperature_c {
        println!("Temperature  : {:.1} °C", t);
    }
    println!("WeightMut    : {:?}", c.weight_mutation);

    // IOMMU group (useful for VFIO setup)
    match akida_driver::vfio::iommu_group(info.pcie_address()) {
        Ok(g) => println!("IOMMU group  : {}", g),
        Err(_) => println!("IOMMU group  : (not available — IOMMU disabled?)"),
    }

    Ok(())
}

fn cmd_bind_vfio(pcie_addr: &str) -> Result<()> {
    println!("Binding {} to vfio-pci ...", pcie_addr);
    akida_driver::vfio::bind_to_vfio(pcie_addr)?;
    println!("Done. IOMMU group: {}", akida_driver::vfio::iommu_group(pcie_addr)?);
    println!("Grant access:  sudo chown $USER /dev/vfio/{}", akida_driver::vfio::iommu_group(pcie_addr)?);
    Ok(())
}

fn cmd_unbind_vfio(pcie_addr: &str) -> Result<()> {
    println!("Unbinding {} from vfio-pci ...", pcie_addr);
    akida_driver::vfio::unbind_from_vfio(pcie_addr)?;
    println!("Done.");
    Ok(())
}

fn cmd_iommu_group(pcie_addr: &str) -> Result<()> {
    let group = akida_driver::vfio::iommu_group(pcie_addr)?;
    println!("IOMMU group for {pcie_addr}: {group}");
    println!("Device file: /dev/vfio/{group}");
    Ok(())
}
