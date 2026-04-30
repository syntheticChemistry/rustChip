// SPDX-License-Identifier: AGPL-3.0-or-later

//! VFIO hardware probe — userspace BAR access without kernel driver.
//!
//! Unlike `probe_sram` (which uses sysfs resource mmap), this binary
//! accesses BARs through the VFIO device fd. This is the correct path
//! for VFIO-bound devices and works at user level with `/dev/vfio/`
//! group permissions.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin probe_vfio
//! cargo run --bin probe_vfio -- 0000:e2:00.0
//! ```

use akida_driver::{NpuBackend, VfioBackend};
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

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Akida VFIO Probe — Userspace BAR Access");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let pcie_addr = std::env::args()
        .nth(1)
        .or_else(discover_first_akida)
        .ok_or("No PCIe address given and no Akida device found")?;

    println!("Target: {pcie_addr}");

    let group = akida_driver::vfio::iommu_group(&pcie_addr)?;
    println!("IOMMU group: {group}  →  /dev/vfio/{group}");
    println!();

    println!("── Initializing VFIO Backend ───────────────────────────────");
    let start = Instant::now();
    let mut backend = VfioBackend::init(&pcie_addr)?;
    let init_time = start.elapsed();

    let caps = backend.capabilities();
    println!("  Init time     : {init_time:?}");
    println!("  Chip version  : {:?}", caps.chip_version);
    println!("  NPUs          : {}", caps.npu_count);
    println!("  SRAM          : {} MB", caps.memory_mb);
    println!(
        "  PCIe          : Gen{} x{} ({:.1} GB/s)",
        caps.pcie.generation, caps.pcie.lanes, caps.pcie.bandwidth_gbps
    );
    println!("  Ready         : {}", backend.is_ready());
    println!("  Backend       : {}", backend.backend_type());
    println!();

    println!("── BAR0 Control Register Read ────────────────────────────────");
    let _probes: &[(usize, &str)] = &[
        (0x0000, "DEVICE_ID"),
        (0x0004, "VERSION"),
        (0x0008, "STATUS"),
        (0x000C, "CONTROL"),
        (0x0010, "NPU_COUNT"),
        (0x0014, "SRAM_SIZE"),
        (0x0020, "IRQ_STATUS"),
        (0x0024, "IRQ_ENABLE"),
        (0x0100, "MODEL_ADDR_LO"),
        (0x0104, "MODEL_ADDR_HI"),
        (0x0108, "MODEL_SIZE"),
        (0x0200, "INPUT_ADDR_LO"),
        (0x0300, "OUTPUT_ADDR_LO"),
        (0x0400, "INFER_START"),
        (0x0404, "INFER_STATUS"),
    ];

    // BAR0 is already mapped by VfioBackend::init as control_regs
    // We read through the public `is_ready()` and power interface.
    // For detailed register inspection we map BAR1.
    println!("  (BAR0 is mapped internally by VfioBackend)");
    println!("  is_ready()    : {}", backend.is_ready());

    match backend.measure_power() {
        Ok(w) => println!("  Power         : {w:.2} W"),
        Err(e) => println!("  Power         : unavailable ({e})"),
    }
    println!();

    println!("── BAR1 SRAM Map & Probe ────────────────────────────────────");
    let start = Instant::now();
    match backend.map_bar1() {
        Ok(()) => {
            let map_time = start.elapsed();
            println!("  BAR1 mapped   : {} bytes ({} MB) in {map_time:?}",
                backend.sram_size(),
                backend.sram_size() / (1024 * 1024));

            let mut non_zero = 0u32;
            let mut total = 0u32;
            let offsets_to_probe: Vec<usize> = (0..64)
                .map(|i| i * 0x1000)
                .chain((0..20).map(|i| i * 0x4_0000))
                .collect();

            for &offset in &offsets_to_probe {
                if offset + 4 > backend.sram_size() {
                    break;
                }
                match backend.read_sram_u32(offset) {
                    Ok(val) => {
                        total += 1;
                        if val != 0 && val != 0xFFFF_FFFF {
                            non_zero += 1;
                            if non_zero <= 16 {
                                println!("    {offset:#010x}: {val:#010x}");
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            println!("  Probed {total} offsets, {non_zero} non-zero/non-FF");
            if non_zero == 0 {
                println!("  SRAM is empty (no model loaded) — expected for clean start");
            }
        }
        Err(e) => {
            println!("  BAR1 map failed: {e}");
        }
    }
    println!();

    println!("── SRAM Write/Readback Test ─────────────────────────────────");
    let test_offset = 0x0000;
    let test_pattern: u32 = 0xCAFE_BABE;

    match backend.read_sram_u32(test_offset) {
        Ok(original) => {
            match backend.write_sram_u32(test_offset, test_pattern) {
                Ok(()) => {
                    match backend.read_sram_u32(test_offset) {
                        Ok(readback) => {
                            let _ = backend.write_sram_u32(test_offset, original);
                            if readback == test_pattern {
                                println!("  Write {test_pattern:#010x} → Read {readback:#010x}  PASS");
                            } else {
                                println!("  Write {test_pattern:#010x} → Read {readback:#010x}  MISMATCH");
                            }
                        }
                        Err(e) => println!("  Readback failed: {e}"),
                    }
                }
                Err(e) => println!("  Write failed: {e}"),
            }
        }
        Err(e) => println!("  SRAM not accessible for test: {e}"),
    }
    println!();

    println!("── DMA Allocation Test ─────────────────────────────────────");
    match backend.alloc_dma(4096) {
        Ok(buf) => {
            println!("  Allocated 4096-byte DMA buffer");
            println!("  IOVA: {:#x}", buf.iova());
            println!("  Host size: {}", buf.as_slice().len());
        }
        Err(e) => println!("  DMA allocation failed: {e}"),
    }
    println!();

    println!("── Inference Readiness ─────────────────────────────────────");
    println!("  Backend ready : {}", backend.is_ready());
    println!("  Model loaded  : (no model loaded yet)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  VFIO probe complete — hardware accessible at userspace level");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
