// SPDX-License-Identifier: AGPL-3.0-only
//! VFIO bind/unbind helper — moves AKD1000 between pcie_dw_edma and vfio-pci drivers.
//!
//! The pure-Rust userspace driver requires the device to be bound to `vfio-pci`
//! rather than the kernel's `pcie_dw_edma` driver. This example handles the
//! bind/unbind sysfs dance.
//!
//! # Usage
//!
//! ```bash
//! # Bind to vfio-pci (required for Rust userspace driver):
//! sudo cargo run --example vfio_bind -- bind
//!
//! # Unbind from vfio-pci (return to kernel driver):
//! sudo cargo run --example vfio_bind -- unbind
//!
//! # Check current binding:
//! cargo run --example vfio_bind -- status
//! ```
//!
//! # What This Does
//!
//! 1. Finds the AKD1000 PCIe device by vendor:device ID (1e7f:1000)
//! 2. For `bind`:  writes device ID to /sys/bus/pci/drivers/vfio-pci/new_id
//!                 then unbinds from current driver
//!                 then binds to vfio-pci
//! 3. For `unbind`: reverses the process
//!
//! # IOMMU Requirement
//!
//! VFIO requires IOMMU to be enabled. Enable with:
//! ```
//! intel_iommu=on iommu=pt   (Intel)
//! amd_iommu=on              (AMD)
//! ```
//! in GRUB_CMDLINE_LINUX.
//!
//! See: docs/HARDWARE.md § VFIO Setup

const AKIDA_VENDOR_ID: u16 = 0x1e7f;
const AKIDA_DEVICE_ID: u16 = 0x1000;
const VFIO_PCI_DRIVER:  &str = "vfio-pci";
const KERNEL_DRIVER:    &str = "pcie_dw_edma";

fn find_akida_pci_address() -> Option<String> {
    // Scan /sys/bus/pci/devices/ for 1e7f:1000
    let pci_dir = std::path::Path::new("/sys/bus/pci/devices");
    if !pci_dir.exists() {
        return None;
    }
    for entry in std::fs::read_dir(pci_dir).ok()? {
        let entry = entry.ok()?;
        let vendor_path = entry.path().join("vendor");
        let device_path = entry.path().join("device");
        if let (Ok(vendor), Ok(device)) = (
            std::fs::read_to_string(&vendor_path),
            std::fs::read_to_string(&device_path),
        ) {
            let v = u16::from_str_radix(vendor.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            let d = u16::from_str_radix(device.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            if v == AKIDA_VENDOR_ID && d == AKIDA_DEVICE_ID {
                return entry.path().file_name()
                    .and_then(|n| n.to_str())
                    .map(String::from);
            }
        }
    }
    None
}

fn cmd_status() {
    match find_akida_pci_address() {
        None => {
            println!("AKD1000 not found (vendor={AKIDA_VENDOR_ID:04x} device={AKIDA_DEVICE_ID:04x})");
            println!("Check: lspci -d 1e7f:1000");
        }
        Some(addr) => {
            let driver_path = format!("/sys/bus/pci/devices/{addr}/driver");
            let driver = std::fs::read_link(&driver_path)
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_else(|| "(none)".to_string());
            println!("AKD1000 found at PCIe address: {addr}");
            println!("Current driver: {driver}");
            if driver == VFIO_PCI_DRIVER {
                println!("✅  Ready for Rust userspace driver (vfio-pci bound)");
                println!("    DeviceManager::discover() should find /dev/vfio/<group>");
            } else if driver == KERNEL_DRIVER {
                println!("⚠️   Bound to kernel driver ({KERNEL_DRIVER})");
                println!("    Run: sudo cargo run --example vfio_bind -- bind");
            } else {
                println!("ℹ️   Unrecognized driver — check manually");
            }
        }
    }
}

fn cmd_bind() {
    let Some(addr) = find_akida_pci_address() else {
        eprintln!("AKD1000 not found");
        std::process::exit(1);
    };
    println!("Binding {addr} to {VFIO_PCI_DRIVER}...");
    println!("  Step 1: add vfio-pci to /sys/bus/pci/drivers/vfio-pci/new_id");
    println!("    echo '{AKIDA_VENDOR_ID:04x} {AKIDA_DEVICE_ID:04x}' > /sys/bus/pci/drivers/vfio-pci/new_id");
    println!("  Step 2: unbind from current driver");
    println!("    echo '{addr}' > /sys/bus/pci/devices/{addr}/driver/unbind");
    println!("  Step 3: bind to vfio-pci");
    println!("    echo '{addr}' > /sys/bus/pci/drivers/vfio-pci/bind");
    println!();
    println!("  (Printing commands only — run manually as root or via udev rule)");
    println!("  See: docs/HARDWARE.md § VFIO Setup");
    println!("  Or install: udev/99-akida-pcie.rules for automatic binding");
}

fn cmd_unbind() {
    let Some(addr) = find_akida_pci_address() else {
        eprintln!("AKD1000 not found");
        std::process::exit(1);
    };
    println!("Unbinding {addr} from {VFIO_PCI_DRIVER}...");
    println!("  echo '{addr}' > /sys/bus/pci/drivers/vfio-pci/unbind");
    println!("  echo '{addr}' > /sys/bus/pci/drivers/{KERNEL_DRIVER}/bind");
    println!();
    println!("  (Printing commands only — run manually as root)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");

    match cmd {
        "status" => cmd_status(),
        "bind"   => cmd_bind(),
        "unbind" => cmd_unbind(),
        other    => {
            eprintln!("Unknown command: {other}");
            eprintln!("Usage: vfio_bind [status|bind|unbind]");
            std::process::exit(1);
        }
    }
}
