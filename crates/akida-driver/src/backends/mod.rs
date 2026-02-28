//! NPU backend implementations
//!
//! Three backends available:
//! - **Kernel**: Uses `/dev/akida*` (requires C kernel module, best performance)
//! - **VFIO**: Pure Rust with DMA via IOMMU (no C module, good performance)
//! - **Userspace**: Memory-mapped PCIe BARs (pure Rust, no DMA, development)
//!
//! Deep Debt Compliance:
//! - Runtime capability discovery (no hardcoding)
//! - Comprehensive error handling
//! - Graceful fallbacks

pub mod kernel;
pub mod mmap;
pub mod software;
pub mod userspace;

pub use kernel::KernelBackend;
pub use software::SoftwareBackend;
pub use userspace::UserspaceBackend;

/// Read NPU power consumption from hwmon sysfs (pure Rust, no `glob` crate).
///
/// Enumerates `/sys/bus/pci/devices/{addr}/hwmon/hwmon*/power1_average`
/// using `std::fs::read_dir` instead of the `glob` external dependency.
pub(crate) fn read_hwmon_power(pcie_address: &str) -> Option<f32> {
    let hwmon_dir = format!("/sys/bus/pci/devices/{pcie_address}/hwmon");
    for entry in std::fs::read_dir(&hwmon_dir).ok()?.flatten() {
        let power_path = entry.path().join("power1_average");
        if let Ok(content) = std::fs::read_to_string(&power_path) {
            if let Ok(microwatts) = content.trim().parse::<u64>() {
                #[allow(clippy::cast_precision_loss)]
                let watts = microwatts as f32 / 1_000_000.0;
                return Some(watts);
            }
        }
    }
    None
}
