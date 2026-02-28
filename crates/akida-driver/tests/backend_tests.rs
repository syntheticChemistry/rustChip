//! Backend validation tests
//!
//! Tests that kernel and userspace backends produce identical results

use akida_driver::{select_backend, BackendSelection};

#[test]
#[ignore] // Requires hardware
fn test_kernel_backend() {
    let backend = select_backend(BackendSelection::Kernel, "0").expect("Kernel backend init");
    assert!(backend.is_ready());
    println!("Kernel backend: {:?}", backend.backend_type());
    println!("  NPUs: {}", backend.capabilities().npu_count);
    println!("  Memory: {} MB", backend.capabilities().memory_mb);
}

#[test]
#[ignore] // Requires hardware
fn test_userspace_backend() {
    let backend = select_backend(BackendSelection::Userspace, "0000:a1:00.0")
        .expect("Userspace backend init");
    assert!(backend.is_ready());
    println!("Userspace backend: {:?}", backend.backend_type());
    println!("  NPUs: {}", backend.capabilities().npu_count);
    println!("  Memory: {} MB", backend.capabilities().memory_mb);
}

#[test]
#[ignore] // Requires hardware
fn test_both_backends_identical_capabilities() {
    // Initialize both backends
    let kernel = select_backend(BackendSelection::Kernel, "0").expect("Kernel backend");
    let userspace =
        select_backend(BackendSelection::Userspace, "0000:a1:00.0").expect("Userspace backend");

    // Capabilities should be identical
    assert_eq!(
        kernel.capabilities().npu_count,
        userspace.capabilities().npu_count,
        "NPU counts differ"
    );

    assert_eq!(
        kernel.capabilities().memory_mb,
        userspace.capabilities().memory_mb,
        "Memory sizes differ"
    );

    assert_eq!(
        kernel.capabilities().chip_version,
        userspace.capabilities().chip_version,
        "Chip versions differ"
    );

    println!("✅ Both backends report identical capabilities");
}

#[test]
#[ignore] // Requires hardware
fn test_power_measurement() {
    // Test kernel backend power query
    if let Ok(backend) = select_backend(BackendSelection::Auto, "0") {
        match backend.measure_power() {
            Ok(power) => {
                println!("NPU power: {power:.2}W");
                assert!(power > 0.0 && power < 100.0, "Power out of range");
            }
            Err(e) => println!("ℹ️  Power measurement unavailable: {e}"),
        }
    }
}
