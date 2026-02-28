//! Integration tests for dual-backend NPU driver
//!
//! Deep Debt: Tests verify both kernel and userspace backends produce identical results

use akida_driver::{
    backends::{KernelBackend, UserspaceBackend},
    select_backend, BackendSelection, NpuBackend,
};

/// Test that both backends discover the same capabilities
#[test]
#[ignore] // Requires actual Akida hardware
fn test_backend_capability_parity() {
    // Initialize both backends
    let kernel = KernelBackend::init("/dev/akida0").expect("Failed to init kernel backend");
    let userspace =
        UserspaceBackend::init("0000:01:00.0").expect("Failed to init userspace backend");

    // Verify capabilities match
    let kernel_caps = kernel.capabilities();
    let userspace_caps = userspace.capabilities();

    assert_eq!(
        kernel_caps.chip_version, userspace_caps.chip_version,
        "Chip version mismatch"
    );
    assert_eq!(
        kernel_caps.npu_count, userspace_caps.npu_count,
        "NPU count mismatch"
    );
    assert_eq!(
        kernel_caps.memory_mb, userspace_caps.memory_mb,
        "Memory size mismatch"
    );
}

/// Test that both backends produce identical inference results
#[test]
#[ignore] // Requires actual Akida hardware and model
fn test_backend_inference_parity() {
    // Create a simple test pattern (not a real model)
    let model_data = vec![0u8; 1024]; // Placeholder for model data

    // Initialize both backends
    let mut kernel = KernelBackend::init("/dev/akida0").expect("Failed to init kernel backend");
    let mut userspace =
        UserspaceBackend::init("0000:01:00.0").expect("Failed to init userspace backend");

    // Load model on both
    kernel
        .load_model(&model_data)
        .expect("Kernel failed to load model");
    userspace
        .load_model(&model_data)
        .expect("Userspace failed to load model");

    // Create test input
    let input = vec![0.5f32; 784]; // MNIST-sized input

    // Run inference on both
    let kernel_output = kernel.infer(&input).expect("Kernel inference failed");
    let userspace_output = userspace.infer(&input).expect("Userspace inference failed");

    // Verify outputs match (allow small floating-point error)
    assert_eq!(
        kernel_output.len(),
        userspace_output.len(),
        "Output size mismatch"
    );
    for (i, (k, u)) in kernel_output
        .iter()
        .zip(userspace_output.iter())
        .enumerate()
    {
        let diff = (k - u).abs();
        assert!(
            diff < 1e-5,
            "Output mismatch at index {i}: kernel={k}, userspace={u}, diff={diff}"
        );
    }
}

/// Test that backend selection logic is correct
#[test]
fn test_backend_selection() {
    // Auto selection should return a backend
    let result = select_backend(BackendSelection::Auto, "/dev/akida0");
    match result {
        Ok(backend) => {
            // Should get a valid backend type
            let backend_type = backend.backend_type();
            println!("Auto selected backend: {backend_type:?}");
        }
        Err(_) => {
            // OK if no hardware present
            println!("No Akida hardware detected (expected in most environments)");
        }
    }

    // Kernel selection should try kernel backend
    let result = select_backend(BackendSelection::Kernel, "/dev/akida0");
    if std::path::Path::new("/dev/akida0").exists() {
        assert!(
            result.is_ok(),
            "Kernel backend should work when /dev/akida0 exists"
        );
    } else {
        assert!(
            result.is_err(),
            "Kernel backend should fail when /dev/akida0 missing"
        );
    }
}

/// Test that userspace backend gracefully handles missing hardware
#[test]
fn test_userspace_missing_hardware() {
    // Try to init with invalid PCIe address
    let result = UserspaceBackend::init("0000:ff:ff.f");
    assert!(
        result.is_err(),
        "Should fail gracefully on missing hardware"
    );
}

/// Test that kernel backend gracefully handles missing device
#[test]
fn test_kernel_missing_device() {
    // Try to init with non-existent device
    let result = KernelBackend::init("/dev/akida999");
    assert!(result.is_err(), "Should fail gracefully on missing device");
}

/// Test reservoir computing with both backends
#[test]
#[ignore] // Requires actual Akida hardware
fn test_reservoir_parity() {
    // Create small test reservoir (flattened arrays)
    let w_in = vec![1.0f32; 100 * 784];
    let w_res = vec![0.0f32; 100 * 100];

    // Initialize both backends
    let mut kernel = KernelBackend::init("/dev/akida0").expect("Failed to init kernel backend");
    let mut userspace =
        UserspaceBackend::init("0000:01:00.0").expect("Failed to init userspace backend");

    // Load reservoir on both
    kernel
        .load_reservoir(&w_in, &w_res)
        .expect("Kernel failed to load reservoir");
    userspace
        .load_reservoir(&w_in, &w_res)
        .expect("Userspace failed to load reservoir");

    // Create test input
    let input = vec![0.5f32; 784];

    // Run inference on both
    let kernel_output = kernel.infer(&input).expect("Kernel inference failed");
    let userspace_output = userspace.infer(&input).expect("Userspace inference failed");

    // Verify outputs match
    assert_eq!(kernel_output.len(), userspace_output.len());
    for (k, u) in kernel_output.iter().zip(userspace_output.iter()) {
        assert!((k - u).abs() < 1e-5);
    }
}
