# Integration Guide

How to use `rustChip` in a downstream project, including the GPU+NPU
co-location pattern.

---

## 1. Minimal Integration

Add to `Cargo.toml`:

```toml
[dependencies]
akida-driver = { git = "https://github.com/ecoPrimal/rustChip" }
```

Or path-reference if vendored:
```toml
akida-driver = { path = "../rustChip/crates/akida-driver" }
```

Basic usage:

```rust
use akida_driver::{DeviceManager, InferenceExecutor, InferenceConfig};

fn main() -> anyhow::Result<()> {
    let mgr  = DeviceManager::discover()?;
    if mgr.device_count() == 0 {
        println!("No Akida hardware");
        return Ok(());
    }

    let caps = mgr.devices()[0].capabilities();
    println!("Found: {} NPUs, {} MB SRAM", caps.npu_count, caps.memory_mb);

    let mut exec = InferenceExecutor::new(mgr.open_first()?);
    let input    = vec![0.0f32; 50]; // your feature vector
    let result   = exec.run(&input, InferenceConfig::default())?;

    println!("Output: {:?}", result.outputs);
    Ok(())
}
```

---

## 2. GPU + NPU Co-location

The heterogeneous pipeline pattern: GPU computes physics/features, NPU
performs inference on the result. This is the production pattern validated
in Experiment 022 (5,978 live NPU calls in lattice QCD).

```
GPU (your code — wgpu/vulkano/ash/cuda)
  ↓  [GPU result buffer → host memory]
CPU (host buffer — Vec<f32> or &[f32])
  ↓  [akida-driver DMA → NPU SRAM]
NPU (akida-driver — inference)
  ↓  [NPU output → host buffer]
CPU (your application reads result)
```

### Example with wgpu (GPU side yours, NPU side rustChip)

```rust
use akida_driver::{DeviceManager, InferenceExecutor};

// Your GPU computation produces a feature vector (host-readable buffer)
let gpu_features: Vec<f32> = run_gpu_compute(&wgpu_device, &your_shader);

// NPU inference (zero GPU cycles stolen — independent PCIe device)
let npu_mgr  = DeviceManager::discover()?;
let mut exec = InferenceExecutor::new(npu_mgr.open_first()?);
let result   = exec.run(&gpu_features, Default::default())?;

// Combine: GPU produces → NPU classifies → CPU steers
steer_simulation(result.outputs[0]);
```

### Latency budget

```
GPU compute (your workload):     variable — depends on shader complexity
GPU→CPU readback:                ~1–10 µs  (pinned memory + wgpu map_async)
CPU→NPU DMA (37 MB/s):           ~14 µs for 512-float feature vector
NPU inference (54 µs base):      54–400 µs depending on model and batch
NPU→CPU readback:                ~5 µs  (DMA output buffer)
──────────────────────────────────────────────────────────────────────
Total NPU overhead:              ~70–430 µs  (0.07–0.43 ms)
```

For workloads where GPU trajectory takes ~7 s (e.g. lattice QCD HMC), the
NPU overhead is 0.006% of wall time. That's the operating point for Exp 022.

### Connecting to toadStool / barracuda (if you use it)

If your GPU code runs through toadStool's `barracuda` crate:

```rust
// GPU side (your barracuda code — NOT in this repo)
let features = barracuda::run_observable_shader(&device, &config)?;

// NPU side (this repo — standalone)
let mut npu_exec = akida_driver::InferenceExecutor::new(
    DeviceManager::discover()?.open_first()?
);
let phase_label = npu_exec.run(&features, Default::default())?;
```

The interface point is `&[f32]` — a CPU-resident float slice. That's the
seam. GPU codebase produces it; this codebase consumes it.

No import from toadStool needed. No toadStool dep in `Cargo.toml`. The
integration is a runtime data handoff, not a compile-time dependency.

---

## 3. Batch Mode (2.4× throughput)

For throughput-critical workloads, use batch inference:

```rust
use akida_driver::{InferenceConfig, BatchCapabilities};

let batch_size = mgr.devices()[0].capabilities()
    .batch
    .as_ref()
    .map(|b| b.optimal_batch)
    .unwrap_or(1);

let config = InferenceConfig {
    batch_size,
    ..Default::default()
};

// Collect batch_size inputs
let batch_inputs: Vec<f32> = (0..batch_size)
    .flat_map(|_| gpu_features.iter().copied())
    .collect();

let result = exec.run(&batch_inputs, config)?;
// result.outputs has batch_size × output_dim values
```

Reference: Discovery 3 — `batch=8` achieves 390 µs/sample vs 948 µs/sample
at batch=1 (2.4× speedup by amortising PCIe round-trip).

---

## 4. Multiple Classifiers via Weight Mutation

Discovery 6 shows weight mutation (~14 ms overhead) without full reprogram.
This enables running 3 different classifiers by hot-swapping weights:

```rust
// Classifiers: phase, transport, anomaly — loaded from .fbz files
let models = [phase_classifier, transport_predictor, anomaly_detector];

let mut exec = InferenceExecutor::new(mgr.open_first()?);

// Load base program structure (program_info + program_data) once
exec.load_program(&models[0].program_info, &models[0].program_data)?;

for (i, model) in models.iter().enumerate() {
    // Swap weights only (~14 ms vs ~full reprogram)
    exec.update_weights(&model.weights)?;  // set_variable() path

    let result = exec.run(&input_features, Default::default())?;
    println!("Classifier {i}: {:?}", result.outputs);
}
```

At 14 ms per swap, 3 classifiers = 42 ms overhead per HMC trajectory.
At 7 s/trajectory, that's 0.6% overhead for 3× the inference capability.

---

## 5. VFIO Setup (one-time)

For VFIO backend (recommended — no kernel module required):

```bash
# 1. Enable IOMMU in BIOS + kernel
echo "intel_iommu=on iommu=pt" >> /etc/default/grub  # or amd_iommu=on
update-grub && reboot

# 2. Load vfio-pci at boot
echo "vfio-pci" >> /etc/modules-load.d/vfio.conf

# 3. Bind the device (one-time, after first reboot with IOMMU enabled)
sudo cargo run --bin akida -- bind-vfio 0000:a1:00.0  # use your PCIe address

# 4. Persist binding via udev (optional)
echo 'SUBSYSTEM=="pci", ATTR{vendor}=="0x1e7c", ATTR{device}=="0xbca1", \
      RUN+="/bin/sh -c '"'"'echo 1e7c bca1 > /sys/bus/pci/drivers/vfio-pci/new_id'"'"'"' \
    >> /etc/udev/rules.d/99-akida-vfio.rules

# 5. Grant user access
IOMMU_GROUP=$(akida iommu-group 0000:a1:00.0)
sudo chown $USER /dev/vfio/$IOMMU_GROUP
# or: sudo usermod -aG vfio $USER
```

After this one-time setup, `cargo run --bin enumerate` works without root.

---

## 6. What this repo does NOT provide

| Feature | Where to look |
|---------|---------------|
| GPU compute (WGSL shaders) | Your project (wgpu, vulkano, ash) |
| Model training / weight optimization | External (TensorFlow, PyTorch, etc.) |
| Model compilation (QuantizeML + CNN2SNN) | BrainChip SDK (Python MetaTF) |
| Multi-chip routing (multiple AKD1000s) | Planned Phase D extension |
| Python bindings | Not planned — use akida-cli as subprocess |
| Windows support | Not planned (VFIO is Linux-specific) |

The GPU portion of the heterogeneous pipeline — the WGSL lattice QCD shaders,
the BarraCuda physics engine, the heterogeneous dispatch system — lives in
a separate repository. This repo is the NPU half only. The interface is `&[f32]`.
