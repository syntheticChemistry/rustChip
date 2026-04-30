# Integration Guide

How to use `rustChip` in a downstream project, including the GPU+NPU
co-location pattern.

---

## 1. Minimal Integration

Add to `Cargo.toml`:

```toml
[dependencies]
akida-driver = { git = "https://github.com/syntheticChemistry/rustChip" }
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

### Connecting to the sovereign compute trio

rustChip is a standalone extraction from the sovereign compute pipeline.
The full pipeline has three primals that form the node's atomic structure:

```
coralReef (HOW — compile)
  Sovereign GPU compiler: WGSL/SPIR-V/GLSL → native NVIDIA/AMD.
  No LLVM, no Mesa, no vendor SDK.
  VFIO passthrough via ember/glowplug architecture.
      ↓  [compiled shader → toadStool dispatch queue]
toadStool (WHERE — dispatch)
  GPU/NPU/CPU discovery, tolerance-based routing, NpuBackendDispatch.
  Decides which device runs which workload.
      ↓  [dispatch decision → barraCuda or rustChip]
barraCuda (WHAT — compute)                    rustChip (WHAT — infer)
  900+ WGSL shaders, DF64, QCD, FHE.           NPU inference, SRAM, VFIO.
  Produces feature vectors as &[f32].           Consumes &[f32], returns &[f32].
      ↓                                             ↓
  [GPU result → host memory]                   [NPU result → host memory]
      └──────────── merge on CPU ───────────────────┘
```

For ecosystem context, see [primals.eco](https://primals.eco) and
[specs/EVOLUTION.md](EVOLUTION.md).

**Integration with barraCuda** — if your GPU code runs barraCuda shaders:

```rust
// GPU side (barraCuda code — NOT in this repo)
let features = barracuda::run_observable_shader(&device, &config)?;

// NPU side (this repo — standalone)
let mut npu_exec = akida_driver::InferenceExecutor::new(
    DeviceManager::discover()?.open_first()?
);
let phase_label = npu_exec.run(&features, Default::default())?;
```

**Integration with coralReef** — if you use coralReef's sovereign compiler
instead of vendor GPU drivers, the VFIO passthrough patterns are shared.
coralReef's `ember`/`glowplug` architecture (fd sharing for HMB2-era GPU
BAR access) is the same pattern rustChip uses for NPU BAR mapping.
Downstream projects that already use coralReef's VFIO container can share
the IOMMU context with rustChip's VFIO backend:

```rust
// coralReef opens the VFIO container for GPU access
let container_fd = coralreef::vfio::open_container()?;

// rustChip can join the same IOMMU context (future: shared container API)
// Today: independent containers, same VFIO pattern, same udev rules
let npu_backend = akida_driver::select_backend(
    BackendSelection::Vfio,
    "0000:e2:00.0",
)?;
```

**Integration with toadStool** — toadStool's `NpuBackendDispatch` already
understands how to route work to NPU backends. rustChip's `NpuBackend`
trait is extracted from toadStool's — they are ABI-compatible by design.

The interface point is always `&[f32]` — a CPU-resident float slice. That's
the seam. No import from any trio member is needed in `Cargo.toml`. The
integration is a runtime data handoff, not a compile-time dependency.

### Using `test-mocks` for downstream integration tests

If you depend on `akida-driver` and want to run integration tests without
hardware or the full `SoftwareBackend` simulation, enable the `test-mocks`
feature:

```toml
[dev-dependencies]
akida-driver = { git = "https://github.com/syntheticChemistry/rustChip", features = ["test-mocks"] }
```

This exposes `SyntheticNpuBackend` — a minimal deterministic mock that
implements `NpuBackend`, always returns input as output, and reports ready
immediately. It is suitable for testing dispatch logic and data plumbing,
not numerical correctness.

```rust
use akida_driver::SyntheticNpuBackend;

let mut backend = SyntheticNpuBackend::coverage_default();
let output = backend.infer(&[1.0, 2.0, 3.0])?;
assert_eq!(output, vec![1.0, 2.0, 3.0]); // identity — input == output
```

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

## 6. SRAM Access for Diagnostics and Online Learning

rustChip provides direct read/write access to all on-chip SRAM. Use this for
model verification, weight inspection, online learning, and hardware diagnostics.

### Model load verification

After loading a model, verify it was written correctly via SRAM readback:

```rust
use akida_driver::ModelLoader;

let mut loader = ModelLoader::new(backend);
loader.load(&model_bytes)?;

// Verify: reads back SRAM and compares to expected bytes
let verification = loader.verify_with_sram(&model_bytes)?;
assert!(verification.matches);
println!("Verified {} bytes, {} mismatches", verification.bytes_checked, verification.mismatches);
```

### Direct weight mutation (zero-DMA)

For small weight patches (online learning, evolution), skip DMA entirely:

```rust
// Patch 256 bytes of weights at a specific SRAM offset
backend.mutate_weights(weight_offset, &new_weight_bytes)?;

// Run inference with the mutated weights — no reprogram needed
let result = backend.infer(&input)?;
```

### Raw SRAM inspection

```rust
// Read 4 KB of SRAM at a specific offset
let data = backend.read_sram(0x1000, 4096)?;

// Or use SramAccessor for userspace access (no VFIO needed)
let mut sram = SramAccessor::open("0000:a1:00.0")?;
let bar0_regs = sram.read_register(0x0)?;        // device ID
let np_data = sram.read_bar1(np_offset, 1024)?;   // NP SRAM contents
```

### probe_sram binary

For interactive SRAM exploration:

```bash
cargo run --bin probe_sram              # BAR0 register dump + BAR1 probe
cargo run --bin probe_sram -- scan      # find all non-zero data in BAR1
cargo run --bin probe_sram -- test      # write/readback test (destructive)
```

---

## 7. What this repo does NOT provide

| Feature | Where to look |
|---------|---------------|
| GPU compute (WGSL shaders) | Your project (wgpu, vulkano, ash) |
| Model training / weight optimization | External (TensorFlow, PyTorch, etc.) |
| Model compilation (QuantizeML + CNN2SNN) | BrainChip SDK (Python MetaTF) |
| Multi-chip routing (multiple AKD1000s) | Planned Phase D extension |
| Python bindings | Not planned — use akida-cli as subprocess |
| Windows support | Not planned (VFIO is Linux-specific) |

The GPU portion of the heterogeneous pipeline lives in the sovereign compute trio:
[barraCuda](https://github.com/ecoPrimals/barraCuda) (WGSL shaders and math),
[coralReef](https://github.com/ecoPrimals/coralReef) (GPU compiler),
[toadStool](https://github.com/ecoPrimals/toadStool) (dispatch and routing).
This repo is the NPU half only. The interface is `&[f32]`.
