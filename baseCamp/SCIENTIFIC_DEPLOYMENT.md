# Deploying a Neuromorphic Processor for Scientific and Data Workloads

**Date:** April 29, 2026
**Audience:** Researchers and engineers considering neuromorphic hardware for
scientific computing, sensor processing, or data pipeline acceleration
**Hardware:** BrainChip AKD1000 (AKD1500 compatible with minor register changes)
**Software:** rustChip — pure Rust, no vendor SDK at runtime

---

## Abstract

This document describes how to deploy a BrainChip Akida neuromorphic processor
for scientific and data workloads using rustChip, a pure Rust driver and model
toolkit. We cover the hardware capabilities, deployment architecture, model
design patterns, and concrete results from four production physics workloads.

The core argument: neuromorphic inference at 17,000–24,000 Hz and 1.0–1.5 µJ
per inference fills a specific gap in scientific computing — the "steering
layer" between expensive simulations and human decision-making. The NPU runs
continuously, classifying simulation states and triggering actions, while the
GPU/CPU handles the heavy computation.

---

## 1. When to Use an NPU

An NPU is the right tool when your workload has these characteristics:

| Characteristic | Why it fits the NPU |
|---------------|---------------------|
| **Small input tensors** (1–1000 floats) | The AKD1000 is optimized for low-dimensional inputs via PCIe |
| **High inference rate** (>1000 Hz) | 17,800–24,000 Hz at batch=8, with 54 µs chip latency |
| **Low energy budget** | 1.0–1.5 µJ per inference; 900 mW board floor |
| **Classification or regression** | Conv+FC architectures map directly to NPs |
| **Streaming or online** | Reservoir computing keeps state across calls |
| **Deterministic** | Same input always produces the same output |
| **Concurrent models** | 7 models simultaneously on one chip (814/1000 NPs) |

An NPU is the **wrong** tool when:

- Your model has >1000 NPs worth of parameters (use a GPU)
- You need attention/transformer layers (no hardware support)
- Your input is a high-resolution image processed end-to-end (GPU is faster)
- You need training on-chip (inference only — train elsewhere, deploy here)

### The Steering Layer Pattern

```
┌───────────────────────────────────────────────────────────┐
│                    Scientific Pipeline                     │
│                                                           │
│  ┌─────────────┐    ┌─────────┐    ┌──────────────────┐  │
│  │ Simulation   │───▶│ NPU     │───▶│ Decision / Action│  │
│  │ (GPU / CPU)  │    │ Steering│    │ (store, alert,   │  │
│  │              │◀───│ Layer   │    │  reconfigure)    │  │
│  └─────────────┘    └─────────┘    └──────────────────┘  │
│                                                           │
│  Heavy compute         Fast classifier    Human-time      │
│  (seconds/sample)      (54 µs/sample)    decisions        │
└───────────────────────────────────────────────────────────┘
```

The simulation produces observables. The NPU classifies them instantly.
The result informs what the simulation does next — or what the operator
sees in a dashboard.

---

## 2. Hardware Setup

### What you need

- BrainChip AKD1000 PCIe card (vendor `0x1e7c`, device `0xbca1`)
- Linux with IOMMU enabled (`intel_iommu=on` or `amd_iommu=on` in kernel params)
- `vfio-pci` kernel module loaded
- Rust toolchain (for building rustChip)

### Installation

```bash
# Clone and build
git clone https://github.com/syntheticChemistry/rustChip
cd rustChip
cargo build --release

# Bind the device to VFIO (one-time, or install udev rule)
cargo run --release --bin akida -- bind-vfio 0000:e2:00.0

# Or install persistent udev rule:
sudo cp specs/99-akida-vfio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Verify hardware access

```bash
cargo run --release --bin akida -- verify

# Expected output:
#   PCIe discovery ... OK (1 device(s))
#     [Akd1000] 0000:e2:00.0 — 80 NPs, 10 MB SRAM
#     IOMMU group 92 — accessible
#   VFIO container  ... present
#   All checks passed. Hardware is ready.
```

No kernel module compilation. No Python. No root access after udev setup.

---

## 3. Designing Models for Scientific Workloads

### Architecture patterns that work

**Pattern 1: Classifier (discrete output)**

```
Observables → InputConv(N→M, 1×1) → FC(M→K)
  N = number of input features
  M = hidden dimension (64–128 typical)
  K = number of classes
```

Used for: phase classification, regime detection, anomaly flagging.

Example: SU(3) phase classifier. 3 observables → 64 features → 2 classes.
67 NPs, 21,200 Hz, 100% test accuracy.

**Pattern 2: Regression (continuous output)**

```
Observables → InputConv(N→M, 1×1) → FC(M→K)
  Same structure, but output is continuous, not argmax'd
```

Used for: transport coefficient prediction, surrogate models.

Example: WDM transport predictor. 6 observables → 128 features → 3 outputs.
134 NPs, 17,800 Hz, 3–4% mean relative error.

**Pattern 3: Reservoir readout (temporal streaming)**

```
Time series → CPU reservoir (sparse W_res) → activations → NPU readout → output
```

The reservoir (Echo State Network) runs on CPU. The readout — the trained
linear layer — runs on the NPU. This splits the workload: CPU handles the
recurrent dynamics; NPU handles the fast classification.

Used for: thermalization detection, chaotic time series, streaming sensors.

Example: ESN thermalization detector. 50-step history → 128 features → 1 score.
179 NPs, 18,500 Hz, 0 false thermalization events in 5,978 calls.

**Pattern 4: Multi-head (shared features, multiple outputs)**

```
Observables → InputConv(N→M) → [FC(M→K₁), FC(M→K₂), ..., FC(M→Kₙ)]
  SkipDMA routes features to multiple FC layers in one pass
```

Used for: simultaneous classification + regression from shared features.

Example: 11-head conductor. 1 reservoir program (179 NPs) → 11 independent
outputs at single-program latency (54 µs total, not 11 × 54 µs).

### Weight quantization

All Akida weights are int4 (4-bit integers, range [-8, 7]). The quantization
recipe:

1. Train your model in float32 (on CPU/GPU, using your framework of choice)
2. Quantize post-training: max-abs scaling per layer
3. Pack into `.fbz` format via the BrainChip SDK or `ProgramBuilder`

```rust
// Max-abs quantization
fn quantize_int4(weights: &[f32]) -> Vec<i8> {
    let scale = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let inv = if scale > 0.0 { 7.0 / scale } else { 0.0 };
    weights.iter().map(|&w| (w * inv).round().clamp(-8.0, 7.0) as i8).collect()
}
```

For scientific workloads, int4 quantization introduces 1–5% error relative
to float32 — acceptable for steering decisions, not for final measurements.

### NP budget planning

The AKD1000 has 1,000 Neural Processors. Each model consumes a fixed number
of NPs based on its architecture. Plan your deployment:

```
NP budget: 1,000
──────────────────────────────────────────────
 Model                           NPs   Cumulative
──────────────────────────────────────────────
 Phase classifier                 67          67
 Anderson classifier              68         135
 Transport predictor             134         269
 ESN thermalization              179         448
 Keyword spotter (DS-CNN)        380         828
──────────────────────────────────────────────
 Spare                           172       1,000
```

Five simultaneous scientific classifiers with 172 NPs to spare.

---

## 4. Production Results

### Lattice QCD thermalization steering (hotSpring Experiment 022)

**Problem:** Monte Carlo simulation requires thermalization — the Markov
chain must reach equilibrium before measurements. Detecting this
traditionally means manual plaquette plot inspection.

**Solution:** ESN readout on AKD1000. Given 50 consecutive plaquette values,
output a thermalization score.

| Metric | Value |
|--------|-------|
| Throughput | 18,500 Hz (batch=8) |
| Latency | 54 µs (chip) |
| Energy | 1.4 µJ/inference |
| Production calls | 5,978 over 24 hours |
| False early stops | 0 |
| Thermalization savings | 63% (skip early configurations) |

The NPU runs continuously alongside the GPU simulation. When the
thermalization score crosses threshold, measurement collection begins.
63% of traditionally-wasted configurations were correctly skipped.

### WDM transport coefficient surrogate (hotSpring Experiment 022)

**Problem:** Transport coefficients (diffusion D*, viscosity η*, thermal
conductivity λ*) require expensive MD simulation. Real-time estimation
is needed for simulation steering.

**Solution:** 6-input, 3-output regression on AKD1000.

| Metric | Value |
|--------|-------|
| Throughput | 17,800 Hz |
| Mean error D* | 3.1% |
| Mean error η* | 3.8% |
| Mean error λ* | 4.2% |

At 17,800 Hz NPU throughput vs the GPU's 60 Hz simulation loop, the
NPU services transport coefficient requests with 99.7% idle time.

### SU(3) phase boundary discovery (hotSpring Experiment 022)

**Problem:** Classify lattice QCD configurations as confined or deconfined
from 3 observables (plaquette, real and imaginary Polyakov loop).

**Solution:** Point-wise InputConv + FC classifier.

| Metric | Value |
|--------|-------|
| Throughput | 21,200 Hz |
| Test accuracy | 100% |
| β_c discovery | 5.69 (matches Bazavov et al.) |

The classifier's confidence gradient reveals the phase boundary: β_c = 5.69
from observables alone, without being given β directly.

### Anderson localization regime (groundSpring Experiment 028)

**Problem:** Classify Anderson localization regime from 4 spectral
observables: level spacing ratio, participation ratio, spectral gap
ratio, DOS curvature.

| Metric | Value |
|--------|-------|
| Throughput | 22,400 Hz |
| Accuracy (localized) | 99.2% |
| Accuracy (diffusive) | 99.1% |
| Accuracy (critical) | 87.3% |
| W_c estimate | 16.26 ± 0.95 |

Lower accuracy at the critical point is physically expected — multifractal
eigenstates genuinely interpolate between phases.

---

## 5. Deployment Patterns

### Pattern A: Sidecar to GPU simulation

The NPU sits alongside a GPU. The simulation produces observables
periodically; the NPU classifies them and feeds results back.

```
GPU simulation loop:
  for step in 0..N:
    evolve_system()
    if step % checkpoint_interval == 0:
      observables = extract_observables()
      result = npu.infer(observables)    // 54 µs
      if result.thermalized():
        begin_measurement()
```

**Latency impact:** 54 µs per NPU call vs seconds per simulation step.
The NPU call is invisible in the simulation timeline.

### Pattern B: Streaming sensor pipeline

The NPU ingests a continuous stream of sensor readings and emits
classifications or anomaly scores.

```
sensor → buffer(50 samples) → NPU classify → alert/store
  ↑                                              ↓
  └──── feedback (reconfigure sensor) ◄──────────┘
```

At 18,500 Hz, the NPU can handle ~18,500 sensor windows per second.
If each window is 50 samples at 1 kHz sampling, that's 50× real-time
processing headroom.

### Pattern C: Multi-model observatory

Load multiple classifiers simultaneously for different scientific domains:

```
Observatory chip (AKD1000, 1000 NPs):
  Slot 0: Phase classifier        67 NPs
  Slot 1: Anderson classifier     68 NPs
  Slot 2: Transport predictor    134 NPs
  Slot 3: ESN readout            179 NPs
  Slot 4: Anomaly sentinel        96 NPs
  ────────────────────────────────
  Total:                         544 NPs (456 spare)
```

Each slot runs independently. A single chip serves 5 different scientific
inference tasks simultaneously with no cross-contamination.

### Pattern D: Online adaptation

Swap model weights at 86 µs via `set_variable()`. Use this for:
- Adapting classifiers to changing experimental conditions
- A/B testing different model versions in real-time
- Evolutionary model search (136 generations/second)

---

## 6. Reproducibility

Every claim in this document is reproducible:

| Claim | Reproduction method |
|-------|---------------------|
| 21 BrainChip models parse | `cargo test --test zoo_regression -- --ignored` |
| 4 physics models parse | Same test suite (25 total) |
| Parser throughput 13.7 MB/s | `benchmark_parse_throughput` test |
| `akida parse` output format | `cargo run --bin akida -- parse <any.fbz>` |
| Zoo export completeness | `python scripts/export_zoo.py` → 21/21 |
| Physics export | `python scripts/export_physics.py` → 4/4 |
| NP budgets fit AKD1000 | `test_np_budget_within_chip_limit` test |
| Physics models co-locate | `test_physics_co_location_fits` test |

Hardware-specific claims (throughput, latency, energy) require a live
AKD1000. The relevant experiments are documented in the spring repos:
hotSpring Exp 022, groundSpring Exp 028.

---

## 7. Comparison to Alternatives

### vs. GPU inference (PyTorch, TensorRT)

| Dimension | GPU | NPU (AKD1000) |
|-----------|-----|----------------|
| Throughput (small model) | ~50,000 Hz | ~20,000 Hz |
| Latency | ~1 ms (kernel launch overhead) | 54 µs |
| Energy/inference | ~100 µJ | 1.4 µJ |
| Idle power | 30–300 W | 0.9 W |
| Concurrent models | 1 (time-sliced) | 7 (NP-isolated) |
| Determinism | Non-deterministic by default | Deterministic always |

The NPU wins on latency, energy, idle power, concurrency, and determinism.
The GPU wins on raw throughput and model complexity. Use both.

### vs. CPU inference (ONNX Runtime, tflite)

| Dimension | CPU | NPU (AKD1000) |
|-----------|-----|----------------|
| Throughput | ~5,000 Hz (small FC) | ~20,000 Hz |
| Energy/inference | ~500 µJ | 1.4 µJ |
| Setup | `pip install onnxruntime` | `cargo build` + VFIO |
| Model flexibility | Any ONNX model | Akida-compatible only |

The NPU is 4× faster and 350× more energy-efficient for compatible models.
The CPU handles arbitrary architectures the NPU cannot.

### vs. FPGA (Xilinx, Intel)

| Dimension | FPGA | NPU (AKD1000) |
|-----------|------|----------------|
| Flexibility | Fully custom | Fixed architecture |
| Development time | Weeks–months (HLS/RTL) | Hours (`.fbz` + cargo build) |
| Energy efficiency | Comparable to NPU | Comparable to FPGA |
| Tooling | Vivado, Quartus | Rust (`cargo build`) |

FPGAs are more flexible but orders of magnitude slower to develop for.
The NPU is a "deploy in a day" option for compatible workloads.

---

## 8. Getting Started

### Minimal scientific deployment (30 minutes)

1. **Build rustChip**
   ```bash
   git clone https://github.com/syntheticChemistry/rustChip
   cd rustChip && cargo build --release
   ```

2. **Bind hardware** (one-time)
   ```bash
   cargo run --release --bin akida -- setup
   cargo run --release --bin akida -- verify
   ```

3. **Export a model** (one-time, requires Python 3.10)
   ```bash
   python3.10 -m venv .zoo-venv && source .zoo-venv/bin/activate
   pip install akida akida-models quantizeml cnn2snn
   python scripts/export_physics.py  # or export_zoo.py
   deactivate
   ```

4. **Inspect**
   ```bash
   cargo run --release --bin akida -- parse baseCamp/zoo-artifacts/phase_classifier.fbz
   ```

5. **Deploy** — integrate `akida-models` and `akida-driver` into your pipeline
   ```toml
   # your Cargo.toml
   [dependencies]
   akida-models = { path = "path/to/rustChip/crates/akida-models" }
   akida-driver = { path = "path/to/rustChip/crates/akida-driver" }
   ```

### Your first custom model

See `baseCamp/conversion/from_scratch.md` for the complete walkthrough.
The minimum viable model is:

1. Train a small FC network on your data (any framework)
2. Export weights as `.npy` or JSON
3. Quantize to int4
4. Build `.fbz` via the BrainChip SDK (one-time Python step)
5. Load in Rust: `Model::from_file("your_model.fbz")`

---

## 9. Licensing

rustChip is scyBorg-licensed:

| Layer | License |
|-------|---------|
| Code | AGPL-3.0-or-later |
| Documentation (this file) | CC-BY-SA-4.0 |
| Game mechanics | ORC |

The symbiotic exception protocol offers hardware partners linking exceptions
in exchange for silicon documentation or hardware access. The driver's
existence demonstrates capability; the open license invites collaboration.

---

## References

- rustChip: https://github.com/syntheticChemistry/rustChip
- BrainChip Akida documentation: https://doc.brainchipinc.com/
- Linux VFIO: https://docs.kernel.org/driver-api/vfio.html
- NeuroBench: https://neurobench.ai/
- ecoPrimals: https://primals.eco
