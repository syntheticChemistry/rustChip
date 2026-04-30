# The Akida Model Zoo — A Practical Guide

**Date:** April 30, 2026
**Status:** 29 models (25 exported + 4 Rust-native), parsed and regression-tested in pure Rust.
            Pure Rust conversion pipeline: import, quantize, serialize, compress — no Python.

---

## What the Zoo Is

The Akida model zoo is the complete set of neural network models that can
run on BrainChip's AKD1000/AKD1500 neuromorphic processors. It spans three
sources:

| Source | Models | Coverage |
|--------|--------|----------|
| BrainChip MetaTF | 21 pretrained | Vision, audio, gesture, detection, segmentation, 3D |
| ecoPrimals physics | 4 validated | Lattice QCD, plasma transport, Anderson localization |
| NeuroBench | 2 reference | Chaotic time series, ECG anomaly |

rustChip can parse every model in this zoo without Python, without the
vendor SDK, and without a kernel module.

### What "Parse" Means

```
.fbz file on disk
  → Snappy decompression (with zero-padding linear probe)
    → FlatBuffer extraction (program_info + program_data)
      → Layer graph, weight blocks, version, structure
        → Ready for hardware load via VFIO
```

`Model::from_file("anything.fbz")` does this in 0.6–320 ms depending on
model size. The parser is tested against all 25 zoo models on every commit
(`cargo test --test zoo_regression -- --ignored`).

---

## How to Use the Zoo

### Quick start: inspect a model

```bash
# Parse and display structure of any .fbz file
cargo run --bin akida -- parse baseCamp/zoo-artifacts/ds_cnn_kws.fbz

# Output:
#   Akida Model: baseCamp/zoo-artifacts/ds_cnn_kws.fbz
#   ============================================================
#   File size         : 41360 bytes (40.39 KB)
#   Decompressed      : 35424 bytes (34.59 KB)
#   SDK version       : 2.19.1
#   Layers            : 9
#   Weight blocks     : 1
#   Parse time        : 3.844ms
```

### Quick start: check what's cached

```bash
# Show status of all 28 registered models
cargo run --bin akida -- zoo-status
```

### Exporting the full zoo (one-time, requires Python 3.10)

The Python SDK is used as a validation oracle — it generates the `.fbz`
files that prove the Rust parser works. Once exported, the Python environment
is disposable.

```bash
python3.10 -m venv .zoo-venv
source .zoo-venv/bin/activate
pip install akida akida-models quantizeml cnn2snn
python scripts/export_zoo.py
python scripts/export_physics.py
deactivate  # done with Python
```

This produces 25 `.fbz` files in `baseCamp/zoo-artifacts/` and two JSON
manifests with ground-truth metadata.

### Using a model in Rust

```rust
use akida_models::prelude::*;

// Parse the model
let model = Model::from_file("baseCamp/zoo-artifacts/yolo_voc.fbz")?;
println!("version: {}", model.version());
println!("layers: {}", model.layer_count());
println!("weights: {} blocks", model.weights().len());

// On hardware (requires AKD1000 + VFIO setup)
let mgr = akida_driver::DeviceManager::discover()?;
let device = mgr.open_first()?;
// ... load model data to device via program_external()
```

---

## The 21 BrainChip Models

Every model below was exported from the Akida Python SDK v2.19.1, parsed
by rustChip's FBZ parser, and verified in the regression test suite.

### Image Classification

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| AkidaNet ImageNet (1.0) | 224×224×3 | 5,269 KB | [224,224,3] → [1,1,1000] |
| AkidaNet18 ImageNet | 224×224×3 | 2,827 KB | [224,224,3] → [1,1,1000] |
| AkidaNet PlantVillage | 224×224×3 | 1,402 KB | [224,224,3] → [1,1,38] |
| AkidaNet VWW | 96×96×3 | 304 KB | [96,96,3] → [1,1,2] |
| MobileNet ImageNet | 224×224×3 | 5,028 KB | [224,224,3] → [1,1,1000] |
| GXNOR MNIST | 28×28×1 | 203 KB | [28,28,1] → [1,1,10] |

### Face Analysis

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| AkidaNet FaceID | 112×96×3 | 2,834 KB | [112,96,3] → [1,1,10575] |
| VGG UTK Face | 32×32×3 | 150 KB | [32,32,3] → [1,1,1] |

### Object Detection

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| CenterNet VOC | 384×384×3 | 2,864 KB | [384,384,3] → [96,96,24] |
| YOLO VOC | 224×224×3 | 4,368 KB | [224,224,3] → [7,7,125] |
| YOLO WiderFace | 224×224×3 | 4,239 KB | [224,224,3] → [7,7,18] |

### Segmentation

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| Akida UNet Portrait | 128×128×3 | 1,302 KB | [128,128,3] → [128,128,1] |

### Gesture and Video

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| ConvTiny Gesture (DVS) | 64×64×10 | 174 KB | [64,64,10] → [1,1,10] |
| ConvTiny Handy Samsung | 120×160×2 | 172 KB | [120,160,2] → [1,1,9] |
| TENN ST DVS128 | 128×128×2 | 225 KB | [128,128,2] → [1,1,10] |
| TENN ST Eye Tracking | 80×106×2 | 275 KB | [80,106,2] → [3,4,3] |
| TENN ST Jester | 100×100×3 | 1,611 KB | [100,100,3] → [1,1,27] |

### Audio and Speech

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| DS-CNN KWS | 49×10×1 | 41 KB | [49,10,1] → [1,1,33] |
| TENN Recurrent SC12 | 1×256×1 | 70 KB | [1,256,1] → [1,1,12] |
| TENN Recurrent UORED | 1×256×1 | 37 KB | [1,256,1] → [1,1,4] |

### 3D Point Cloud

| Model | Input | .fbz size | I/O shapes |
|-------|-------|-----------|------------|
| PointNet++ ModelNet40 | 8×256×3 | 343 KB | [8,256,3] → [1,1,40] |

---

## The 4 Physics Models

These are custom architectures developed across ecoPrimals springs and
validated on live AKD1000 hardware. They demonstrate that rustChip handles
bespoke scientific models, not just the vendor's pretrained catalog.

| Model | Spring | Architecture | Throughput | Validated |
|-------|--------|-------------|------------|-----------|
| ESN readout | hotSpring | Conv(1→64→128)+FC(128→1) | 18,500 Hz | Exp 022 (5,978 calls) |
| Phase classifier | hotSpring | Conv(3→64)+FC(64→2) | 21,200 Hz | Exp 022 (100%) |
| Transport predictor | hotSpring | Conv(6→128)+FC(128→3) | 17,800 Hz | Exp 022 |
| Anderson classifier | groundSpring | Conv(4→64)+FC(64→3) | 22,400 Hz | Exp 028 |

Reproduce with: `python scripts/export_physics.py`

Detailed architecture documentation: `baseCamp/models/physics/`

---

## Rust-Native Models (No Python Required)

These models are generated entirely by the pure Rust pipeline (`akida convert`).
No Python SDK, no TensorFlow, no vendor toolchain. Weights are synthetic
(random initialization) — the architectures match real deployment targets.

| Model | Architecture | Purpose | .fbz size | Generated by |
|-------|-------------|---------|-----------|-------------|
| ESN Multi-Head (3-out) | InputConv(50)→FC(128)→FC(1) | Multi-observable readout | 3.6 KB | `akida convert` |
| ESN 3-Head Transport | InputConv(50)→FC(64)→FC(3) | D*/eta*/lambda* prediction | 2.0 KB | `akida convert` |
| Streaming Sensor 12ch | InputConv(1)→FC(256)→FC(128)→FC(12) | 12-channel sensor fusion | 13.4 KB | `akida convert` |
| Adaptive Sentinel | InputConv(64)→FC(128)→FC(1) | Domain-shift drift detection | 4.6 KB | `akida convert` |

Generate any of these yourself:

```bash
# ESN multi-head readout
cargo run -p akida-cli -- convert \
  --weights "random:6400" \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output my_esn.fbz --bits 4

# Streaming sensor fusion
cargo run -p akida-cli -- convert \
  --weights "random:12800" \
  --arch "InputConv(1,1,1) FC(256) FC(128) FC(12)" \
  --output my_sensor.fbz --bits 8

# Or with real weights from .npy or .safetensors:
cargo run -p akida-cli -- convert \
  --weights trained_weights.npy \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output production_model.fbz --bits 4
```

The Rust-native models prove the pipeline works end-to-end without any Python
dependency. To train real weights, use your framework of choice (PyTorch, JAX,
SNNTorch), export to `.npy` or `.safetensors`, and convert in Rust.

---

## Extension into the Rust Ecosystem

### What Rust Gives You That Python Doesn't

**1. No runtime dependency chain.** The Python SDK requires TensorFlow,
MetaTF, QuantizeML, CNN2SNN, NumPy, h5py, and 40+ transitive packages.
rustChip needs `cargo build`.

**2. Compile-time guarantees.** The `ZooModel` enum exhaustively lists every
model variant. Add a new model and the compiler forces you to handle it in
`filename()`, `np_budget()`, `source()`, `task()`, and every other method.
Python discovers missing handler functions at runtime.

**3. No GIL.** Parse 25 models in parallel with Rayon. The Python SDK is
single-threaded per process.

**4. Embeddable.** `akida-models` is a regular Rust library. It compiles
to any target `rustc` supports — embedded Linux, WASM (for model inspection
in the browser), cross-compiled ARM64 for edge nodes.

**5. VFIO without kernel modules.** The driver uses VFIO passthrough
(`/dev/vfio/`), which works on any Linux kernel with IOMMU. No rebuilding
`akida_pcie.ko` when your kernel updates.

### What This Unlocks

**Model inspection tooling.** Build Rust CLI tools, web services, or
desktop apps that parse `.fbz` files without installing 3 GB of Python
packages. The `akida parse` command is an example — it runs in 250 ms
including cold start.

**Heterogeneous compute pipelines.** Combine GPU math (`barraCuda`),
NPU inference (`rustChip`), and CPU orchestration in a single Rust binary.
No FFI, no subprocess spawning, no serialization between languages.

```
barraCuda shader → &[f32] → rustChip NPU → &[f32] → application
```

**Multi-model orchestration.** Load 7 models simultaneously on one AKD1000
(814 of 1,000 NPs). The `MultiTenantDevice` API manages NP allocation,
isolation verification, and per-slot inference. This is not possible through
the Python SDK's single-model `model.map(device)` interface.

**Online weight evolution.** The `NpuEvolver` generates new model variants
at 136 generations/second by mutating weights via `set_variable()` (86 µs
per swap). This requires tight hardware control that Python's GIL and
SDK abstraction make impractical.

**Continuous deployment.** Ship a self-contained binary that discovers
hardware, loads models, and runs inference. No Python virtualenv, no
package manager, no version conflicts between TensorFlow and NumPy.

---

## Adding Your Own Models

### From an existing .fbz file

```rust
let model = Model::from_file("your_model.fbz")?;
```

Done. Any `.fbz` produced by the BrainChip SDK, by `cnn2snn.convert()`,
or by `scripts/export_zoo.py` will parse.

### From PyTorch, JAX, or SNNTorch weights

Export your trained weights to `.npy` or `.safetensors`, then convert entirely in Rust:

```bash
# From .npy weights
cargo run -p akida-cli -- convert \
  --weights my_model_weights.npy \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output my_model.fbz --bits 4

# From .safetensors weights
cargo run -p akida-cli -- convert \
  --weights checkpoint.safetensors \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output my_model.fbz --bits 4
```

The conversion pipeline handles quantization (symmetric per-layer int1/2/4/8),
nibble packing, FlatBuffer serialization, and Snappy compression. Round-trip
verification is automatic.

See `baseCamp/conversion/from_pytorch.md` and `from_snntorch.md` for
framework-specific weight export instructions.

### From scratch in Rust

Build models programmatically with the `ProgramBuilder`:

```rust
use akida_models::builder::ProgramBuilder;

let mut builder = ProgramBuilder::new();
builder.add_input_conv(50, 128, 3);
builder.add_fc(128, 1);
let binary = builder.build();  // returns serialized .fbz bytes
```

Or use `program_external()` to inject hand-built FlatBuffer programs directly.

---

## Validation Status

| Check | Status |
|-------|--------|
| All 21 BrainChip models exported | `scripts/export_zoo.py` — 21/21 |
| All 4 physics models exported | `scripts/export_physics.py` — 4/4 |
| 4 Rust-native models generated | `akida convert` — 4/4 |
| Rust parser handles every .fbz | `zoo_regression.rs` — 25/25 + 4 Rust-native |
| Layer counts non-zero | All 29 models |
| Decompression ratios sane | All 29 models |
| Parse throughput > 1 MB/s | 13.7 MB/s measured |
| `akida parse` works for any .fbz | Tested on all 29 |
| `akida zoo-status` matches enum | 25/28 cached (BrainChip/physics) |
| `ZooModel` enum covers all exports | 28 variants, 25 with artifacts |
| `akida convert` round-trip verified | 4/4 Rust-native models |
| Pure Rust pipeline end-to-end | import → quantize → serialize → compress |

---

## Further Reading

| Topic | Document |
|-------|----------|
| Individual model architectures | `baseCamp/models/physics/` and `models/edge/` |
| Conversion from other frameworks | `baseCamp/conversion/` |
| Zoo source analysis | `baseCamp/zoos/brainchip_metatf.md` |
| NeuroBench benchmarks | `baseCamp/zoos/neurobench.md` |
| SNN framework compatibility | `baseCamp/zoos/snntorch.md` |
| Multi-model deployment | `baseCamp/systems/multi_tenancy.md` |
| Scientific deployment guide | `baseCamp/SCIENTIFIC_DEPLOYMENT.md` |
| Rust NPU ecosystem analysis | `baseCamp/RUST_NPU_ECOSYSTEM.md` |
| Spring NPU workloads | `baseCamp/spring-profiles/README.md` |
