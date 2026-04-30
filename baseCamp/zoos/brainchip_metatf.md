# BrainChip MetaTF Model Zoo

**URL:** https://doc.brainchipinc.com/api_reference/cnn2snn.html
**Access:** Python `akida.AkidaNet`, `MetaTF`, `QuantizeML`, `CNN2SNN`
**License:** BrainChip proprietary (models free for evaluation, check commercial terms)
**Status:** Fully exported April 2026 — 21 models converted to `.fbz`, all parse with rustChip

---

## Overview

BrainChip's MetaTF ecosystem produces compiled Akida model binaries (`.fbz` files)
from standard neural network architectures. The pipeline is:

```
Keras/TensorFlow model
  → CNN2SNN (architecture conversion to SNN)
    → QuantizeML (quantize to int4/int8)
      → Compile → .fbz binary
        → akida.Model.map(device) → AKD1000 inference
```

The `.fbz` files are FlatBuffer binaries (see `akida-chip/src/program.rs`).
rustChip's `akida-models` crate parses these directly — bypassing the entire
Python stack.

---

## Model Catalog (exported April 2026, Akida SDK 2.19.1)

All 21 models below are exported via `scripts/export_zoo.py` and parse with
`akida-models::Model::from_file()`. Run `cargo run --bin akida -- zoo-status`
to see live cache status.

| Model | Task | Input | .fbz size | Rust parse |
|-------|------|-------|-----------|------------|
| **AkidaNet ImageNet** | Classification | 224×224×3 | 5,269 KB | OK |
| **AkidaNet18 ImageNet** | Classification | 224×224×3 | 2,827 KB | OK |
| **AkidaNet PlantVillage** | Disease classification | 224×224×3 | 1,402 KB | OK |
| **AkidaNet VWW** | Visual Wake Words | 96×96×3 | 304 KB | OK |
| **AkidaNet FaceID** | Face identification | 112×96×3 | 2,834 KB | OK |
| **Akida UNet Portrait** | Segmentation | 128×128×3 | 1,302 KB | OK |
| **CenterNet VOC** | Object detection | 384×384×3 | 2,864 KB | OK |
| **ConvTiny Gesture** | DVS gesture | 64×64×10 | 174 KB | OK |
| **ConvTiny Handy Samsung** | Gesture | 120×160×2 | 172 KB | OK |
| **DS-CNN KWS** | Keyword spotting | 49×10×1 | 41 KB | OK |
| **GXNOR MNIST** | Digit classification | 28×28×1 | 203 KB | OK |
| **MobileNet ImageNet** | Classification | 224×224×3 | 5,028 KB | OK |
| **PointNet++ ModelNet40** | 3D point cloud | 8×256×3 | 343 KB | OK |
| **TENN Recurrent SC12** | Speech commands | 1×256×1 | 70 KB | OK |
| **TENN Recurrent UORED** | Audio | 1×256×1 | 37 KB | OK |
| **TENN Spatiotemporal DVS128** | DVS gesture | 128×128×2 | 225 KB | OK |
| **TENN Spatiotemporal Eye** | Eye tracking | 80×106×2 | 275 KB | OK |
| **TENN Spatiotemporal Jester** | Video gesture | 100×100×3 | 1,611 KB | OK |
| **VGG UTK Face** | Age regression | 32×32×3 | 150 KB | OK |
| **YOLO VOC** | Object detection | 224×224×3 | 4,368 KB | OK |
| **YOLO WiderFace** | Face detection | 224×224×3 | 4,239 KB | OK |

2 models not available for Akida v2: `akidanet_edge_imagenet`, `akidanet_faceidentification_edge`.

---

## Rust Integration Analysis

### Direct `.fbz` loading (works today)

Any model compiled to `.fbz` via MetaTF can be loaded by `akida-models`:

```rust
use akida_models::Model;

// Load a MetaTF-compiled .fbz
let model = Model::from_file("ds_cnn_kws.fbz")?;
println!("program_info: {} bytes", model.program_info().len());
println!("program_data: {} bytes", model.program_data().len());
println!("layers: {}", model.layer_count());

// Load onto hardware
let mut device = DeviceManager::discover()?.open_first()?;
device.write(model.program_info())?;
// DMA program_data to IOVA, then program_external()
```

### What MetaTF does that we don't (yet) replace

| MetaTF capability | rustChip status | Path |
|------------------|-----------------|------|
| Keras → SNN conversion (CNN2SNN) | ❌ Python only | Third-party (`snnix` crate, queued) |
| Float → int4 quantization (QuantizeML) | ❌ Python only | `akida-models` quantization module (queued) |
| Architecture compilation → .fbz | ❌ Python only | `program_external()` for hand-built |
| Weight loading (set_variable) | ✅ Rust | `akida-driver::InferenceExecutor::update_weights()` |
| model.map() → hardware | ✅ Rust | `akida-driver::DeviceManager::open_first()` |
| Batch inference | ✅ Rust | `InferenceConfig { batch_size: 8 }` |
| Power measurement | ✅ Rust | `caps.power_mw` via hwmon |
| Clock mode selection | ✅ Rust | `akida-driver::ClockMode` |

### The gap that matters

We can load and run any `.fbz` file. We cannot currently *produce* `.fbz` files
from an arbitrary Keras model without MetaTF. The path to close this gap:

1. **Short term:** Ship pre-compiled `.fbz` files for common models
2. **Medium term:** Implement `akida-models::builder` — hand-build programs
   using `program_external()` format (already reverse-engineered)
3. **Long term:** Implement Rust quantization (`akida-models::quantize`) —
   given any float weight matrix, produce int4 weights + program binary

---

## Models Relevant to ecoPrimals

| MetaTF model | ecoPrimals domain | Extension |
|-------------|-------------------|-----------|
| ESN variants | All springs (reservoir computing) | Already extended — see `models/physics/` |
| DS-CNN KWS | wetSpring sentinel | Acoustic anomaly detection |
| DVS Gesture | groundSpring metalForge | Hardware event capture benchmarks |
| AkidaNet | neuralSpring | Transfer learning for physics spectra |

---

## Getting .fbz Files

### Automated export (recommended)

```bash
# One-time setup: Python 3.10 venv as validation oracle
python3.10 -m venv .zoo-venv
source .zoo-venv/bin/activate
pip install akida akida-models quantizeml cnn2snn

# Export all 21 pretrained models
python scripts/export_zoo.py

# Verify with Rust parser
cargo run --bin akida -- zoo-status
cargo run --bin akida -- parse baseCamp/zoo-artifacts/ds_cnn_kws.fbz
```

### Manual download

BrainChip distributes examples at: https://github.com/Brainchip-Inc/akida_examples

Any `.fbz` file can be loaded by `akida-models`:

```bash
cargo run --bin akida -- parse <model.fbz>
```
