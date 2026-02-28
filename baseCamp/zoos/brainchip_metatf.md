# BrainChip MetaTF Model Zoo

**URL:** https://doc.brainchipinc.com/api_reference/cnn2snn.html
**Access:** Python `akida.AkidaNet`, `MetaTF`, `QuantizeML`, `CNN2SNN`
**License:** BrainChip proprietary (models free for evaluation, check commercial terms)
**Status:** Reviewed February 2026; source for `.fbz` files used in rustChip benchmarks

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

## Model Catalog

| Model | Task | Input | Accuracy | Program size | AKD1000 latency |
|-------|------|-------|----------|-------------|----------------|
| **AkidaNet 0.5** | ImageNet classification | 160×160×3 | top-1 65.6% | ~400 KB | ~800 µs |
| **AkidaNet 1.0** | ImageNet classification | 224×224×3 | top-1 70.6% | ~1.6 MB | ~2.1 ms |
| **DS-CNN** | Keyword spotting (35 words) | 49×10 MFCC | 93.8% | ~280 KB | ~700 µs |
| **MobileNetV1 0.25** | ImageNet | 128×128×3 | top-1 47.4% | ~180 KB | ~600 µs |
| **MobileNetV2** | ImageNet | 224×224×3 | top-1 68.2% | ~880 KB | ~1.4 ms |
| **ResNet** | ImageNet | 224×224×3 | top-1 72.4% | ~2.1 MB | ~3.2 ms |
| **VGG-like** | CIFAR-10 | 32×32×3 | 92.7% | ~240 KB | ~680 µs |
| **YOLO v2** | Object detection | 224×224×3 | mAP 42.4% | ~1.8 MB | ~2.8 ms |
| **Face recognition** | LFW verification | 112×112×3 | 99.1% LFW | ~1.2 MB | ~1.9 ms |
| **DVS Gesture** | Event-based gesture | 64×64×2 | 97.9% | ~120 KB | ~580 µs |
| **Occlusion Detection** | Traffic/safety | 320×240×3 | mAP 0.28 | ~1.6 MB | ~2.3 ms |

*Latency estimates from MetaTF benchmarks; may vary from rustChip measurements.*

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

BrainChip distributes pre-compiled `.fbz` files through:

```python
# Python (requires akida SDK, not Rust)
from akida_models import AkidaNetModel
model = AkidaNetModel(classes=1000)
model.save("akidanet.fbz")
```

Or download from: https://github.com/Brainchip-Inc/akida_examples

These `.fbz` files can then be loaded directly by `akida-models`:

```bash
# Using akida-cli (after downloading .fbz)
akida info 0  # verify hardware
cargo run --example load_to_device -- --model ds_cnn_kws.fbz
```
