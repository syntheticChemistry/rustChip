# AkidaNet — ImageNet Classifier

**Architecture:** MobileNet-style SNN, AkidaNet 0.5 (160×160)
**Status:** 📋 Analysis complete; .fbz available from BrainChip
**Task:** ImageNet 1000-class classification
**Source:** BrainChip MetaTF official release

---

## Architecture Overview

AkidaNet is BrainChip's flagship classification model. It's a MobileNetV1-derived
architecture with depthwise separable convolutions, adapted for Akida's
sparse event-based processing.

```
Input: uint8[160×160×3]  (RGB image, int8 normalized)
  │
  ▼
Conv(3→32, stride=2)           ← 160→80
  │
  ▼  ×13 blocks
DWConv(32→32, stride=1|2) + PtConv(32→64→128→...)
  │  progressive channel widening with ×0.5 width multiplier
  ▼
GlobalAveragePooling
  │
  ▼
FC(→1000)
  │  softmax → ImageNet class
  ▼
Output: float[1000]
```

Model variants:
| Variant | Width | Input | Top-1 | Program size | Latency |
|---------|-------|-------|-------|-------------|---------|
| AkidaNet 0.5 | 0.5 | 160×160 | 65.6% | ~400 KB | ~800 µs |
| AkidaNet 1.0 | 1.0 | 224×224 | 70.6% | ~1.6 MB | ~2.1 ms |

---

## Relevance to ecoPrimals

Direct relevance to ecoPrimals domains is limited — the physics springs
don't do ImageNet classification. However:

1. **Field genomics sentinel (Paper 09)**: AkidaNet as feature extractor
   before species classification layer. Use AkidaNet backbone (drop FC head),
   append custom FC(512→N_species).

2. **Transfer learning platform**: AkidaNet 0.5 backbone is the standard
   starting point for custom classification tasks. Fine-tune the FC head on:
   - HAB species (bloom detection, Paper 04)
   - Crop disease (agricultural IoT, Paper 08)
   - Substrate categories (metalForge visual benchmarks)

3. **NP budget baseline**: AkidaNet 0.5 uses ~450 NPs, leaving 550 free.
   Running alongside ECG anomaly + phase classifier is feasible.

---

## Transfer Learning Pattern

```rust
// Sketch: load AkidaNet backbone, swap FC head for custom task
// (Requires akida-models::builder — queued for 0.2)

// Step 1: Load pre-compiled AkidaNet .fbz
let backbone = Model::from_file("akidanet_0.5_backbone.fbz")?;

// Step 2: Build custom head
let head = ProgramBuilder::new()
    .fully_connected(out: n_classes)     // N classes for your domain
    .with_weights_int4(&domain_weights)  // trained on your data
    .compile()?;

// Step 3: Stitch backbone + head
let full_model = backbone.append_head(head)?;

// Step 4: Run
let device = DeviceManager::discover()?.open_first()?;
device.program_external(&full_model.program_info, &full_model.program_data)?;
```

This pattern (frozen backbone + custom head) is how MetaTF supports
transfer learning. rustChip would implement it in `akida-models::builder`.

---

## Quantized Input Note

AkidaNet expects uint8 input (not float32). The Akida int8 input pipeline:

```rust
// Normalize to [0, 255] uint8 before inference
fn normalize_rgb(pixels: &[[f32; 3]]) -> Vec<u8> {
    pixels.iter().flat_map(|p| {
        p.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
    }).collect()
}
```

This is the same normalization used for ECG and ESN int8 inputs —
consistent with the AKD1000's native int8 processing capability.
