# Vision — Detection, Segmentation, Classification, Face Analysis

**Spring origin:** Zoo (BrainChip pretrained models)
**Maturity:** Models parsed and validated; full I/O shapes documented
**Zoo models:** YOLO VOC/WiderFace, CenterNet VOC, UNet Portrait, AkidaNet variants, PointNet++

---

## Problem

Visual perception at the edge: cameras produce continuous image streams that
must be analyzed in real time — detect objects, segment regions, classify
scenes, identify faces — all without cloud connectivity, with low power, and
at frame-rate latency.

The Akida architecture processes quantized convolutional features natively.
The zoo contains 14 vision models spanning five sub-domains, from 203 KB
(GXNOR MNIST) to 5.3 MB (AkidaNet ImageNet).

| Task | Question | Zoo model | Input |
|------|----------|-----------|-------|
| **Object detection** | Where are the objects? What class? | YOLO VOC, CenterNet VOC | 224×224×3, 384×384×3 |
| **Face detection** | Where are faces? Who is it? | YOLO WiderFace, AkidaNet FaceID | 224×224×3, 112×96×3 |
| **Segmentation** | Pixel-level foreground/background | UNet Portrait 128 | 128×128×3 |
| **Classification** | What is this? (1000 classes) | AkidaNet, MobileNet | 224×224×3 |
| **Specialized classification** | Plant disease? Age estimate? | PlantVillage, VGG UTK Face | 224×224×3, 32×32×3 |
| **3D point cloud** | What is this 3D shape? | PointNet++ ModelNet40 | 8×256×3 |

---

## Model Catalog

### Object Detection

| Model | Input | Output | .fbz | Notes |
|-------|-------|--------|------|-------|
| YOLO VOC | 224×224×3 | 7×7×125 (grid + 5 anchors × 25) | 4,368 KB | Pascal VOC 20-class |
| YOLO WiderFace | 224×224×3 | 7×7×18 (grid + faces) | 4,239 KB | Face detection |
| CenterNet VOC | 384×384×3 | 96×96×24 (heatmap + offset + size) | 2,864 KB | Anchor-free detection |

### Segmentation

| Model | Input | Output | .fbz | Notes |
|-------|-------|--------|------|-------|
| UNet Portrait 128 | 128×128×3 | 128×128×1 (mask) | 1,302 KB | Portrait foreground/background |

### Classification

| Model | Input | Output | .fbz | Notes |
|-------|-------|--------|------|-------|
| AkidaNet ImageNet 1.0 | 224×224×3 | 1×1×1000 | 5,269 KB | ImageNet top-1 |
| AkidaNet18 ImageNet | 224×224×3 | 1×1×1000 | 2,827 KB | Lighter variant |
| MobileNet ImageNet | 224×224×3 | 1×1×1000 | 5,028 KB | MobileNet architecture |
| AkidaNet PlantVillage | 224×224×3 | 1×1×38 | 1,402 KB | 38 plant diseases |
| AkidaNet VWW | 96×96×3 | 1×1×2 | 304 KB | Visual wake words |
| GXNOR MNIST | 28×28×1 | 1×1×10 | 203 KB | Digit classification |

### Face Analysis

| Model | Input | Output | .fbz | Notes |
|-------|-------|--------|------|-------|
| AkidaNet FaceID | 112×96×3 | 1×1×10575 | 2,834 KB | Face embedding (10,575-dim) |
| VGG UTK Face | 32×32×3 | 1×1×1 | 150 KB | Age regression |

### 3D

| Model | Input | Output | .fbz | Notes |
|-------|-------|--------|------|-------|
| PointNet++ ModelNet40 | 8×256×3 | 1×1×40 | 343 KB | 3D shape classification |

---

## Rust Path

### Image preprocessing

```rust
fn preprocess_image(
    rgb_bytes: &[u8],    // raw RGB, HWC layout
    width: usize,
    height: usize,
    target_w: usize,     // model input width (e.g. 224)
    target_h: usize,     // model input height (e.g. 224)
) -> Vec<f32> {
    let mut output = Vec::with_capacity(target_h * target_w * 3);
    for y in 0..target_h {
        for x in 0..target_w {
            let src_x = x * width / target_w;
            let src_y = y * height / target_h;
            let idx = (src_y * width + src_x) * 3;
            output.push(rgb_bytes[idx] as f32 / 255.0);
            output.push(rgb_bytes[idx + 1] as f32 / 255.0);
            output.push(rgb_bytes[idx + 2] as f32 / 255.0);
        }
    }
    output
}
```

### Object detection with YOLO

```rust
use akida_models::prelude::*;

let model = Model::from_file("baseCamp/zoo-artifacts/yolo_voc.fbz")?;
let backend = akida_driver::SoftwareBackend::new();

let image = load_and_preprocess("photo.jpg", 224, 224);
let output = backend.infer(&model, &image)?;
// output: [7, 7, 125] — 7×7 grid, 5 anchors × (4 bbox + 1 conf + 20 classes)

let detections = decode_yolo_output(&output, 7, 5, 20, CONFIDENCE_THRESHOLD);
for det in &detections {
    println!("Class {} at ({:.0},{:.0})–({:.0},{:.0}), conf={:.2}",
        VOC_CLASSES[det.class], det.x1, det.y1, det.x2, det.y2, det.confidence);
}
```

### Portrait segmentation

```rust
let model = Model::from_file("baseCamp/zoo-artifacts/akida_unet_portrait128.fbz")?;
let image = load_and_preprocess("portrait.jpg", 128, 128);
let mask = backend.infer(&model, &image)?;
// mask: [128, 128, 1] — per-pixel foreground probability

let binary_mask: Vec<bool> = mask.iter().map(|&p| p > 0.5).collect();
```

### Plant disease classification

```rust
let model = Model::from_file("baseCamp/zoo-artifacts/akidanet_plantvillage.fbz")?;
let leaf_image = load_and_preprocess("leaf_sample.jpg", 224, 224);
let classes = backend.infer(&model, &leaf_image)?;

let disease_idx = classes.iter().copied().enumerate()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .map(|(idx, _)| idx)
    .unwrap();

println!("Diagnosis: {} (confidence: {:.1}%)",
    PLANT_DISEASES[disease_idx], classes[disease_idx] * 100.0);
```

---

## Output

| Metric | Value | Notes |
|--------|-------|-------|
| Vision models parsed | 14/14 | All BrainChip vision models |
| Largest model | 5,269 KB (AkidaNet ImageNet) | Full ImageNet classifier |
| Smallest model | 150 KB (VGG UTK Face) | Minimal face analysis |
| Parse throughput | 13.7 MB/s | Across all models |
| YOLO output grid | 7×7×125 | 5 anchors × 25 values |

---

## Extension Points

**Custom object classes.** Retrain YOLO or CenterNet on your dataset
(medical imaging, satellite imagery, industrial inspection), export weights,
convert with `akida convert`.

**Multi-camera fusion.** Run multiple detection models on separate NP regions
using multi-tenancy (`baseCamp/systems/multi_tenancy.md`). Each camera
stream gets its own model slot.

**Agricultural monitoring.** The PlantVillage model (38 diseases) is a
starting point. Extend with crop-specific classifiers trained on local
disease datasets.

**Privacy-preserving face analysis.** FaceID generates 10,575-dimensional
embeddings on-device. No raw images leave the edge — only compact embeddings
for downstream matching.

**3D scene understanding.** PointNet++ processes point clouds from LiDAR or
depth cameras. Combine with 2D detection for multi-modal scene analysis.

---

## CLI Quick Test

```bash
# Parse representative models from each sub-domain
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/yolo_voc.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/akida_unet_portrait128.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/akidanet_imagenet.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/pointnet_plus_modelnet40.fbz
```
