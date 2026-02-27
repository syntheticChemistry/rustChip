# DS-CNN — Keyword Spotting

**Architecture:** Depthwise Separable CNN → FC(128→35)
**Status:** 📋 Analysis complete; conversion plan documented
**Task:** Classify 35-word Google Speech Commands vocabulary
**Source:** NeuroBench KWS benchmark; BrainChip MetaTF `ds_cnn_kws.fbz`

---

## Architecture

```
Input: float[49×10]  (49 MFCC frames × 10 coefficients)
  │
  ▼
Conv2D(in=1, out=64, kernel=3×3, stride=1)  + BatchNorm + ReLU
  │
  ▼  (repeated N times, N=4 for AKD1000 version)
DWConv2D(64, kernel=3×3) + BN + ReLU → PtConv2D(64→64) + BN + ReLU
  │
  ▼
GlobalAveragePooling
  │
  ▼
FC(128→35)
  │
  ▼
Output: float[35]  (class scores for 35 keyword classes)
```

NeuroBench-reported accuracy: **93.8% top-1** on Google Speech Commands v2 test set.

---

## Status and Path to rustChip

The `.fbz` file for this model is available from BrainChip's `akida_examples`
repository and potentially from Hugging Face `brainchip/ds-cnn-kws`.

Loading today (zero conversion needed):

```rust
use akida_models::Model;
use akida_driver::DeviceManager;

let model = Model::from_file("ds_cnn_kws.fbz")?;
let mgr = DeviceManager::discover()?;
let mut device = mgr.open_first()?;

// Write program_info to device registers
device.write(model.program_info())?;

// DMA program_data + weight buffers to IOVA, then:
// for each 490-element input:
let scores = device.infer(&mfcc_features, &InferenceConfig::default())?;
let keyword_id = scores.argmax();
```

---

## MFCC Preprocessing (Rust)

The model expects 49 frames × 10 MFCC coefficients. The preprocessing pipeline:

```rust
// Sketch: MFCC computation in Rust (not yet in akida-models)
// Audio: 1 second at 16 kHz = 16,000 samples
// Frame length: 400 samples (25 ms)
// Frame hop:    160 samples (10 ms)
// → 99 frames, take 49 central frames
// → 10 mel filterbank channels per frame
// → 49×10 = 490 floats

fn compute_mfcc(audio: &[f32]) -> [f32; 490] {
    // window → FFT → mel filterbank → log → DCT → normalize
    todo!("queued: akida_models::audio::mfcc")
}
```

The preprocessing is the main implementation gap — the hardware path (`.fbz` loading
and inference) already works. Audio MFCC in Rust is a well-defined problem.
Candidate crates: `rustfft` + manual filterbank, or `mfcc` crate.

---

## ecoPrimals Extension: Acoustic Anomaly Sentinel

The wetSpring sentinel (baseCamp paper 04) uses ESN on NPU for environmental
monitoring. The DS-CNN KWS architecture can extend this:

| Standard KWS | Sentinel extension |
|-------------|-------------------|
| 35 fixed keywords | Domain vocabulary: "splash," "pump," "motor" |
| Google Speech Commands | Field recordings (HAB bloom sounds, soil cracking) |
| Classification | Anomaly score + keyword ID |

The architecture change is just retraining the FC head on domain data.
The hardware execution path is identical.

---

## NeuroBench Parity Target

| Metric | NeuroBench (Python SDK + C kmod) | rustChip target |
|--------|----------------------------------|----------------|
| Accuracy | 93.8% | Same (same .fbz, same model) |
| Throughput | ~1,400 Hz | ≥ 1,400 Hz (VFIO overhead ≤ 5%) |
| Energy | ~700 µJ/inference | Same |

If rustChip matches NeuroBench numbers with the VFIO backend, it proves
the Rust driver is production-complete for this task class.
