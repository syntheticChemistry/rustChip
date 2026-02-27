# NeuroBench

**URL:** https://github.com/NeuroBench/neurobench
**Paper:** NeuroBench: A Framework for Benchmarking Neuromorphic Computing (2023)
**Targets:** Intel Loihi 2, BrainChip Akida, spatio-temporal datasets
**License:** Apache 2.0
**Status:** Reviewed February 2026; key benchmarks mapped to rustChip

---

## What NeuroBench is

NeuroBench is a hardware-agnostic benchmark framework for neuromorphic computing.
It defines:

1. **Benchmarks** — standardized tasks with fixed datasets and evaluation metrics
2. **Workloads** — reference model architectures for each task
3. **Metrics** — accuracy, latency, throughput, energy, model size

Critically: NeuroBench has **hardware-validated results for Akida** (AKD1000)
published in the benchmark paper. These are the reference numbers rustChip's
benchmarks aim to reproduce.

---

## Static Data Benchmarks

Fixed dataset, all samples presented before evaluation.

| Benchmark | Task | Dataset | AKD1000 accuracy | AKD1000 latency |
|-----------|------|---------|-----------------|----------------|
| **keyword_spotting** | 35-word KWS | Google Speech Commands v2 | 93.8% top-1 | ~700 µs |
| **image_classification** | Image classification | CIFAR-10 | 92.7% | ~680 µs |
| **gesture_recognition** | 11-class DVS gesture | DVS128 Gesture | 97.9% | ~580 µs |
| **ecg_anomaly** | 2-class ECG | MIT-BIH Arrhythmia | 97.4% | ~540 µs |
| **face_detection** | Binary face/no-face | Custom | 99.1% | ~600 µs |

## Streaming Data Benchmarks

Online processing — each sample arrives as a stream.

| Benchmark | Task | Dataset | AKD1000 accuracy | Notes |
|-----------|------|---------|-----------------|-------|
| **chaotic_mslp** | Chaotic time series | Mean Sea Level Pressure | sMAPE 3.8% | ESN reservoir |
| **wireless_channel** | Channel estimation | Synthetic | NMSE −6.2 dB | RNN-based |
| **ecg_streaming** | Streaming ECG anomaly | Wearable ECG | 97.1% | Event-based |

---

## Connection to ecoPrimals

The streaming benchmarks are closest to ecoPrimals workloads:

| NeuroBench | ecoPrimals analogue | Overlap |
|-----------|---------------------|---------|
| `chaotic_mslp` | ESN thermalization detector | Both: reservoir → readout → streaming prediction |
| `ecg_streaming` | Transport coefficient predictor | Both: online multi-output regression |
| `gesture_recognition` | Phase classifier | Both: event → discrete class |

The ESN architecture in `chaotic_mslp` is architecturally identical to our
thermalization detector — same InputConv → FC readout structure. Different
training data, same hardware execution.

---

## Reproducing NeuroBench on rustChip

The NeuroBench AKD1000 results used the C kernel module + Python SDK. Our goal:
reproduce the same accuracy and latency via the Rust VFIO driver.

### Keyword Spotting (DS-CNN)

Status: **planned** — `models/edge/ds_cnn_kws.md`

```rust
// Once ds_cnn_kws.fbz is loaded (from MetaTF or hand-built):
use akida_models::Model;
use akida_driver::DeviceManager;

let model = Model::from_file("ds_cnn_kws.fbz")?;
let mgr   = DeviceManager::discover()?;
let mut exec = InferenceExecutor::new(mgr.open_first()?);

// Process MFCC features (49×10 = 490 float features)
let mfcc = compute_mfcc(&audio_frame);  // your MFCC impl
let result = exec.run(&mfcc, InferenceConfig { batch_size: 8, ..Default::default() })?;
let keyword_id = result.outputs.iter().enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(i, _)| i)
    .unwrap_or(0);
```

### Chaotic Time Series (ESN)

Status: **extends ecoPrimals ESN** — closely related to `models/physics/esn_readout.md`

The MSLP chaotic prediction task and our lattice QCD thermalization detector
use the same hardware path. The difference is training data, not architecture.

A rustChip `chaotic_esn.fbz` would be:
```
InputConv(1, 1, T→RS) → FC(RS→1)
  where RS = reservoir size (128, 256, or 512)
  trained on MSLP sequences
```

This is identical to the ESN readout we run in production. Pre-trained weights
for MSLP would come from the NeuroBench repository; the hardware execution path
is already validated.

---

## NeuroBench Metric Definitions

For comparison:

| Metric | Definition | AKD1000 profile |
|--------|-----------|-----------------|
| Accuracy | Task-specific (top-1, sMAPE, NMSE) | Competitive with GPU at 1000× less power |
| Latency (total) | End-to-end including bus | ~540–800 µs (PCIe dominated) |
| Latency (compute) | Chip compute only | ~0.7 µs |
| Throughput | Samples/second | 1,250–1,850 Hz (single), 2,566–3,100 Hz (batch=8) |
| Energy (chip) | J/inference on chip only | **1.4 µJ** |
| Energy (board) | J/inference including PCIe card | ~500 µJ (board floor dominated) |
| Parameters | Model parameter count | 50K–2M depending on task |

---

## What rustChip contributes to NeuroBench

NeuroBench's Akida results were produced with:
- Python SDK (MetaTF 2.19)
- C kernel module (akida_pcie.ko)

rustChip would be the first open Rust implementation of any NeuroBench
benchmark on Akida hardware. Key advantages:
- VFIO backend: no kernel module rebuild per kernel version
- Reproducible: `cargo run --bin neurobench_kws` instead of Python env setup
- Extensible: add physics benchmarks not in NeuroBench (thermalization, transport)
