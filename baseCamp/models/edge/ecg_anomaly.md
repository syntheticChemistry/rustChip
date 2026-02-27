# ECG Anomaly Detection

**Architecture:** FC(64→32) → FC(32→2)
**Status:** 📋 Analysis complete
**Task:** Classify ECG heartbeats as normal or anomalous
**Source:** NeuroBench ECG benchmark; MIT-BIH Arrhythmia Database

---

## The Model

The NeuroBench ECG anomaly model is intentionally small — designed for
wearable deployment (coin-cell viable). The architecture is pure FC:

```
Input: float[64]  (64 time-domain ECG samples around R-peak, ~0.2s window)
  │
  ▼
FC(64→32) + threshold neurons
  │
  ▼
FC(32→2)
  │  softmax → {normal, anomaly}
  ▼
Output: float[2]  (class probabilities)
```

Only **96 NPs** required — leaves 904 free on AKD1000 for concurrent workloads.

---

## Measured (NeuroBench)

| Metric | Value |
|--------|-------|
| Accuracy | 97.4% on MIT-BIH test set |
| Throughput | ~2,200 Hz single call |
| Energy (chip) | **1.1 µJ** |
| Model size | ~40 KB .fbz |

This is the most energy-efficient model in the NeuroBench suite.
At 1.1 µJ, a 1000 mAh coin cell (3.7V = 13.3 kJ) provides:
**12.1 billion inferences** — theoretical 60-year battery life at 24 Hz.

---

## ecoPrimals Extension: Sentinel Monitoring

The wetSpring Paper 04 (Sentinel Microbes) uses AKD1000 for anomaly detection.
The ECG model architecture (small FC chain) is the same architecture used
for sentinel classifiers. Cross-domain transfer:

| ECG model | Sentinel extension |
|-----------|-------------------|
| R-peak window | Species abundance window (64-sample timeseries) |
| Normal / anomaly | Healthy community / dysbiosis |
| MIT-BIH | Field sensor time series (16S abundance) |

The hardware execution path is identical. Only the training data changes.

---

## Multi-task Packing

Because this model uses only 96 NPs out of 1,000:

```
AKD1000 NP budget: 1,000 total

Allocation option A (full utilization):
  Slot 1: ECG anomaly      (96 NPs)  — wearable health monitor
  Slot 2: Phase classifier  (67 NPs)  — physics
  Slot 3: Anderson class.   (68 NPs)  — spectral
  Slot 4: Transport pred.  (134 NPs) — WDM surrogate
  Slot 5: ESN readout      (179 NPs) — thermalization
  Slot 6–10: DS-CNN KWS    (280 NPs) — keyword spotting
  ─────────────────────────────────────
  Total used: 824 NPs, 176 spare
```

The `set_variable()` API (Discovery 6) enables switching between loaded models
at ~86 µs without reprogramming. All five could be loaded simultaneously
via separate `program_external()` calls into different SRAM regions,
hot-swapped at runtime.

This is the "NPU GPU conductor" pattern from hotSpring Exp 023.
