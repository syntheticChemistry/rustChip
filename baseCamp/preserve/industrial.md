# Industrial — Predictive Maintenance, Sensor Fusion, Domain-Shift Detection

**Spring origin:** Architecture patterns (cross-spring)
**Maturity:** Architecture validated; models generated via pure Rust pipeline
**Zoo models:** Streaming sensor 12ch, adaptive sentinel, gesture/event models

---

## Problem

Industrial systems — manufacturing lines, power plants, infrastructure
networks — produce high-volume sensor data from vibration sensors,
temperature probes, current monitors, pressure transducers, acoustic
emission detectors. The question is always: "Is this normal?"

Predictive maintenance, anomaly detection, and quality control all reduce
to the same inference pattern: a feature vector from sensor data, classified
by a small model, producing a decision that gates expensive action
(inspection, shutdown, recalibration).

The NPU fits because:
- Sensors produce continuous streams → streaming inference
- Decisions must be real-time → microsecond latency
- Edge deployment (factory floor, substation) → no cloud dependency
- Multi-sensor fusion → multi-channel models

Four sub-problems:

| Task | Question | Sensors |
|------|----------|---------|
| **Vibration anomaly** | Is this machine operating normally? | Accelerometers (3-axis) |
| **Multi-sensor fusion** | What is the system state across 12 channels? | Mixed sensor array |
| **Domain-shift detection** | Has the operating regime changed? | All sensors (statistical) |
| **Gesture/event detection** | What action did the operator perform? | Camera, DVS, IMU |

---

## Model

### Vibration Anomaly Classifier

```
Architecture: InputConv(6,1,1) → FC(64) → FC(3)
Quantization: int4 symmetric per-layer
Input:        6 features (RMS, peak, crest factor per axis × 2 axes)
Output:       3-class [normal, warning, critical]
Build:        akida convert --weights vibration_model.npy \
                --arch "InputConv(6,1,1) FC(64) FC(3)" --bits 4
```

### 12-Channel Sensor Fusion

```
Architecture: InputConv(1,1,1) → FC(256) → FC(128) → FC(12)
Quantization: int8 symmetric per-layer
Input:        streaming sensor features from 12 heterogeneous channels
Output:       12-class system state (per-subsystem health)
.fbz:         baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz (Rust-native)
```

This model handles the case where a complex system has many different
sensor types: temperature, pressure, vibration, current, flow rate, etc.
Each output channel corresponds to a subsystem health assessment.

### Adaptive Sentinel (domain-shift)

```
Architecture: InputConv(64,3,1) → FC(128) → FC(1)
Quantization: int4 symmetric per-layer
Input:        64-feature state vector (statistical summaries of recent sensor window)
Output:       scalar anomaly score ∈ [0, 1]
.fbz:         baseCamp/zoo-artifacts/adaptive_sentinel.fbz (Rust-native)
```

Domain shift means the distribution of sensor readings has changed — maybe
gradually (tool wear, seasonal variation) or suddenly (process upset, sensor
failure). The sentinel watches for drift using a learned reference
distribution.

### Gesture and Event Detection (operator actions)

The zoo includes models for event-driven visual classification:

```
ConvTiny Gesture (DVS)    — 64×64×10 → 10 gesture classes (174 KB)
ConvTiny Handy Samsung    — 120×160×2 → 9 hand gesture classes (172 KB)
TENN ST DVS128           — 128×128×2 → 10 event classes (225 KB)
```

These process dynamic vision sensor (DVS) output — sparse, event-driven
data ideal for monitoring operator actions in safety-critical environments.

---

## Rust Path

### Feature extraction — vibration

```rust
struct VibrationReading {
    x: f32,
    y: f32,
    z: f32,
    timestamp: f64,
}

fn extract_vibration_features(window: &[VibrationReading]) -> Vec<f32> {
    let n = window.len() as f32;

    let rms_x = (window.iter().map(|r| r.x * r.x).sum::<f32>() / n).sqrt();
    let rms_y = (window.iter().map(|r| r.y * r.y).sum::<f32>() / n).sqrt();
    let peak_x = window.iter().map(|r| r.x.abs()).fold(0.0f32, f32::max);
    let peak_y = window.iter().map(|r| r.y.abs()).fold(0.0f32, f32::max);
    let crest_x = if rms_x > 0.0 { peak_x / rms_x } else { 0.0 };
    let crest_y = if rms_y > 0.0 { peak_y / rms_y } else { 0.0 };

    vec![rms_x, rms_y, peak_x, peak_y, crest_x, crest_y]
}
```

### Multi-sensor monitoring system

```rust
use akida_models::prelude::*;

let fusion_model = Model::from_file("baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz")?;
let sentinel_model = Model::from_file("baseCamp/zoo-artifacts/adaptive_sentinel.fbz")?;
let backend = akida_driver::SoftwareBackend::new();

loop {
    let sensors = poll_all_sensors();  // 12 channels

    // Per-subsystem health
    let health = backend.infer(&fusion_model, &sensors)?;
    for (i, &score) in health.iter().enumerate() {
        if score < SUBSYSTEM_THRESHOLDS[i] {
            alert_subsystem(i, score);
        }
    }

    // Global domain-shift check
    let state = compute_statistical_state(&sensor_history);
    let anomaly = backend.infer(&sentinel_model, &state)?;
    if anomaly[0] > DRIFT_THRESHOLD {
        trigger_maintenance_review(anomaly[0]);
    }

    std::thread::sleep(std::time::Duration::from_millis(POLL_MS));
}
```

### Multi-model on one chip

```rust
// Run vibration + fusion + sentinel simultaneously on one AKD1000
// using multi-tenancy (see baseCamp/systems/multi_tenancy.md)
let device = mgr.open_first()?;
let slot_vib = device.load_at_offset(&vibration_model, 0)?;
let slot_fuse = device.load_at_offset(&fusion_model, 100)?;
let slot_sent = device.load_at_offset(&sentinel_model, 500)?;

// Each model gets its own NP region — no interference
let vib_result = device.infer_slot(slot_vib, &vib_features)?;
let fuse_result = device.infer_slot(slot_fuse, &fuse_features)?;
let sent_result = device.infer_slot(slot_sent, &sent_features)?;
```

---

## Output

| Metric | Value | Notes |
|--------|-------|-------|
| Streaming sensor model size | 13.4 KB | 12-channel fusion |
| Adaptive sentinel model size | 4.6 KB | Domain-shift detection |
| Models fit on one AKD1000 | 3+ simultaneously | Multi-tenancy verified |
| Inference latency | ~54 µs per model | Hardware measurement |
| Gesture models parsed | 5/5 | DVS + temporal event |

---

## Extension Points

**Frequency-domain features.** Replace time-domain vibration features with
FFT-based spectral features (dominant frequency, spectral centroid, harmonic
ratios) for better discrimination of bearing faults vs imbalance vs
misalignment.

**Progressive degradation tracking.** Use the adaptive sentinel's anomaly
score as a time series. When the score trends upward over days/weeks,
schedule preventive maintenance before the threshold is reached.

**Remaining useful life (RUL).** Train a regression model
(InputConv→FC→FC(1)) on historical failure data to predict time-to-failure.
Convert with `akida convert` and deploy alongside the anomaly classifier.

**Safety monitoring.** The DVS gesture models detect operator actions in
real time. Extend to safety-critical applications: detect when a worker
enters a hazardous zone, verify lockout-tagout procedures, monitor proper
PPE usage.

**Edge fleet management.** Deploy rustChip on multiple edge nodes (one per
machine/asset), with each node running its own model suite. The self-contained
Rust binary (`cargo build --release`) deploys without Python, without
containers, without cloud connectivity.

---

## CLI Quick Test

```bash
# Parse the industrial models
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/adaptive_sentinel.fbz

# Parse gesture/event models for operator monitoring
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/conv_tiny_gesture_dvs.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/conv_tiny_handy_samsung.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/tenn_st_dvs128.fbz
```
