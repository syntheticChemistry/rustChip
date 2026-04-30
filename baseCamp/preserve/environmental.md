# Environmental — Bloom Surveillance, ET₀, Soil Moisture, Ecosystem Monitoring

**Spring origin:** airSpring, wetSpring
**Maturity:** NPU validation binaries exist for ecology and bloom monitoring
**Zoo models:** Streaming sensor 12ch, adaptive sentinel, bloom ESN

---

## Problem

Environmental monitoring generates high-cadence sensor data — weather
stations, water quality probes, soil moisture arrays, satellite feeds.
Decisions range from "is this reading anomalous?" to "is an ecological
regime shift happening?"

The common thread: continuous data streams that need real-time classification
without cloud latency, because the monitoring station is in a field, on a
buoy, or at a remote watershed. NPU inference provides microsecond
classification at milliwatt power.

Four sub-problems:

| Task | Question | Data source |
|------|----------|------------|
| **Bloom surveillance** | Is a harmful algae bloom forming or intensifying? | Water quality sensors |
| **ET₀ high-cadence** | What is the reference evapotranspiration right now? | Weather station (T, RH, wind, radiation) |
| **Soil moisture anomaly** | Is the soil moisture profile consistent with expectations? | Multi-depth TDR sensors |
| **Ecosystem sentinel** | Has the ecosystem drifted from its baseline regime? | Multi-modal sensor array |

---

## Model

### Bloom Surveillance ESN

```
Architecture: InputConv(1,1,1) → FC(256) → FC(128) → FC(12)
Quantization: int8 symmetric per-layer
Input:        streaming sensor features (chlorophyll-a, cyanobacteria, turbidity,
              temperature, pH, DO, conductivity, ...)
Output:       12-channel classification (bloom type × severity × trend)
.fbz:         baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz (Rust-native)
```

wetSpring's `validate_temporal_esn_bloom` and `validate_npu_bloom_sentinel`
validate this architecture. The 12 output channels encode:
- 4 bloom species groups (cyanobacteria, dinoflagellate, diatom, mixed)
- 3 severity levels (emerging, active, declining)

### ET₀ High-Cadence Estimator

```
Architecture: InputConv(6,1,1) → FC(64) → FC(1)
Quantization: int4 symmetric per-layer
Input:        6 weather features (T_min, T_max, T_mean, RH, wind, solar radiation)
Output:       scalar ET₀ estimate (mm/day)
Build:        akida convert --weights et0_trained.npy --arch "InputConv(6,1,1) FC(64) FC(1)" --bits 4
```

airSpring validates Penman-Monteith, Hargreaves, and Priestley-Taylor ET₀
methods. An NPU regressor can provide sub-second ET₀ estimates for
irrigation scheduling without computing the full energy balance.

### Adaptive Sentinel (domain-shift detection)

```
Architecture: InputConv(64,3,1) → FC(128) → FC(1)
Quantization: int4 symmetric per-layer
Input:        64-feature environmental state vector
Output:       scalar anomaly score ∈ [0, 1]
.fbz:         baseCamp/zoo-artifacts/adaptive_sentinel.fbz (Rust-native)
```

The sentinel watches for distributional drift — when the sensor readings
no longer match the baseline regime. This triggers alerts, model retraining,
or fallback to conservative control strategies.

---

## Rust Path

### Feature extraction — bloom monitoring

```rust
struct WaterQualityReading {
    chlorophyll_a: f32,    // µg/L
    cyanobacteria: f32,    // cells/mL
    turbidity: f32,        // NTU
    temperature: f32,      // °C
    ph: f32,
    dissolved_oxygen: f32, // mg/L
    conductivity: f32,     // µS/cm
    timestamp: f64,        // Unix epoch
}

fn extract_bloom_features(readings: &[WaterQualityReading]) -> Vec<f32> {
    let latest = readings.last().unwrap();
    vec![
        latest.chlorophyll_a,
        latest.cyanobacteria,
        latest.turbidity,
        latest.temperature,
        latest.ph,
        latest.dissolved_oxygen,
        latest.conductivity,
    ]
}
```

### Continuous monitoring loop

```rust
use akida_models::prelude::*;

let bloom_model = Model::from_file("baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz")?;
let sentinel_model = Model::from_file("baseCamp/zoo-artifacts/adaptive_sentinel.fbz")?;
let backend = akida_driver::SoftwareBackend::new();

loop {
    let reading = poll_sensor();
    sensor_buffer.push(reading);

    // Bloom classification
    let bloom_features = extract_bloom_features(&sensor_buffer);
    let bloom_result = backend.infer(&bloom_model, &bloom_features)?;

    // Domain-shift detection
    let state_vector = compute_environmental_state(&sensor_buffer);
    let anomaly = backend.infer(&sentinel_model, &state_vector)?;

    if anomaly[0] > DRIFT_THRESHOLD {
        log_drift_alert(&sensor_buffer, anomaly[0]);
    }

    if is_bloom_active(&bloom_result) {
        report_bloom(&bloom_result, &reading);
    }

    std::thread::sleep(std::time::Duration::from_secs(POLL_INTERVAL));
}
```

---

## Output

| Metric | Value | Source |
|--------|-------|--------|
| Streaming sensor model size | 13.4 KB | Rust-native conversion |
| Adaptive sentinel model size | 4.6 KB | Rust-native conversion |
| airSpring NPU ecology validation | Passing | `validate_npu_eco` |
| wetSpring bloom sentinel | Validated | `validate_npu_bloom_sentinel` |
| wetSpring sentinel stream | Validated | `validate_npu_sentinel_stream` |

---

## Extension Points

**Real-time irrigation control.** Combine ET₀ estimation with soil moisture
anomaly detection. When soil moisture deviates from the ET₀-predicted
trajectory, the system adjusts irrigation automatically.

**Multi-station fusion.** Run bloom classifiers for multiple monitoring
stations on one AKD1000 using multi-tenancy. Each station gets its own NP
slot, with a shared sentinel model watching for cross-station anomalies.

**Satellite-derived features.** Augment in-situ sensors with NDVI,
chlorophyll-a, and SST from satellite feeds. The feature extraction step
handles the fusion; the NPU classifies the combined feature vector.

**Climate scenario evaluation.** airSpring's `validate_climate_scenario`
and `validate_season_wb` validate water balance under CMIP6 projections.
An NPU classifier can flag scenarios where the water balance enters
critical drought or surplus regimes.

**Drought indexing.** airSpring computes SPI, SPEI, and Palmer drought
indices. An NPU regressor trained on historical index sequences can
provide early warning of drought onset.

---

## Validation Binaries

```bash
# airSpring — NPU ecology
cd airSpring/barracuda/
cargo run --bin validate_npu_eco
cargo run --bin validate_npu_high_cadence

# wetSpring — bloom monitoring
cd wetSpring/barracuda/
cargo run --bin validate_npu_bloom_sentinel
cargo run --bin validate_npu_sentinel_stream
cargo run --bin validate_temporal_esn_bloom
```
