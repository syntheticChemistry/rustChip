# The Rust NPU Ecosystem — What rustChip Enables

**Date:** April 29, 2026

---

## The Current State

There is no Rust NPU ecosystem. As of April 2026, every neuromorphic
processor ships with a Python SDK, a C++ inference engine, or both.
There is no pure-Rust driver for any neuromorphic chip — Akida, Loihi,
SpiNNaker, or otherwise.

rustChip is the first.

This document describes what becomes possible when neuromorphic inference
is a native Rust capability, and how the wider Rust ecosystem can build on it.

---

## What Exists Today

### The rustChip Crate Architecture

```
rustChip/crates/
├── akida-chip      silicon model (register map, NP mesh, BAR layout)
├── akida-driver    full driver (VFIO, kernel, userspace, software backends)
├── akida-models    FBZ parser, ProgramBuilder, model zoo interface
├── akida-bench     benchmark suite, hardware experiments
└── akida-cli       command-line tool (enumerate, bind, parse, zoo-status)
```

Each crate is a standard Rust library publishable to crates.io. They depend
on each other through workspace references and expose clean public APIs.

### What the crates provide

| Crate | What it does | Key types |
|-------|-------------|-----------|
| `akida-chip` | Hardware abstraction | `Capabilities`, `ChipVersion`, `NpMesh`, `SramModel` |
| `akida-driver` | Device management and inference | `DeviceManager`, `InferenceExecutor`, `VfioBackend` |
| `akida-models` | Model parsing and construction | `Model`, `ZooModel`, `ModelZoo`, `ProgramBuilder` |
| `akida-bench` | Performance measurement | Benchmark harnesses, experiment runners |
| `akida-cli` | User-facing tool | `akida enumerate`, `parse`, `zoo-status`, `verify` |

---

## What Rust Unlocks

### 1. Zero-Copy Integration with Compute Pipelines

Rust's ownership model enables zero-copy data flow between GPU computation
and NPU inference:

```rust
// GPU shader output arrives as a pinned buffer
let gpu_output: &[f32] = barracuda_shader.output_slice();

// Feed directly to NPU — no serialization, no Python boundary
let classification = npu_executor.infer(gpu_output, config)?;

// Use result to steer next GPU dispatch
if classification.phase() == Phase::Deconfined {
    gpu_scheduler.switch_to(FineMeasurement);
}
```

In Python, this requires: GPU → numpy → Python object → numpy → NPU SDK → Python
object → numpy → GPU. Each boundary is a copy and a GIL acquisition.

In Rust, this is: `&[f32]` → `&[f32]`. The data never moves.

### 2. Compile-Time Model Catalog

The `ZooModel` enum encodes every known model variant as a type:

```rust
pub enum ZooModel {
    AkidaNetImagenet,
    DsCnnKws,
    EsnQcdThermalization,
    PhaseClassifierSu3,
    // ... 28 variants total
}
```

Adding a new model forces the developer to handle it in every match arm:
`filename()`, `np_budget()`, `source()`, `task()`, `validation()`,
`expected_size_bytes()`, `throughput_hz()`, `chip_energy_uj()`.

The compiler catches incomplete model handling. Python catches it at runtime,
if at all.

### 3. Fearless Concurrency for Multi-Tenant NPU

The AKD1000 can run 7 models simultaneously (814/1000 NPs). Managing
concurrent access to NP regions requires correctness guarantees:

```rust
let device = MultiTenantDevice::new(hardware)?;

// Each slot borrows a disjoint NP region — the borrow checker
// prevents overlapping access at compile time
let slot_0 = device.allocate(PhaseClassifier, 0..67)?;
let slot_1 = device.allocate(AndersonClassifier, 67..135)?;

// These can run concurrently — Rust proves they don't alias
rayon::join(
    || slot_0.infer(&phase_input),
    || slot_1.infer(&anderson_input),
);
```

In C/Python, NP region overlap is a runtime crash. In Rust, it's a compile error.

### 4. Embedded and Cross-Compilation

`akida-models` (the parser) has no system dependencies. It compiles to:

- **x86_64-linux** — servers, workstations
- **aarch64-linux** — ARM64 edge nodes (Jetson, Raspberry Pi)
- **wasm32** — browser-based model inspection (no hardware, just parsing)
- **riscv64gc** — RISC-V development boards

The driver (`akida-driver`) requires Linux + VFIO, but the model parsing
and inspection layer works everywhere Rust compiles.

### 5. Safe Hardware Access

VFIO ioctls and MMIO are inherently unsafe. rustChip contains the unsafety:

```rust
// Unsafe operations are isolated in akida-driver/src/vfio/ and mmio.rs
// Every unsafe block has a documented safety invariant

#[allow(unsafe_code)]
/// # Safety
/// Caller must ensure `fd` is a valid VFIO device file descriptor
/// and `region` is within the device's BAR range.
pub unsafe fn map_bar(fd: RawFd, region: &RegionInfo) -> Result<MmapRegion> {
    // ...
}
```

The public API is entirely safe:

```rust
// No unsafe in user code
let mgr = DeviceManager::discover()?;
let info = mgr.device(0)?;
println!("NPUs: {}", info.capabilities().npu_count);
```

---

## Integration Points with the Rust Ecosystem

### Scientific Computing

| Crate | Integration |
|-------|-------------|
| `ndarray` | `Model::input_shape()` returns dimensions compatible with `ndarray::Array` |
| `nalgebra` | Reservoir weight matrices for ESN can be `nalgebra::DMatrix<f32>` |
| `rayon` | Parallel model parsing, parallel multi-tenant inference |
| `polars` / `arrow` | Sensor data frames → NPU classification → annotated output |

### Async / Networking

| Crate | Integration |
|-------|-------------|
| `tokio` | Async inference tasks for streaming sensor data |
| `tonic` / `grpc` | gRPC inference service wrapping `InferenceExecutor` |
| `axum` | REST API for model inspection and hardware status |
| `rumqttc` | MQTT sensor ingestion → NPU → MQTT classification output |

### Embedded / Edge

| Crate | Integration |
|-------|-------------|
| `embedded-hal` | Future: NPU behind `embedded-hal` SPI/I2C for non-PCIe Akida variants |
| `defmt` | Low-overhead logging for NPU diagnostics on embedded targets |
| `probe-rs` | Flash and debug Akida-connected microcontrollers |

### Observability

| Crate | Integration |
|-------|-------------|
| `tracing` | Already integrated — `RUST_LOG=debug` enables full inference tracing |
| `metrics` | Export throughput, latency, energy counters to Prometheus |
| `opentelemetry` | Distributed tracing across GPU → NPU → storage pipeline |

---

## Architectural Patterns

### Pattern 1: Inference Service

```rust
use axum::{Router, Json, extract::State};
use akida_models::prelude::*;
use akida_driver::DeviceManager;

struct AppState {
    model: Model,
    executor: InferenceExecutor,
}

async fn classify(
    State(state): State<AppState>,
    Json(input): Json<Vec<f32>>,
) -> Json<Vec<f32>> {
    let result = state.executor.infer(&input, Default::default()).unwrap();
    Json(result.outputs)
}
```

A complete inference microservice in ~20 lines. Deploys as a single
binary with no Python runtime.

### Pattern 2: Streaming Pipeline

```rust
use tokio::sync::mpsc;

async fn sensor_pipeline(mut rx: mpsc::Receiver<SensorFrame>) {
    let model = Model::from_file("sentinel.fbz").unwrap();
    let executor = InferenceExecutor::new(/* ... */);

    while let Some(frame) = rx.recv().await {
        let features = extract_features(&frame);
        let result = executor.infer(&features, Default::default()).unwrap();

        if result.anomaly_score() > THRESHOLD {
            alert_operator(&frame, &result).await;
        }
    }
}
```

Continuous sensor monitoring with NPU classification. The `tokio` runtime
handles backpressure; the NPU processes at 18,500 Hz — far faster than
any sensor can produce data.

### Pattern 3: GPU+NPU Co-location

```rust
// barraCuda produces GPU shader output
let plasma_observables = barracuda::run_md_step(&lattice)?;

// rustChip classifies on NPU
let transport = npu.infer(&plasma_observables, config)?;

// Result feeds back to GPU
barracuda::apply_transport_correction(&lattice, &transport)?;
```

No FFI boundary. No serialization. Both `barraCuda` and `rustChip` operate
on `&[f32]` slices. The only thing crossing the PCIe bus is the inference
data — 24 bytes for a 6-float input.

---

## What the Ecosystem Is Missing (Opportunities)

### Model training in Rust

rustChip handles inference, not training. The Rust ecosystem needs:

- **`burn`** — already supports training; would need an Akida quantization backend
- **`candle`** — Hugging Face's Rust ML framework; post-training int4 export to `.fbz`
- **`tch-rs`** — PyTorch bindings; could generate `.fbz` via `cnn2snn` FFI

Today, training happens in Python and deployment in Rust. A Rust training
framework with `.fbz` export would close the loop.

### FlatBuffer schema for `.fbz`

The `.fbz` format is undocumented by BrainChip. rustChip reverse-engineered
it (varint + Snappy + FlatBuffer), but a formal `.fbs` schema would enable:

- `flatc`-generated Rust types for zero-copy access
- Schema validation before hardware load
- Forward compatibility with AKD1500 format changes

### Quantization library

A standalone `akida-quantize` crate would provide:

- Max-abs per-layer quantization (what we use today)
- Per-channel quantization (higher accuracy)
- Mixed-precision (int4 weights + int8 activations)
- Calibration dataset support

This is the missing piece between "I have a trained model" and "I have a
`.fbz` file" — currently filled by the Python SDK as a one-time oracle.

### Hardware abstraction trait

A `NeuromorphicDevice` trait would abstract over NPU vendors:

```rust
pub trait NeuromorphicDevice {
    fn discover() -> Result<Vec<Self>>;
    fn capabilities(&self) -> &Capabilities;
    fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle>;
    fn infer(&self, handle: ModelHandle, input: &[f32]) -> Result<Vec<f32>>;
}
```

rustChip's `akida-driver` could implement this for Akida. Future drivers
for Loihi, SpiNNaker, or new chips could implement the same trait. User
code would be hardware-agnostic.

---

## The Sovereignty Argument

The vendor stack for every neuromorphic processor is:

1. Proprietary kernel module (C, GPL-2.0, binary blobs)
2. Proprietary C++ inference engine (closed source)
3. Python SDK wrapping the C++ engine (pip install, 40+ dependencies)
4. Vendor-controlled model format (undocumented binary)

You cannot inspect what happens between your input and the hardware's output.

rustChip replaces layers 1–3 with inspectable Rust code and documents layer 4.
The driver is AGPL — if you modify it, you share the modification. If you
don't modify it, you use it freely. BrainChip gets a working open driver
they didn't have to write. Users get hardware they can audit.

This is the scyBorg exception protocol: the driver's existence demonstrates
capability; the open license invites collaboration rather than demanding it.

---

## Getting Involved

The model zoo is validated. The driver works on live hardware. The crates
compile. What's needed next:

1. **More models** — export models from your domain and test against the parser
2. **`ProgramBuilder::serialize()`** — complete the Rust-native `.fbz` writer
3. **Quantization crate** — standalone int4/int8 quantization without Python
4. **Hardware abstraction** — trait design for multi-vendor neuromorphic support
5. **Training integration** — `burn` or `candle` backend for Akida deployment

Repository: https://github.com/syntheticChemistry/rustChip
License: AGPL-3.0-or-later (code), CC-BY-SA-4.0 (docs), ORC (game mechanics)
