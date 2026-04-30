# rustChip Leverage Guide

**What it provides:** Pure Rust neuromorphic inference — parse, convert, load, and run models on BrainChip Akida processors (AKD1000/AKD1500) without Python, without the vendor SDK, without a kernel module. Philosophy: hardware is accessible; the NPU is just another compute substrate alongside GPU and CPU.

---

## CLI Surface

The `akida` binary provides the command-line interface:

| Command | What it does |
|---------|-------------|
| `akida parse <file.fbz>` | Parse and display structure of any Akida model |
| `akida zoo-status` | Show status of all 28 registered models |
| `akida convert --weights <src> --arch <layers> --output <file> --bits <n>` | Convert weights to `.fbz` via pure Rust pipeline |
| `akida guidestone` | Run self-leveling validation: parse models, verify pipeline, benchmark |
| `akida discover` | Probe for Akida hardware via VFIO |
| `akida probe-sram` | Interactive SRAM diagnostics (requires hardware) |
| `akida bench` | Run benchmark suite |

### Examples

```bash
# Parse any .fbz model (no hardware needed)
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/ds_cnn_kws.fbz

# Convert weights to .fbz (no Python needed)
cargo run -p akida-cli -- convert \
  --weights trained.npy \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output model.fbz --bits 4

# Check zoo status
cargo run -p akida-cli -- zoo-status

# Self-validate the installation
cargo run -p akida-cli -- guidestone
```

---

## Library Dependency

### As a Rust crate dependency

```toml
# Model parsing and conversion (no hardware)
akida-models = { path = "crates/akida-models" }

# Hardware driver (VFIO inference)
akida-driver = { path = "crates/akida-driver" }

# Silicon model (register maps, NP mesh, BAR layout)
akida-chip = { path = "crates/akida-chip" }
```

### Feature-gated hardware access

```toml
# Software-only (default) — parse, convert, simulate
akida-driver = { path = "crates/akida-driver" }

# With hardware (requires VFIO + AKD1000)
akida-driver = { path = "crates/akida-driver", features = ["vfio"] }
```

### Minimal example — parse a model

```rust
use akida_models::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("model.fbz")?;
    println!("SDK version: {}", model.version());
    println!("Layers: {}", model.layer_count());
    println!("Weights: {} blocks", model.weights().len());
    Ok(())
}
```

### Convert weights to .fbz (library API)

```rust
use akida_models::import::load_npy;
use akida_models::quantize::quantize_per_layer;
use akida_models::schema::build_fbz;

let weights = load_npy(Path::new("weights.npy"))?;
let quantized = quantize_per_layer(&weights.data, 4);  // int4
let fbz_bytes = build_fbz(&model_descriptor);
std::fs::write("output.fbz", &fbz_bytes)?;
```

### Run inference (software backend)

```rust
use akida_driver::SoftwareBackend;
use akida_models::prelude::*;

let model = Model::from_file("model.fbz")?;
let backend = SoftwareBackend::new();
let result = backend.infer(&model, &input_features)?;
```

### Run inference (hardware via VFIO)

```rust
use akida_driver::DeviceManager;
use akida_models::prelude::*;

let model = Model::from_file("model.fbz")?;
let mgr = DeviceManager::discover()?;
let device = mgr.open_first()?;
let result = device.infer(&model, &input_features)?;
```

---

## Standalone Patterns

These patterns work with rustChip alone — no ecoPrimals workspace needed.

**Model inspection.** Parse any `.fbz` file and extract layer graph, weights, version, and structure. Works on all 28 zoo models plus any `.fbz` from the vendor SDK.

**Pure Rust conversion.** Import weights from `.npy` or `.safetensors`, quantize (int1/2/4/8 symmetric per-layer or per-channel), serialize to FlatBuffer, Snappy-compress, and write `.fbz`. Round-trip verified.

**Multi-model loading.** Load 7+ models simultaneously on one AKD1000 via NP address partitioning (`MultiTenantDevice`). Each model gets an isolated NP region.

**Online weight evolution.** Mutate weights on live hardware at 136 generations/second via direct SRAM writes (`NpuEvolver`). Enables evolutionary optimization without model reloading.

**Hardware fingerprinting.** The Temporal PUF module extracts unique device signatures from int4 quantization noise patterns — physical unclonable function without dedicated PUF circuitry.

**Domain-shift detection.** The `DriftMonitor` (`adaptive sentinel`) watches for distributional changes in inference inputs and triggers alerts or automatic recovery.

**Hybrid ESN.** The `HybridEsn` executor runs the reservoir in software (or on GPU) and the readout on NPU. This splits the computation at the natural architectural boundary.

**GuideStone verification.** The `akida guidestone` command runs self-leveling validation: parses all cached models, verifies the conversion pipeline, benchmarks throughput, and produces a reproducibility artifact.

**Sovereign boot (glowplug).** The absorbed glowplug module manages VFIO device lifecycle — bind, warm boot, tear down — without external orchestrator or kernel module. Derived from coralReef's ember/glowplug architecture.

**Science demos.** Five standalone binaries reproducing peer-reviewed NPU science claims:

| Binary | NPU Pattern | Science Domain |
|--------|-------------|----------------|
| `science_lattice_esn` | Hybrid ESN | Lattice QCD steering (hotSpring) |
| `science_bloom_sentinel` | Streaming Sentinel | Harmful algal bloom detection (wetSpring) |
| `science_spectral_triage` | Microsecond Gatekeeper | LC-MS spectral triage (wetSpring) |
| `science_crop_classifier` | Online Adaptation | Seasonal crop stress via (1+1)-ES (airSpring) |
| `science_precision_ladder` | Precision Discipline | f64 → f32 → int8 → int4 degradation |

Each binary is a standalone organism: it runs with `cargo run --bin <name>`, requires no external data, and reproduces the claim end-to-end using `SoftwareBackend`.

---

## Connection to the Wider Ecosystem

rustChip is designed as a standalone tool that naturally hands off to the ecoPrimals compute stack. No compile-time dependency on any primal — but the interfaces align.

### Compute Trio Integration

| Primal | Role | Interface with rustChip |
|--------|------|------------------------|
| [toadStool](https://github.com/ecoPrimals/toadStool) | WHERE — dispatch | toadStool's `akida-driver` and `akida-models` crates are upstream of rustChip's ports. toadStool adds heterogeneous dispatch (GPU/NPU/CPU tolerance routing). |
| [coralReef](https://github.com/ecoPrimals/coralReef) | HOW — compile | rustChip's VFIO driver mirrors coralReef's `ember`/`glowplug` architecture. Both use `/dev/vfio/` for sovereign hardware access without kernel modules. |
| [barraCuda](https://github.com/ecoPrimals/barraCuda) | WHAT — compute | NPU inference output (`&[f32]`) feeds directly into barraCuda WGSL shaders. No serialization, no IPC — shared memory buffers. |

### Data flow

```
barraCuda GPU shader → &[f32] → rustChip NPU → &[f32] → application
```

Or with toadStool dispatch:

```
toadStool.dispatch("inference", tolerance=1e-3)
  → routes to rustChip NPU (if available, within tolerance)
  → falls back to barraCuda GPU (DF64 path)
  → falls back to CPU (reference)
```

### Spring Usage

| Spring | How it uses rustChip's patterns |
|--------|-------------------------------|
| hotSpring | ESN reservoir + NPU readout for lattice QCD steering. `npu-hw` feature gates `akida-driver` dependency. `MultiHeadNpu` for multi-observable physics. |
| wetSpring | ESN → int8 classifiers for biology (QS phase, phylo placement, bloom sentinel, spectral triage). `npu` feature gate. 11 `validate_npu_*` binaries. |
| airSpring | NPU ecology classifiers (`validate_npu_eco`, `validate_npu_high_cadence`). ET₀ high-cadence estimation. |
| neuralSpring | Quantized weight export (`quantize_affine_i8_f64`), dispatch cost models with `npu_available` flags. Architecture validation (LeNet, LSTM, transformer parity). |

### IPC (if running as a service)

rustChip does not currently expose a JSON-RPC service (it is a library + CLI). Integration is via library dependency or CLI invocation:

```bash
# CLI integration
akida parse model.fbz | jq .layers

# Library integration
use akida_models::prelude::*;
```

If biomeOS orchestration is needed, register rustChip capabilities via `capability.register`:

```
capability.register { name: "npu.inference", provider: "rustChip", ... }
capability.register { name: "npu.parse", provider: "rustChip", ... }
capability.register { name: "npu.convert", provider: "rustChip", ... }
```

---

## What rustChip Does Not Do

| Concern | Who handles it |
|---------|---------------|
| GPU computation | barraCuda (WGSL shaders, DF64 emulation) |
| GPU compilation | coralReef (WGSL → native SASS/GCN) |
| Heterogeneous dispatch | toadStool (tolerance-based GPU/NPU/CPU routing) |
| Storage | nestGate (content-addressed dedup) |
| Cryptography | bearDog (FHE, signing, key management) |
| Provenance | sweetGrass (attribution braids) |
| Networking | songBird (discovery, federation) |
| Orchestration | biomeOS (capability routing, graph execution) |
| UI/visualization | petalTongue (SSE dashboards, data channels) |

rustChip owns one thing: **neuromorphic inference on Akida hardware, in Rust, without dependencies.**

---

## Key Documents

| Document | What it covers |
|----------|---------------|
| [QUICKSTART.md](QUICKSTART.md) | Clone → build → parse in 5 commands |
| [baseCamp/ZOO_GUIDE.md](baseCamp/ZOO_GUIDE.md) | Full model zoo (28 models) and conversion pipeline |
| [baseCamp/preserve/README.md](baseCamp/preserve/README.md) | Nature Preserve — 7 domain application patterns |
| [specs/AI_CONTEXT.md](specs/AI_CONTEXT.md) | Entry point for AI coding assistants |
| [specs/SILICON_SPEC.md](specs/SILICON_SPEC.md) | AKD1000/AKD1500 silicon capabilities |
| [specs/DRIVER_SPEC.md](specs/DRIVER_SPEC.md) | Driver architecture and safety rules |
| [specs/INTEGRATION_GUIDE.md](specs/INTEGRATION_GUIDE.md) | Integration with hotSpring / toadStool |
| [baseCamp/GUIDESTONE_CERTIFICATION.md](baseCamp/GUIDESTONE_CERTIFICATION.md) | GuideStone verification status |
| [CHANGELOG.md](CHANGELOG.md) | Change history |
