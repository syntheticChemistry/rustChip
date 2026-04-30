# Physics — Lattice QCD, Transport, Phase Classification

**Spring origin:** hotSpring
**Maturity:** Production — 5,978 live hardware calls on AKD1000
**Zoo models:** ESN readout, phase classifier, transport predictor

---

## Problem

Lattice quantum chromodynamics (QCD) simulations generate configurations via
Hybrid Monte Carlo (HMC). Each trajectory takes 10–60 seconds on GPU. The
simulation must decide: accept or reject? continue or thermalize? which
observable to measure next?

These are low-latency classification and regression decisions that happen
between expensive GPU compute steps. The NPU provides sub-millisecond
inference that can steer the simulation without blocking the GPU.

Three sub-problems:

| Task | Question | Latency budget |
|------|----------|---------------|
| **ESN readout** | Given reservoir state, what is the thermalization quality? | < 1 ms |
| **Phase classification** | Confined or deconfined (SU(3) gauge theory)? | < 1 ms |
| **Transport prediction** | What are D*, eta*, lambda* from current observables? | < 1 ms |

---

## Model

### ESN Readout (lattice thermalization steering)

```
Architecture: InputConv(50,1,1) → FC(128) → FC(1)
Quantization: int4 symmetric per-layer
Input:        50-element reservoir state vector (echo state network)
Output:       scalar quality estimate ∈ [0, 1]
.fbz:         baseCamp/zoo-artifacts/esn_readout.fbz (physics export)
              baseCamp/zoo-artifacts/esn_multi_head_3.fbz (Rust-native)
```

The ESN reservoir runs in software (or on GPU via barraCuda). Only the
readout layer — the part that maps reservoir state to prediction — runs on
the NPU. This is the hybrid architecture documented in hotSpring's
`HybridEsn` executor.

### Phase Classifier (SU(3) confinement)

```
Architecture: InputConv(3,1,1) → FC(64) → FC(2)
Quantization: int4 symmetric per-layer
Input:        3 lattice observables (plaquette, Polyakov loop, topological charge)
Output:       2-class probability [confined, deconfined]
.fbz:         baseCamp/zoo-artifacts/phase_classifier.fbz
```

### Transport Predictor (diffusion, viscosity, conductivity)

```
Architecture: InputConv(6,1,1) → FC(128) → FC(3)
Quantization: int4 symmetric per-layer
Input:        6 thermodynamic observables
Output:       3 transport coefficients (D*, eta*, lambda*)
.fbz:         baseCamp/zoo-artifacts/esn_3head_transport.fbz (Rust-native)
              baseCamp/zoo-artifacts/transport_predictor.fbz (physics export)
```

---

## Rust Path

### Parse and inspect the model

```rust
use akida_models::prelude::*;

let model = Model::from_file("baseCamp/zoo-artifacts/esn_readout.fbz")?;
assert!(model.layer_count() >= 3);
```

### Feature extraction (domain-specific)

The reservoir state comes from an echo state network. In hotSpring, this is
the `reservoir_update` step that runs either on CPU, GPU (via barraCuda's
`esn_reservoir_update.wgsl`), or as a software simulation.

```rust
fn extract_reservoir_state(
    lattice_observables: &[f64],
    reservoir_weights: &[f64],
    reservoir_state: &mut [f64],
) -> Vec<f32> {
    // tanh(W_res * state + W_in * observables)
    for i in 0..reservoir_state.len() {
        let mut sum = 0.0;
        for j in 0..reservoir_state.len() {
            sum += reservoir_weights[i * reservoir_state.len() + j] * reservoir_state[j];
        }
        for (k, &obs) in lattice_observables.iter().enumerate() {
            sum += reservoir_weights[reservoir_state.len() * reservoir_state.len() + i * lattice_observables.len() + k] * obs;
        }
        reservoir_state[i] = sum.tanh();
    }
    reservoir_state.iter().map(|&x| x as f32).collect()
}
```

### NPU inference

```rust
// Software backend (no hardware required)
let backend = akida_driver::SoftwareBackend::new();
let result = backend.infer(&model, &reservoir_features)?;
let quality: f32 = result[0];

// Hardware backend (requires AKD1000 + VFIO)
let mgr = akida_driver::DeviceManager::discover()?;
let device = mgr.open_first()?;
let result = device.infer(&model, &reservoir_features)?;
```

### Steering decision

```rust
const THERMALIZATION_THRESHOLD: f32 = 0.85;

if quality > THERMALIZATION_THRESHOLD {
    // Configuration is thermalized — accept and measure
    measure_observables(&lattice);
} else {
    // Continue thermalizing — run more HMC steps
    hmc_step(&mut lattice);
}
```

---

## Output

| Metric | Measured value | Source |
|--------|---------------|--------|
| ESN readout throughput | 18,500 inferences/sec | hotSpring Exp 022 |
| Phase classifier accuracy | 100% (confined vs deconfined) | hotSpring Exp 022 |
| Transport predictor outputs | All finite, physically bounded | hotSpring Exp 022 |
| Total live hardware calls | 5,978 over 24 hours | hotSpring lattice production |
| Latency per inference | ~54 µs | Hardware measurement |

The key result: the NPU can make steering decisions faster than the GPU can
set up the next HMC trajectory. This means the NPU is never on the critical
path — it provides "free" intelligence between compute steps.

---

## Extension Points

**Multi-head steering.** hotSpring's `MultiHeadNpu` pattern runs 3–7 readout
models simultaneously on one AKD1000 (814 of 1,000 NPs). Each head monitors
a different observable. See `baseCamp/systems/npu_conductor.md`.

**Different gauge groups.** Replace the SU(3) phase classifier with SU(2) or
U(1) by retraining the readout layer. The reservoir architecture is
gauge-group agnostic.

**Plasma transport.** The transport predictor architecture works for any
system where you need to estimate transport coefficients from macroscopic
observables — including dense plasma (Yukawa OCP) validated in hotSpring's
`validate_transport` binary.

**Molecular dynamics.** Replace lattice observables with MD trajectory
features (temperature, pressure, pair correlation). The ESN + NPU readout
pattern is substrate-agnostic.

---

## Validation Binaries (hotSpring)

These require `hotSpring` and the `npu-hw` feature:

```bash
# In hotSpring/barracuda/
cargo run --bin validate_npu_quantization --features npu-hw
cargo run --bin validate_npu_pipeline --features npu-hw
cargo run --bin validate_lattice_npu --features npu-hw
cargo run --bin validate_multi_observable_npu --features npu-hw
```

For software-only validation (no hardware):

```bash
cargo run --bin validate_npu_quantization
cargo run --bin validate_npu_pipeline
```

---

## Further Reading

| Document | Path |
|----------|------|
| ESN readout architecture | `baseCamp/models/physics/esn_readout.md` |
| Phase classifier architecture | `baseCamp/models/physics/phase_classifier.md` |
| Hybrid ESN executor | `baseCamp/systems/hybrid_executor.md` |
| Multi-head NPU conductor | `baseCamp/systems/npu_conductor.md` |
| hotSpring NPU tolerances | `hotSpring/barracuda/src/tolerances/npu.rs` |
