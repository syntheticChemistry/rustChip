# Neuromorphic Coprocessor for Scientific Computing
## Technical Brief — AKD1000 Integration in hotSpring

**Date:** February 27, 2026
**Version:** hotSpring v0.6.14 / metalForge v0.3
**Hardware:** BrainChip AKD1000 (PCIe, BC.00.000.002)
**Repository:** [github.com/syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring)
**License:** AGPL-3.0-only

---

## 1. Project Context

hotSpring is a computational physics validation project that reproduces
peer-reviewed results on consumer hardware using a pure Rust / WGSL compute
stack (BarraCuda). The project covers four physics domains:

| Domain | Papers Reproduced | Validation Checks |
|--------|:-----------------:|:-----------------:|
| Dense plasma molecular dynamics | 6 | 60/60 |
| Lattice QCD (SU(3), quenched + dynamical) | 6 | 32⁴ production |
| Nuclear structure (HFB, SEMF) | 6 | L1/L2/L3 |
| Spectral theory (Anderson, Hofstadter) | 4 | 45/45 |
| **Total** | **22** | **~700 checks, 39/39 suites** |

Current crate statistics:
- **664 tests** (629 library + 31 integration + 4 doc), 0 failures
- **0 clippy warnings** (pedantic + nursery, library + 76 binaries)
- **~150 centralized tolerances** with physics justification and DOI provenance
- **0 hardcoded cross-primal references** — pure capability-based discovery
- **25 WGSL shaders** (lattice QCD, DF64 arithmetic, PRNG, observables)
- **22 experiment journals** documenting every measurement

The stack is vendor-agnostic: WGSL compiles to SPIR-V (Vulkan), Metal,
DX12, or WebGPU. We run on NVIDIA (proprietary + NVK open-source),
AMD (RADV), and Intel (ANV) without code changes.

---

## 2. Why Neuromorphic

The physics pipeline has three compute tiers with fundamentally different
power/precision/latency requirements:

| Tier | Workload | Precision | Power | Latency |
|------|----------|-----------|-------|---------|
| **Heavy compute** | Force calculation, HMC trajectories | f64 / DF64 | 200-350W | ms-scale |
| **Inference** | Transport prediction, phase classification, anomaly detection | int4-int8 | **mW-scale** | μs-scale |
| **Orchestration** | Data routing, scheduling, validation | f64 | 10-65W | ms-scale |

GPUs are optimal for Tier 1. CPUs handle Tier 3. But using a 350W GPU for
Tier 2 inference (a 2,950-MAC ESN readout) is a 10,000× power mismatch.

A neuromorphic coprocessor occupies the Tier 2 niche: microsecond inference
at microwatt power, running continuously alongside the GPU without stealing
compute cycles from physics.

---

## 3. AKD1000 Integration Architecture

### 3.1 Three-Substrate Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Host CPU (Threadripper 3970X)                  │
│                                                                   │
│  ┌──────────────────────┐   ┌───────────────┐   ┌────────────┐ │
│  │  RTX 3090 (24 GB)    │   │  AKD1000      │   │ Titan V    │ │
│  │                      │   │  (PCIe x1)    │   │ (12 GB)    │ │
│  │  DF64 HMC production │   │  ESN steering │   │ f64 oracle │ │
│  │  7.6 s/trajectory    │   │  668 cycles   │   │ validation │ │
│  │  338W, 74°C          │   │  <1 mW chip   │   │ spot-check │ │
│  │  100% utilization    │   │  656 μs PCIe  │   │            │ │
│  └──────────┬───────────┘   └───────┬───────┘   └─────┬──────┘ │
│             │                       │                  │        │
│             └── velocity features ──┘  ── oracle ──────┘        │
│                                                                   │
│  Discovery: probe_npu_available() → sysfs /dev/akida* scan       │
│  Fallback:  NpuSimulator (f32 CPU) when hardware absent           │
└─────────────────────────────────────────────────────────────────┘
```

This pipeline ran in production across two experiments:

- **Experiment 015** (validation): 2,100 HMC trajectories across 3 strategic
  β values. Results validated within 2% of native f64 baseline.
- **Experiment 022** (production, completed Feb 27): 32⁴ lattice, 10
  NPU-steered β points, 5,900 measurement trajectories, 5,978 live NPU
  calls via PCIe. ESN weights bootstrapped from prior runs (cross-run
  learning). 63% thermalization savings. Susceptibility peak χ=32.41
  at β=5.7797 confirming deconfinement transition.

### 3.2 Software Architecture

**Rust hardware adapter** (`barracuda/src/md/npu_hw.rs`):
- Feature-gated (`npu-hw`) for clean builds without hardware present
- `NpuHardware::discover()` — enumerates via akida-driver `DeviceManager`
- `NpuHardware::from_exported(&weights, hw_info)` — load trained ESN weights
- `predict(&features) → f32` — same API as `NpuSimulator`

**Capability-based discovery** (`barracuda/src/discovery.rs`):
- `probe_npu_available()` — feature-gated: akida-driver or sysfs fallback
- No hardcoded device paths — scans `/dev/akida*` with vendor ID `0x1e7c`
- Integrated into metalForge substrate system (`SubstrateKind::Npu`)

**metalForge probe** (`metalForge/forge/src/probe.rs`):
- PCIe sysfs scan for BrainChip vendor ID
- Capabilities: `QuantizedInference` (8-bit, 4-bit), `BatchInference` (max 8), `WeightMutation`
- SRAM budget: 8 MB on-chip

**NpuSimulator** (`barracuda/src/md/reservoir.rs`):
- Full f32 ESN inference matching hardware behavior
- Used for validation on machines without AKD1000
- Same `predict()` interface — swap hardware/software with a feature flag

### 3.3 Validation Coverage

| Binary | Substrates | What It Validates |
|--------|:----------:|-------------------|
| `validate_three_substrate` | CPU+GPU+NPU | Full pipeline parity |
| `validate_titan_oracle` | GPU+NPU+Titan V | DF64 vs native f64 spot-check |
| `validate_multi_observable_npu` | GPU+NPU | Simultaneous plaquette + Polyakov + susceptibility |
| `validate_online_learning` | GPU+NPU | ESN weight updates during production |
| `validate_adaptive_beta_scan` | GPU+NPU | NPU-steered β selection |
| `validate_streaming_pipeline` | CPU→GPU→NPU→CPU | Phase 4b real NPU integration |
| `validate_mixed_substrate` | GPU+NPU | 4 physics domains |

---

## 4. What We Found: SDK vs Silicon

We systematically tested every SDK assumption against actual hardware behavior.
Full details with raw measurements in the hotSpring repository
(`metalForge/npu/akida/BEYOND_SDK.md`).

### 4.1 SDK Assumptions Overturned

| # | SDK Documentation | Actual Hardware Behavior | Method |
|---|-------------------|--------------------------|--------|
| 1 | InputConv: 1 or 3 channels only | **Any channel count works** (tested 1-64) | Direct model build, bypass MetaTF |
| 2 | FC layers run as independent HW sequences | **All FC layers merge into single HW pass** via SkipDMA | Multi-depth model timing |
| 3 | Single-sample inference only | **Batch=8 amortizes PCIe: 948→390 μs/sample** (2.4×) | Batch sweep 1-64 |
| 4 | One clock mode | **3 modes**: Performance / Economy / LowPower | Clock mode API |
| 5 | Max FC width ~hundreds | **Tested to 8192+ neurons**, all map to hardware | Width sweep to SRAM limit |
| 6 | Weight updates require reprogramming | **`set_variable()` updates weights without reprogram** | Weight mutation test |
| 7 | "30 mW" power specification | **Board floor 900 mW; chip compute below measurement noise** | Power monitoring |
| 8 | 8 MB SRAM is the addressable limit | **PCIe BAR1 exposes 16 GB address space** | sysfs BAR probe |
| 9 | Program binary is opaque | **FlatBuffer format: program_info + program_data, weights via DMA** | Binary analysis |
| 10 | Simple inference engine | **C++ engine: SkipDMA, on-chip learning registers, 51-bit threshold SRAM, `program_external()`** | Symbol analysis (1,048 exports) |

### 4.2 Confirmed Silicon Constraints

These are real hardware limits, not SDK restrictions:

| Constraint | Evidence | Workaround |
|------------|----------|------------|
| No tanh activation | Only bounded ReLU in Akida 1.0 | Quantize after host-side tanh; or design native int4+ReLU reservoir |
| Feed-forward only | No recurrence in hardware | Host drives ESN recurrence loop |
| Integer-only arithmetic (int4/int8) | No floating point in datapath | Quantize physics values to uint8 input range |
| PCIe x1 Gen2 latency | ~650 μs minimum round-trip | Batch amortization (batch=8 → 390 μs/sample) |
| 1-bit on-chip learning | SDK limits binary weights + activations | C++ engine symbols suggest broader support (untested) |
| InputConv HW mapping: kernel ≥ 3, channels 1 or 3 | `hw_only=True` rejects others | SW fallback is fine — FC layers do the real work |

### 4.3 Architecture Discoveries

**51-bit threshold SRAM**: The C++ engine exposes `get_tsram_51b_memory_size()`,
indicating more precision in the threshold/comparison logic than the "4-bit
everything" specification suggests. Unexplored.

**16 GB PCIe BAR1**: The full NP mesh decode range is 16 GB (vs 8 MB physical
SRAM). With 78 NPs, each could have ~200 MB of addressable space. Whether
this is usable or just decoder range is unknown.

**`program_external()`**: Raw FlatBuffer injection at a device memory address.
Combined with reverse-engineered program format, this enables custom mesh
programming without the SDK compilation pipeline.

**Three hardware variants in C++ engine**: `akida::v1` (AKD1000), `akida::v2`
(Akida 2.0), `akida::pico` (compact variant). The v2 codebase includes
TENNs, 8-bit weights, LUT activations, and skip connections.

### 4.4 SRAM Direct Access (rustChip)

rustChip adds direct BAR0 register and BAR1 SRAM read/write via `SramAccessor`
and `VfioBackend::map_bar1()`. This capability enables:

- **Model load verification**: Read back NP SRAM after DMA load; compare
  against expected weights from `.fbz` to verify correct deployment.
- **Direct weight mutation**: Bypass `set_variable()` overhead by writing
  to BAR1 SRAM regions directly (via `NpuBackend::mutate_weights()`).
- **Device fingerprinting**: SRAM power-on state and noise patterns provide
  a PUF-like signature for attestation and anti-cloning.

The `probe_sram` binary (probe/scan/test modes) and `LoadVerification`
struct support these workflows. `Capabilities::from_bar0()` extracts runtime
NP count, SRAM size, and mesh topology from BAR0 registers.

---

## 5. Physics Use Cases (Validated + Planned)

### 5.1 Validated: ESN Adaptive Steering

Echo State Network predicts deconfinement transition coupling (β_c) from
plaquette + Polyakov loop time series. The NPU runs the ESN readout layer
while the GPU computes HMC trajectories.

| Metric | Exp 015 (3 pts) | Exp 022 (10 pts, live NPU) |
|--------|:----------------:|:--------------------------:|
| ESN topology | 50-neuron, 8-dim, 1 output | Same + cross-run bootstrap |
| Weight budget | 2,950 parameters (1,475 B) | Same |
| β_c estimate | 5.5051 (error 3.3%) | 5.5657 (error 2.2%) |
| Hardware inference | 668 cycles, 656 μs | Same (PCIe dominated) |
| Total NPU calls | ~600 | 5,978 |
| Thermalization savings | Not measured | 63% (1,260 / 2,000 skipped) |
| Rejection prediction | Not measured | 80.4% accuracy |
| Cross-run learning | No | Yes — bootstrapped from 749 prior points |
| Power | < noise above 900 mW floor | Same |

### 5.2 Validated: Multi-Observable Prediction

Simultaneous prediction of plaquette, Polyakov loop magnitude, and
susceptibility from the same feature vector. Multi-output adds zero
measurable latency (50→1024→10 is same speed as 50→1024→1).

### 5.3 Planned: Native Int4+ReLU Reservoir

Design reservoir dynamics optimized for the hardware's actual compute
model (int4 weights, bounded ReLU activation, event-based sparsity)
rather than approximating a float ESN. Literature supports ternary
reservoirs (Appeltant et al., Larger et al.) for dynamical systems.

### 5.4 Planned: BingoCube Evolutionary Reservoir (Feed-Forward PDE Solver)

The AKD1000's feed-forward-only constraint makes traditional recurrent
architectures (ESN, LSTM) impossible in hardware — the host must drive
the recurrence loop. We are developing BingoCube Reservoir, an
architecture that replaces temporal recurrence with evolutionary
generations, enabling fully feed-forward neuromorphic computation.

**Core concept:** A bingo board is a structured random network — column-
locked, distinct-valued, drawn from a finite but combinatorially vast
space (~10^31 boards for L=5). When input data streams through as the
"caller," the board's response pattern (which cells match, in what order)
is a deterministic random projection of the input. This IS reservoir
computing: random weights project input into a high-dimensional space
where a simple readout can extract structure.

**Why this solves the feed-forward constraint:**

```
Traditional ESN (requires recurrence — AKD1000 cannot do):
  input(t) → reservoir(t) = f(W_in·input(t) + W_res·state(t-1))
                                                    ↑ feedback

BingoCube Reservoir (pure feed-forward — AKD1000 native):
  input → Board₁ response → ┐
  input → Board₂ response → ├→ FC readout → output
  input → Board₃ response → ┘
          ↑ no feedback, N boards run in parallel
```

Multiple boards run simultaneously against the same input stream.
Each board is a different random projection. The ensemble of board
responses replaces the single reservoir's temporal memory with
combinatorial diversity. The FC readout layer (which the AKD1000
handles natively via SkipDMA-merged passes) extracts the prediction.

**Evolutionary generations replace temporal recurrence:**

After a generation of input processing, the boards are evaluated.
Boards whose response patterns correlated with the target observable
inform the construction of the next generation. New boards inherit
structural properties from high-performing ancestors but add new
randomness — constrained evolution applied to the compute substrate.

```
Generation 0: Random boards (naive initialization)
Generation 1: Boards informed by Gen 0 performance
Generation 2: Boards informed by Gen 1 (nautilus shell growing)
   ...
Generation N: Boards evolved to the environment's structure
```

The full evolutionary history forms a nautilus shell — each layer wraps
the previous, preserving heritage while adding new adaptation. When
encountering a new environment (different physics regime, different
data stream), the shell provides "informed randomness" as a starting
point rather than naive random initialization.

**Hardware mapping to AKD1000:**

| BingoCube Concept | AKD1000 Hardware | Advantage |
|-------------------|------------------|-----------|
| Board values (int, column-locked) | int4 weights in NP SRAM | Native format — no quantization loss |
| Board response (sparse matches) | Event-based activation | Zero-compute for non-matching cells |
| Multiple boards in parallel | Multiple NP subsets | 78 NPs, each running a board |
| FC readout | FullyConnected layer | Single HW pass via SkipDMA |
| Board evolution | `set_variable()` weight mutation | 13 ms per generation update |
| Board heritage (nautilus shell) | `.fbz` model serialization | Save/reload evolved boards |

**Target applications:**

| Application | Input (caller) | Boards | Readout |
|-------------|---------------|--------|---------|
| PDE solver (Poisson) | Boundary values, source terms | Random projections of discretized domain | Solution field approximation |
| Phase classification | Plaquette, Polyakov loop | Regime-tuned projections | Phase label + confidence |
| Transport prediction | Velocity features | Coupling-evolved boards | D*, η*, λ* |
| Anomaly detection | Observable time window | Normal-trained boards | Anomaly score |

This replaces the Sandia NeuroFEM approach (which requires Loihi 2's
programmable neuron dynamics) with an architecture native to the
AKD1000's actual silicon: integer arithmetic, feed-forward topology,
event-based sparsity, and on-chip weight storage. The combinatorial
diversity of board ensembles provides the computational richness that
NeuroFEM gets from programmable integrate-and-fire neurons.

**Implementation:** BingoCube core is a 600-line pure Rust crate with
BLAKE3-based cross-binding, progressive reveal, and 100% test coverage.
Published under AGPL-3.0 at `primalTools/bingoCube/`.

### 5.5 Planned: Temporal VACF Pattern Recognition

Encode velocity autocorrelation changes as sparse events. The AKD1000's
event-based processing skips zero-change particles automatically —
a natural fit for MD trajectories where most velocities change slowly
between timesteps.

---

## 6. Near-Future NPU Extension Roadmap

The following extensions are in active development or queued for the next
8-12 weeks. These are independent of any hardware partnership — they use
the AKD1000 already in the system and the existing NpuSimulator fallback.

### 6.1 Cross-Spring NPU Propagation

The NPU substrate currently lives in hotSpring. The shared toadStool
compute library makes it available to all five validation springs. Each
spring has domain-specific inference workloads that map naturally to
low-power neuromorphic coprocessing.

| Spring | NPU Use Case | Input Features | Output | Status |
|--------|-------------|----------------|--------|--------|
| **hotSpring** | Adaptive β steering (QCD) | Plaquette, Polyakov loop, susceptibility | β_c estimate, measurement allocation | ✅ **Production (Exp 022, live NPU)** |
| **hotSpring** | Transport prediction (MD) | Velocity features (8-dim) | D*, η*, λ* | ✅ ESN trained, deployment queued |
| **wetSpring** | QS Phase Classifier | Hill activation, c-di-GMP, community diversity | Anderson regime (3-class) | ✅ **Live on AKD1000** (Exp194, 18.8K Hz) |
| **wetSpring** | Bloom Sentinel | Shannon, Simpson, richness, evenness, Bray-Curtis, temp | Bloom phase (4-class) | ✅ **Live on AKD1000** (Exp194, 18.8K Hz) |
| **wetSpring** | Disorder Classifier | Shannon, Simpson, richness, evenness, W | Anderson regime (3-class) | ✅ **Live on AKD1000** (Exp194, 18.6K Hz) |
| **wetSpring** | Online Readout Evolution | ESN state → mutated weights → fitness | Adaptive classifier | ✅ **Live on AKD1000** (Exp195, 136 gen/sec) |
| **wetSpring** | Metagenomic anomaly monitor | Shannon diversity, Bray-Curtis distance, richness | Drift alert (normal/perturbed/regime-shift) | Queued |
| **airSpring** | Continuous ET₀ prediction | Temperature, humidity, radiation, wind | Evapotranspiration (mm/day) | Queued — 16 papers validated |
| **airSpring** | Irrigation trigger | Soil moisture, crop Kc, ET₀ | Irrigate (yes/no), volume (mm) | Queued (Penny Irrigation) |
| **neuralSpring** | WDM surrogate screening | Plasma (κ, Γ, density) | Transport coefficients, S(q,ω) peak | Queued — nW-01 through nW-05 |
| **neuralSpring** | Protein folding triage | Sequence embedding (ESN compressed) | Structure confidence, fold class | Planned (Phase B sovereign folding) |
| **groundSpring** | Uncertainty monitor | Observable variance, bootstrap width | Confidence flag (converged/unstable) | Queued |

Each row represents a concrete model that will be trained on GPU (f64),
quantized to int4/int8, and deployed to the AKD1000 via the existing
`NpuHardware` adapter. The NpuSimulator provides a software fallback for
development and validation on machines without the physical chip.

### 6.2 PCIe Cross-Talk: GPU→NPU Without CPU Round-Trip

Current architecture: GPU computes features → readback to CPU → CPU sends
to NPU via PCIe. The CPU round-trip adds latency and wastes bandwidth.

**Target architecture**: GPU writes features directly to a shared PCIe
memory region that the NPU reads. The CPU orchestrates but does not touch
the data path. This requires:

1. GPU writes to a mapped host buffer (wgpu `map_async` or persistent mapping)
2. NPU reads from the same physical pages (via `akida_visible_memory` or BAR DMA)
3. CPU issues dispatch commands only — no data copy

This is the same zero-copy pattern used in GPU streaming (hotSpring v0.6.13:
GPU-resident observables eliminated CPU readback for Polyakov loop). Extending
it to GPU→NPU removes the last CPU bottleneck in the inference path.

**Estimated speedup**: Eliminates ~200 μs of host-side memcpy per inference
at batch=8. Combined with the 390 μs/sample batch throughput, this brings
effective NPU latency to ~200 μs/sample — competitive with GPU inference
for small models while consuming zero GPU cycles.

### 6.3 Multi-Model NPU Pipeline

The AKD1000's 8 MB SRAM can hold dozens of small ESN readout models
simultaneously (each is 1-2 KB at int4). Rather than loading one model
at a time, we plan to deploy a bank of specialized models:

| Model | SRAM Budget | Purpose |
|-------|-------------|---------|
| Transport ESN (D*) | 1.5 KB | Diffusion coefficient prediction |
| Transport ESN (η) | 1.5 KB | Viscosity prediction |
| Transport ESN (λ) | 1.5 KB | Thermal conductivity prediction |
| Phase classifier | 2.8 KB | Weak/strong/crystalline regime |
| QS regime detector | 3.0 KB | Localized/extended/critical |
| Anomaly autoencoder | 12 KB | Simulation health monitor |
| ET₀ predictor | 4.0 KB | Evapotranspiration |
| Irrigation trigger | 2.0 KB | Scheduling decision |
| **Total** | **~28 KB** | **< 0.4% of 8 MB SRAM** |

All eight models fit trivially. The NP mesh can switch between models
via `set_variable()` weight mutation (13 ms overhead) or, if programmed
as separate sequences, route to different NP subsets concurrently.

### 6.4 Sovereign NPU Driver (Rust-Native) — PHASE C ACHIEVED

The sovereign driver roadmap has reached Phase C: direct `/dev/akida0`
access via a pure Rust driver, bypassing all vendor code.

```
Phase A: Python SDK → Rust FFI wrapper (current: npu_hw.rs)      ✅ DONE
Phase B: C++ Engine → Rust FFI to libakida.so                     ✅ DONE
Phase C: Direct ioctl / mmap on /dev/akida0 (bypass all vendor)   ✅ DONE (Feb 26, 2026)
Phase D: Rust akida_pcie driver module (full sovereignty)         In progress
```

**Phase C evidence (wetSpring V60, February 26, 2026):**

ToadStool's `akida-driver` crate provides pure Rust access to the AKD1000
via `DeviceManager::discover()` → `AkidaDevice::open()` → `write()`/`read()`.
wetSpring wraps this as `wetspring_barracuda::npu` (feature-gated `npu`).
Three validation binaries (`validate_npu_hardware`, `validate_npu_live`,
`validate_npu_funky`) exercise the full hardware path:

| Metric | Measured (Real AKD1000) |
|--------|------------------------|
| DMA throughput | 37 MB/s sustained (read + write) |
| Single inference | 54 µs (18.5K Hz) |
| Batch inference (8-wide) | 20.7K infer/sec |
| Reservoir weight loading (200×200) | 164 KB in 4.5 ms |
| Online readout switching | 86 µs for 3 swaps (weight mutation) |
| Energy per inference | 1.4 µJ |
| Coin-cell CR2032 (1 Hz) | 11 years |
| Online evolution | 136 gen/sec (real-time adaptive inference) |
| Temporal streaming | 12.9K Hz, p99=76 µs |
| PUF fingerprint | 6.34 bits entropy, deterministic dual-state |

**Zero Python. Zero C++. Zero SDK.** Pure Rust from `discover()` to `infer()`.

This follows the same evolutionary arc as ToadStool's sovereign GPU
compiler: start with vendor APIs, progressively replace with Rust-native
implementations, publish everything under AGPL-3.0.

### 6.5 Quantization Uncertainty Budget (groundSpring)

Every model deployed to the NPU loses precision: f64 → f32 → int8 → int4.
groundSpring's uncertainty quantification framework will measure the
error introduced at each quantization step:

| Transition | Expected Error | Method |
|------------|:-------------:|--------|
| f64 → f32 | ~1e-7 relative | Direct comparison |
| f32 → int8 | ~1e-2 relative | Calibration dataset |
| int8 → int4 | ~5e-2 relative | Calibration dataset |
| f64 → int4 (end-to-end) | ~5e-2 relative | Production parity test |

For each physics observable (D*, β_c, ET₀, QS regime), groundSpring will
establish the quantization error bar and determine whether int4 NPU inference
is sufficient for the decision being made. A phase classification (3 classes)
tolerates 5% error easily. A transport coefficient may not.

This error budget is hardware-independent — it applies to any int4/int8
accelerator, not just Akida. It becomes a reusable validation framework
for every future neuromorphic substrate.

---

## 7. Cross-Spring Ecosystem

hotSpring is one of five validation springs in the ecoPrimals project.
Each spring validates a different scientific domain using the same shared
BarraCUDA compute library:

| Spring | Domain | Papers | Checks | Tests | Shaders |
|--------|--------|:------:|:------:|:-----:|:-------:|
| hotSpring | Computational physics | 22 | 39 suites | 664 | 25 |
| wetSpring | Microbial ecology / metagenomics | 52 | 4,494+ | 906 | 0 (79 upstream) |
| neuralSpring | Neural architectures | 25 + 15 baseCamp | 2,250+ | 581 | 32 |
| airSpring | Atmospheric / hydrological | 16 | 1,199+ | 725 | 0 (absorbed) |
| groundSpring | Uncertainty quantification | 14 | 177 | 205 | 2 |

The NPU substrate propagates through the shared toadStool compute library.
Findings from hotSpring's AKD1000 integration directly enable wetSpring
(real-time QS prediction), neuralSpring (hardware ESN / WDM surrogates),
airSpring (continuous ET₀ monitoring), and groundSpring (quantization
uncertainty budgets).

164+ WGSL shaders circulate across the ecosystem. New hardware substrates
create new evolution pressure across all five springs simultaneously.
Adding one neuromorphic chip produces benchmarks across five scientific
domains — not one.

---

## 8. What We Want to Test Next

### 8.1 For BrainChip Specifically

We have reviewed the AKD1500 datasheet (v1.2, June 2025). The AKD1500
uses the same Akida 1.0 IP as our AKD1000, meaning all Beyond-SDK findings
(FC chain merging, batch amortization, clock modes, weight mutation,
`program_external()`) should transfer directly. Key architectural
differences that matter for our workloads:

| Property | AKD1000 (current) | AKD1500 | Impact on Our Work |
|----------|:-----------------:|:-------:|-------------------|
| PCIe | x1 Gen2 | **x2 Gen2** | **2× bandwidth — directly addresses our #1 bottleneck** (650 μs round-trip) |
| NPU mesh | 78 NPs (mixed) | 32 NPUs, 8 nodes, 3×3 | Less parallelism but cleaner architecture |
| SRAM | 8 MB shared | 100 KB/NPU (3.2 MB) + 1 MB dual-port | ESN fits trivially; wide FC may need restructuring |
| SPI host | None | Dual SPI (master + slave) | Embedded deployment without PCIe host |
| Package | PCIe card | **7×7 mm BGA169** | Chip-level — designed for custom board integration |
| GPIO | None exposed | Up to 24 | Hardware trigger/sync for physics events |
| Sleep | Software | **Hardware SLEEP pin** | Zero-power standby between inference bursts |
| PCIe ID | `1E7C:BCA1` | `1E7C:A500` | Probe code update: one constant |
| Linux driver | akida_pcie | "Supported 5.4-6.8. **No plans for updates**" | Reinforces sovereign Rust driver |

**Driver lifecycle note:** The AKD1500 datasheet states Linux kernel support
covers versions 5.4 through 6.8, with "no plans for updates." This makes
the sovereign Rust driver (Section 6.4) not merely desirable but necessary
for long-term deployment. Our Phase D Rust `akida_pcie` driver would serve
every BrainChip customer facing the same kernel version ceiling.

| Hardware | What We Would Publish |
|----------|-----------------------|
| **AKD1500 eval board (PCIe)** | Head-to-head with AKD1000 on identical ESN workloads. PCIe x2 latency measurement, batch throughput comparison, SRAM capacity mapping. Same Beyond-SDK methodology. |
| **AKD1500 SPI eval board** | First public benchmark of neuromorphic scientific computing over SPI. Latency/throughput comparison vs PCIe. Viability for embedded sensor nodes. |
| **Multi-chip reference design** | Ensemble NPU inference (separate chips for weak vs strong coupling regimes). Power scaling, inter-chip latency, mesh-to-mesh routing. |
| **Akida Pico** | Ultra-constrained inference for edge/embedded physics monitoring. Smallest viable model for real-time anomaly detection. |
| **Engineering contact** | BAR1 address space questions, program format documentation, C++ engine guidance for direct-wire access, AKD1500 register map deltas. |

### 8.2 For Other Neuromorphic Manufacturers

The integration architecture (Rust adapter, capability probe, simulator
fallback, parity testing against f64 CPU reference) is substrate-agnostic.
Adding a new neuromorphic chip requires:

1. A Rust crate wrapping the device driver (FFI or native)
2. A `predict()` implementation matching the `NpuSimulator` API
3. A `discover()` function for the capability probe

Estimated integration time for a new chip: **1-2 weeks** given driver access.

| Chip | Architecture | What We Would Benchmark |
|------|-------------|------------------------|
| Intel Loihi 2 | Programmable spiking neurons | NeuroFEM PDE solver for MD electrostatics |
| SynSense DynapCNN | Spiking CNN accelerator | Temporal pattern recognition on VACF |
| Hailo-8 | Dataflow inference (26 TOPS) | High-throughput ESN on PCIe, power parity |
| Tenstorrent Wormhole | RISC-V + tensor | Mixed-precision physics inference |

---

## 9. Future Hardware Integration Vision

The AKD1500's 7×7 mm BGA package changes what "NPU integration" means.
The AKD1000 is a PCIe card — a peripheral you plug into a motherboard.
The AKD1500 is a chip you solder onto a board. This opens architectural
possibilities that a card form factor cannot.

### 9.1 On-Board NPU: Co-Located with GPU

Current architecture: CPU ↔ PCIe ↔ GPU, CPU ↔ PCIe ↔ NPU. The CPU
mediates all data flow. The GPU→NPU path goes GPU → PCIe → CPU → PCIe → NPU.

**What the BGA form factor enables:** Solder an AKD1500 directly onto a
GPU carrier board or riser. Wire the SPI interface to the GPU board's
management controller or a small FPGA/MCU. The NPU reads feature data
from a shared memory region without traversing the CPU's PCIe root complex.

```
Current (AKD1000 PCIe card):
  GPU ──PCIe──┐
              CPU ──PCIe── NPU
              │
  Latency: GPU→CPU→NPU = ~200 μs memcpy + 650 μs PCIe

Future (AKD1500 on GPU board):
  GPU ──SPI/shared-mem── NPU (on same board)
  │
  CPU (orchestration only, not on data path)
  Latency: GPU→NPU = SPI transfer only, no PCIe round-trip
```

The AKD1500's dual SPI interface supports this: the GPU board's BMC or
an FPGA bridges between GPU memory and the NPU's SPI slave port. The
1 MB dual-port SRAM acts as a shared scratchpad.

For our physics pipeline, this means: the GPU computes HMC trajectories,
writes 8-dimensional velocity features to a shared buffer, and the NPU
reads them directly for ESN inference — no CPU in the loop, no PCIe
bridge. The 24 GPIO pins provide hardware interrupt on inference
completion.

### 9.2 Modular Compute: Learning from AMD's Chiplet Architecture

AMD's Ryzen/EPYC processors use chiplet architecture: separate compute
dies (CCDs), I/O die (IOD), and memory controllers on a single package.
Each chiplet is manufactured independently and assembled on an interposer.
This approach lets AMD mix process nodes, yield-optimize individual dies,
and scale compute by adding chiplets.

**The same principle applies to heterogeneous scientific computing:**

```
Traditional monolith:
  ┌────────────────────────────────┐
  │  GPU die (TSMC 4nm)           │
  │  All compute on one die       │
  │  f64, f32, tensor, fixed-fn   │
  └────────────────────────────────┘

Modular heterogeneous:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ GPU die  │ │ NPU die  │ │ SRAM die │
  │ f32/f64  │ │ int4/int8│ │ scratchpad│
  │ shaders  │ │ event    │ │ shared   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       └─────── interposer ──────┘
```

The AKD1500 at 7×7 mm is small enough to sit alongside a GPU die on a
multi-chip module. If BrainChip licenses the Akida IP (they offer "also
available as IP" per the datasheet), a board designer could integrate
neuromorphic inference directly into a GPU compute module without any
bus overhead at all — die-to-die interconnect at silicon speed.

This is not theoretical. AMD already does this with CPU + GPU + memory
on a single package (APU). Intel does it with CPU + FPGA (Agilex). The
question is whether neuromorphic inference earns its silicon area for
scientific computing workloads. Our benchmarks provide that answer.

### 9.3 Distributed Sensor Nodes: NPU at the Edge

The AKD1500's SPI interface and sleep pin enable a deployment model that
PCIe cards cannot: autonomous sensor nodes running inference without a
host computer.

| Node | Components | Power | Function |
|------|-----------|-------|----------|
| Soil QS monitor | AKD1500 + moisture sensor + MCU | < 100 mW | BingoCube reservoir QS regime detection |
| Weather station | AKD1500 + temp/humidity/rad + MCU | < 150 mW | ET₀ prediction, irrigation trigger |
| Bioreactor monitor | AKD1500 + OD sensor + MCU | < 100 mW | Growth phase classification, anomaly detection |
| Acoustic monitor | AKD1500 + microphone array + MCU | < 200 mW | Environmental soundscape classification |

Each node runs a BingoCube evolutionary reservoir trained on GPU (at the
homelab), quantized to int4, deployed via SPI. The NPU wakes on sensor
event, runs inference in ~1 μs of compute (668 cycles), returns to sleep.
Battery life: months to years depending on sample rate.

The trained nautilus shells (evolved board populations) propagate from
the homelab to the field nodes. Field observations feed back to the
homelab for the next generation of board evolution. The cycle:

```
Homelab (GPU + NPU):
  Train BingoCube reservoir on physics data
  → Quantize winning boards to int4
  → Deploy .fbz models to field nodes

Field (AKD1500 + MCU):
  Run inference on live sensor data
  → Log predictions + raw features
  → Return data to homelab periodically

Homelab:
  Incorporate field data into next generation
  → Evolve boards with real-world feedback
  → Deploy updated .fbz to field nodes
```

This is the "wet lab RV" vision at the hardware level. The AKD1500
isn't just a coprocessor — it's the inference brain for a network of
sovereign sensor nodes. The GPU trains, the NPU deploys, the field
generates data, the cycle continues.

### 9.4 Custom Board Design: The Long Horizon

Building custom PCBs with BGA components is not trivial — it requires
4+ layer boards, reflow soldering, and impedance-controlled routing
(especially for PCIe and SPI at speed). But it is accessible:

- **JLCPCB / PCBWay**: 4-layer boards with BGA assembly, ~$50-200 per
  prototype run
- **KiCad**: Open-source PCB design tool, handles BGA fanout
- **BGA rework stations**: ~$500 for hot-air rework capable of 7mm BGA
- **The AKD1500 datasheet provides**: complete ball map, power sequencing,
  decoupling recommendations, PCIe/SPI layout guidelines, thermal specs

A custom board with an AKD1500 + STM32 MCU + SPI flash + sensor headers
is a realistic project for someone willing to learn PCB design. The
complexity is comparable to building a custom Raspberry Pi HAT — not
trivial, but well within hobbyist/maker capability with modern tools.

The ecoPrimals philosophy applies: start with existing boards (AKD1000
PCIe eval), understand the silicon (Beyond-SDK), build the software
stack (Rust driver), and then evolve toward custom hardware when the
software demands it. The same constrained evolution that drives the
primals drives the hardware.

---

## 10. Integration Methodology

Our approach to new hardware follows a consistent pattern, demonstrated
on both GPU (DF64 discovery) and NPU (Beyond-SDK findings):

```
1. Read the datasheet
2. Test every claim against actual silicon
3. Document what matches and what doesn't
4. Find the architectural capability the SDK doesn't expose
5. Build the Rust adapter
6. Validate against f64 CPU reference (parity testing)
7. Run production physics
8. Publish everything under AGPL-3.0
```

This methodology produces higher-quality benchmarks than vendor marketing
because every measurement is cross-validated against a known-good reference
and every finding is documented with raw data.

---

## 11. References

### Repository Links

- **hotSpring**: [github.com/syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring)
- **AKD1000 hardware profile**: `metalForge/npu/akida/HARDWARE.md`
- **Beyond-SDK findings**: `metalForge/npu/akida/BEYOND_SDK.md`
- **Exploration roadmap**: `metalForge/npu/akida/EXPLORATION.md`
- **Experiment 015 (mixed pipeline validation)**: `experiments/015_MIXED_PIPELINE_BENCHMARK.md`
- **Experiment 022 (live NPU production)**: `experiments/022_MIXED_PIPELINE_NPU_PRODUCTION.md`
- **Cross-spring evolution map**: `experiments/016_CROSS_SPRING_EVOLUTION_MAP.md`
- **Changelog**: `barracuda/CHANGELOG.md` (v0.5.0 → v0.6.14)

### Key Physics Papers Reproduced

1. Murillo & Salin (2004) — Yukawa OCP molecular dynamics
2. Daligault (2006) — Transport coefficients in dense plasmas
3. Wilson (1974) — Lattice gauge theory (quenched SU(3))
4. Kogut & Susskind (1975) — Dynamical fermions
5. Anderson (1958) — Absence of diffusion in random lattices
6. Kachkovskiy & Saenz (2016) — Anderson localization spectral theory
