# AKD1000 Exploration: Novel Applications for Computational Physics

**Date**: February 20, 2026
**Hardware**: BrainChip AKD1000 @ PCIe `08:00.0`, driver loaded, `/dev/akida0`
**SDK**: Akida 2.19.1 installed, hardware detected (78 NPs), benchmarked
**Status**: First hardware benchmarks complete, deployment pipeline validated

---

## The Core Question

What can a 30mW neuromorphic processor do for molecular dynamics and
computational physics that a 200W GPU cannot?

The answer is NOT "the same thing but lower power." The answer is
"fundamentally different workloads that exploit the architecture."

---

## Tier 1: Validated & Ready to Deploy

### 1A. ESN Transport Prediction (from hotSpring reservoir work)

**What**: Echo State Network predicts D* from short velocity trajectories.
**How**: Train on GPU (f64), quantize weights to int4, deploy to AKD1000.
**Why NPU**: Inference is ~2,950 MACs. On NPU: <5mW, <1μs. On GPU: overkill.

```
GPU (MD simulation)                    NPU (ESN inference)
┌─────────────────┐                   ┌─────────────────┐
│ Velocity Verlet │──velocity─────────│ W_in · features │
│ Yukawa forces   │  features         │ W_res · state   │
│ 200W, 250 step/s│  (8 × f32→uint8) │ W_out · state   │
└─────────────────┘                   │ 5mW, ~1μs       │
                                      └────────┬────────┘
                                               │
                                          D* prediction
```

**Implementation path**:
1. Install MetaTF SDK (`pip install akida`)
2. Build Keras model matching our ESN topology
3. Quantize with QuantizeML (int8 input, int4 internal)
4. Convert with CNN2SNN → `.fbz` model
5. Map to device, benchmark FPS and power
6. Compare prediction accuracy against f64 Rust CPU reference

**Risk**: Quantization to int4 may degrade ESN dynamics. The tanh
activation is critical for reservoir computing — replacing it with
ReLU (the only Akida 1.0 activation) changes the reservoir's computational
properties. Mitigation: benchmark quantization error explicitly.

**Estimated effort**: 2-3 days
**Novelty**: Low (standard deployment path)
**Value**: High (proves the GPU→NPU handoff pipeline end-to-end)

### 1B. Phase Classifier for MD Regime Detection

**What**: Classify (κ, Γ) input parameters into physical regimes
(weakly coupled plasma / strongly coupled liquid / crystalline).
**How**: Small ConvNet or MLP trained on transport coefficient patterns.
**Why NPU**: Real-time regime detection at microwatt power. Could run
continuously alongside MD simulation, triggering algorithm changes
(e.g., switch from all-pairs to cell-list when entering liquid regime).

**Implementation path**:
1. Generate training data from existing Sarkas/Daligault transport fits
2. Train 3-class classifier in Keras (tiny: 3 layers, <1000 params)
3. Quantize + convert → deploy to AKD1000
4. Benchmark classification accuracy and latency

**Risk**: Trivial workload for the hardware. Mainly useful as a
deployment validation stepping stone.

**Estimated effort**: 1 day
**Novelty**: Low
**Value**: Medium (validates the toolchain; useful for adaptive simulations)

---

## Tier 2: Novel Applications (Exploit the Architecture)

### 2A. Sparse Force Surrogate via Event-Based Inference

**What**: Train a neural network to approximate short-range pairwise forces
as a function of particle separation. Deploy on NPU for real-time evaluation.

**Why this is interesting**: In MD, the force calculation dominates compute.
For Yukawa: `F = (Γ/r²)(1 + κr) exp(-κr) r̂`. This involves transcendentals
(exp, div) that are expensive on any substrate. A quantized neural surrogate
could approximate the force-distance curve with pure integer MACs.

```
Classical force:   F(r) = Γ/r² × (1 + κr) × exp(-κr)  ← 1 div, 1 exp, 3 mul
Neural surrogate:  F(r) ≈ NN(r_quantized)               ← ~100 int4 MACs
NPU advantage:     Only compute F for non-zero (close neighbor) pairs
                   Event-based: skip distant particles automatically
```

**Critical analysis**: This is probably a bad idea for two reasons:
1. The force function is smooth and cheap. A lookup table beats any NN.
2. Quantization to int4 means only 16 force levels. Physics needs >10
   significant figures for energy conservation.

**Verdict**: Interesting to benchmark, unlikely to be practical for
production MD. However, it's a stepping stone to force-field learning
for complex potentials (EAM, many-body) where the function IS expensive.

**Estimated effort**: 3-4 days
**Novelty**: Medium
**Value**: Low for Yukawa, potentially high for complex potentials

### 2B. Temporal Pattern Recognition on VACF Trajectories

**What**: Instead of computing D* via Green-Kubo integration (which requires
the full VACF), use the NPU to recognize temporal patterns in raw velocity
time series and directly predict transport coefficients.

**Why NPU excels here**: The AKD1000's event-based processing naturally
handles sparse temporal data. In an MD simulation, most particle velocities
change slowly between timesteps. If we encode velocity *changes* as events,
the NPU only processes particles that actually moved significantly.

```
Timestep t:   v = [0.5, -0.3, 0.7, 0.0, -0.1, ...]
Timestep t+1: v = [0.5, -0.3, 0.8, 0.0, -0.1, ...]
                                 ↑
Events:       [particle_2: Δv=+0.1]   ← only 1 event, not N values
```

**Implementation approach**:
- Encode velocity changes as sparse uint8 events (magnitude + sign)
- `InputConvolutional` layer processes event stream with temporal receptive field
- `FullyConnected` readout predicts transport coefficient
- Train on GPU-generated MD data, deploy to NPU

**Risk**: Akida 1.0 doesn't have native temporal convolutions (TENNs).
We'd need to manually unroll time windows into spatial dimensions.
This works but is less elegant than Akida 2.0's `BufferTempConv`.

**Estimated effort**: 1 week
**Novelty**: High
**Value**: High (general-purpose MD observable prediction)

### 2C. On-Chip Anomaly Detection for Simulation Health

**What**: Deploy a small anomaly detector on the NPU that continuously
monitors MD simulation observables (KE, PE, temperature, pressure) and
flags when the simulation has gone wrong (energy blow-up, thermostat
failure, unphysical configurations).

**Why NPU**: Runs at microwatt power. No GPU cycles stolen from the
simulation. Continuous monitoring without overhead.

**Implementation**:
- Train autoencoder on "healthy" simulation observable trajectories
- Reconstruction error > threshold = anomaly
- On-chip learning (STDP) could adapt to new simulation regimes

**Risk**: The on-chip learning on AKD1000 is 1-bit only, which limits
the autoencoder's expressiveness. Fixed-weight deployment is more practical.

**Estimated effort**: 3-4 days
**Novelty**: Medium
**Value**: Medium (nice-to-have for long simulations)

---

## Tier 3: Frontier (Push Hardware Boundaries)

### 3A. Neuromorphic PDE Solver (Inspired by Sandia's NeuroFEM)

**What**: Implement a spiking neural network that solves the Poisson equation
or other PDEs directly on the AKD1000, following Sandia National Lab's
NeuroFEM approach on Intel Loihi 2.

**Background**: Sandia showed that spiking networks can solve sparse linear
systems (Ax = b) by mapping the matrix A to synaptic weights and using
neural dynamics (integrate-and-fire + reset) as an iterative solver. The
steady-state spike rates converge to the solution x.

**Why this matters for ecoPrimals**:
- Electrostatics in MD (Poisson equation for long-range Coulomb)
- PPPM/Ewald methods already decompose the problem into real-space (short-range)
  and k-space (long-range). The k-space part is a convolution = PDE solve.
- If the NPU can solve the k-space part while the GPU handles short-range
  forces, we have true heterogeneous MD.

**Challenge**: NeuroFEM was built on Loihi 2, which has a fundamentally
different architecture (128-compartment neurons with programmable dynamics
vs. AKD1000's simpler event-based model). The AKD1000 lacks:
- Programmable neuron dynamics (no custom integrate-and-fire)
- Feedback connections (feed-forward only in standard mode)
- Multi-compartment neurons

We'd need to map the iterative solve into a feed-forward multi-pass
approach, which is possible but loses the elegance.

**Estimated effort**: 2-3 weeks
**Novelty**: Very high
**Value**: Very high if it works; research-grade exploration

### 3B. Quantized Reservoir Computing with Native Sparsity

**What**: Instead of training an ESN on GPU and deploying weights to NPU,
build a reservoir that is **native to the AKD1000's compute model**.

**Key insight**: Traditional ESN uses f64 weights with tanh activation.
The AKD1000 uses int4 weights with ReLU activation. These are different
mathematical objects. Instead of trying to approximate the f64 ESN on
int4 hardware, design a reservoir that is **optimal for int4 + ReLU**.

**Approach**:
1. Replace tanh with ReLU (the NPU's native activation)
2. Replace f64 weights with 4-bit ternary weights (-1, 0, +1)
3. Exploit sparsity: the reservoir weight matrix is 80% sparse by design;
   on the AKD1000, this means 80% fewer MACs automatically
4. The reservoir's computational properties change (no negative states,
   different echo state property conditions) but may still work for
   transport prediction if carefully designed

**Literature support**: Binary/ternary reservoirs have been studied
(Appeltant et al., Larger et al.) and can achieve competitive performance
with careful hyperparameter tuning.

**Why this is better than Tier 1A**: Instead of losing accuracy by
quantizing a float ESN, we design a system that's native to the hardware.
The int4 ReLU reservoir is the "true" math — matching what the transistors
actually compute.

**Estimated effort**: 1-2 weeks
**Novelty**: Very high
**Value**: High (aligns with ecoPrimals philosophy: shader-originating math)

### 3C. Direct-Wire NPU Access via C++ Engine

**What**: Use the Akida Engine C++ library to bypass the Python SDK and
program the AKD1000 at the register level.

**Why**: The Python SDK assumes neural network workloads. The C++ Engine
exposes `read()` / `write()` on arbitrary addresses, `scratch_memory()`
for DMA, and `akida_visible_memory()` for the full addressable space.

**Exploration targets**:
1. **Map the address space**: What registers exist? What do they control?
2. **Raw weight upload**: Can we write arbitrary 4-bit values to the
   synaptic weight SRAM without going through the model compilation pipeline?
3. **Custom program injection**: The `program()` method accepts a binary
   blob. Can we reverse-engineer the program format and generate our own?
4. **DMA streaming**: Use `scratch_memory` for streaming velocity features
   from GPU → NPU without going through host memory.

**This is the GPU f64 parallel**: The Python SDK is like CUDA — it presents
a limited view of the hardware. The C++ Engine is like Vulkan/wgpu — it
lets us talk to the metal. And just like we found the DF64 core-streaming
strategy by benchmarking actual hardware, we might find capabilities the
SDK doesn't advertise.

**Estimated effort**: 1-2 weeks (highly exploratory)
**Novelty**: Very high
**Value**: Unknown (could be transformative or a dead end)

### 3D. SRAM-Enabled Research Directions (rustChip)

With `SramAccessor` and `probe_sram`, BAR1 SRAM is now directly readable
and writable. New research directions enabled:

| Direction | Approach | Value |
|-----------|----------|-------|
| **Weight inspection** | Read back NP SRAM after load; compare against expected weights from `.fbz` | Model load verification, debugging quantization |
| **Online learning verification** | Read SRAM before/after STDP updates; confirm hardware learning state | Validate on-chip learning beyond SDK |
| **PUF via SRAM noise** | Power-cycle SRAM regions; measure startup bit patterns; extract device fingerprint | Device attestation, anti-cloning |
| **Multi-tenant isolation** | Load tenant A, read SRAM; load tenant B, read SRAM; verify no cross-tenant bleed | Security for shared NPU deployment |

These directions exploit the `NpuBackend::read_sram()`, `verify_load()`, and
`LoadVerification` capabilities added to rustChip. `bench_exp002_tenancy`
Phase 2 with `--hw` flag exercises SRAM isolation verification.

---

## Tier 4: Cross-Substrate Orchestration

### 4A. Heterogeneous MD Pipeline: GPU + NPU

The endgame for ecoPrimals NPU integration:

```
┌──────────────────────────────────────────────────────────┐
│                    Host CPU (i9-12900K)                    │
│  Orchestrator: allocate work, manage data flow             │
│                                                            │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │   RTX 4070 (200W)   │    │   AKD1000 (30mW)         │ │
│  │                     │    │                          │ │
│  │ • Force calculation │    │ • ESN inference (D*)     │ │
│  │ • Velocity Verlet   │    │ • Phase classification   │ │
│  │ • Cell-list build   │    │ • Anomaly detection      │ │
│  │ • Thermostat        │    │ • Observable prediction  │ │
│  │ • VACF accumulation │    │                          │ │
│  │                     │    │ Every 100 MD steps:      │ │
│  │ Continuous: 250 Hz  │    │ features → NPU → D*     │ │
│  │                     │    │ Latency: ~10 μs          │ │
│  └──────────┬──────────┘    └──────────┬───────────────┘ │
│             │                          │                  │
│             └──── velocity features ───┘                  │
│                                                            │
│  Total system power: ~200W (GPU dominates)                 │
│  NPU overhead: 0.015% of system power                     │
│  NPU value: real-time observables without stealing GPU time│
└──────────────────────────────────────────────────────────┘
```

### 4B. Multi-NPU Ensemble (if we get a second AKD1000)

ToadStool's `DualChipEnsemble` from `akida-reservoir-research` is
designed for exactly this:

- **Chip 1**: Weak coupling specialist (Γ < 50)
- **Chip 2**: Strong coupling specialist (Γ ≥ 50)
- **Ensemble output**: Weighted average based on input Γ

Two AKD1000 boards: ~60mW total, full phase diagram coverage.

---

## Priority Ranking

| # | Application | Effort | Novelty | Value | Priority |
|---|-------------|--------|---------|-------|----------|
| 1A | ESN D* deployment | 2-3d | Low | High | **P0** — proves the pipeline |
| 1B | Phase classifier | 1d | Low | Med | **P0** — validates toolchain |
| 2B | Temporal VACF patterns | 1w | High | High | **P1** — novel physics application |
| 3B | Native int4 reservoir | 1-2w | V.High | High | **P1** — aligns with ecoPrimals philosophy |
| 2A | Force surrogate | 3-4d | Med | Low | **P2** — benchmark only |
| 2C | Anomaly detection | 3-4d | Med | Med | **P2** — nice-to-have |
| 3C | Direct-wire C++ | 1-2w | V.High | ? | **P2** — high risk/reward exploration |
| 3A | Neuromorphic PDE | 2-3w | V.High | V.High | **P3** — research frontier |
| 4A | Heterogeneous pipeline | 2-3w | High | V.High | **P3** — requires P0+P1 complete |

---

## What We Learned (Feb 20, 2026 Hardware Session)

### Completed

1. **SDK installed** (Akida 2.19.1), hardware detected: 78 NPs (CNP1×78,
   CNP2×54, FNP2×4, FNP3×18)
2. **Device permissions**: udev rule exists but doesn't trigger on boot.
   Fixed via `pkexec chmod 666 /dev/akida0`. **Evolution target**: solve in
   Rust (`DeviceManager` with proper udev/capabilities handling — tracked in
   `metalForge/README.md` Remaining Work).
3. **Direct model build**: Bypassed the Keras→QuantizeML→CNN2SNN pipeline
   entirely. Built Akida model via native API with `set_variable()` weight
   injection. This is the "direct wire" path.
4. **Hardware benchmark**: ESN readout (50→1 FC) runs in **668 inference
   clocks** on 1 FNP3 node. 752-byte program. But PCIe x1 Gen2 latency
   adds ~650 μs overhead, making hardware 100× slower than CPU for this
   tiny workload.
5. **Power measurement**: Board floor is ~918 mW. Chip inference power is
   below measurement threshold — literally too little to detect above the
   PCIe board's baseline draw.

### Key Discovery: SDK Limits ≠ Hardware Limits (Feb 19-20 Deep Probe)

**See `BEYOND_SDK.md` for full details with measurements.**

Several SDK assumptions were overturned by direct hardware testing:

- ~~InputConv requires 1 or 3 channels~~ → **Any channel count works** (tested 1-64).
  InputConv always runs in SW regardless; channel count is irrelevant to HW FC layers.
- ~~FC layers are independent HW sequences~~ → **All FC layers merge into one HW pass**
  via intra-mesh SkipDMA. Deep networks (2-8 layers tested) add near-zero latency.
- ~~PCIe latency is fixed at ~650μs~~ → **Batch=8 amortizes to 390μs/sample** (2.4× speedup).
- ~~One clock mode~~ → **Three modes**: Performance (901mW), Economy (739mW, 19% slower),
  LowPower (658mW, 9.3× slower).
- ~~Max FC width ~hundreds~~ → **Tested to 8192 neurons wide**, all map to HW.
  Latency crossover (PCIe vs compute) at ~width=512.
- **Multi-output is free**: 50→1024→10 costs the same as 50→1024→1.
- **16GB PCIe BAR1** address space (vs 8MB SRAM spec). Full NP mesh decode range.
- **program_external()**: Raw FlatBuffer program injection. The "metal" API.
- **C++ engine**: SkipDMA, on-chip learning registers, 51-bit threshold SRAM,
  `format_mesh_registers_set_output_to_NPs()` — capabilities far exceed SDK.

### Real Hardware Constraints (confirmed silicon, not SDK)

- **No tanh** — only bounded ReLU in Akida 1.0. Changes reservoir dynamics.
- **Feed-forward only** — no recurrence in hardware. Host drives the loop.
- **InputConv HW mapping** needs kernel ≥ 3 AND channels 1 or 3 (for HRC).
  But SW fallback is fine — the FC layers do the real work.
- **1-bit on-chip learning** — SDK limits to binary weights + binary activations.
  C++ engine symbols suggest broader support; not yet tested.
- **PCIe x1 Gen2** — ~650μs minimum round-trip. Batch amortizes but doesn't eliminate.

### Revised Understanding

The AKD1000 is more capable than the SDK presents. It's a **programmable
sparse integer mesh processor** with:
- Massive FC scaling (8192+ width)
- Near-zero cost for depth (SkipDMA merges layers)
- Batch pipelining through the PCIe interface
- Three power/performance modes
- Raw program injection capability
- 16GB address decode (most unexplored)

For ecoPrimals, this changes the calculus:
- **Deep FC readouts are viable**: 50→256→256→256→1 at 909μs, not just 50→1
- **Multi-observable prediction**: D*, η, λ simultaneously — multi-output is free
- **Batch prediction**: Buffer 8 MD steps → 2.4× throughput
- **Power-optimized continuous monitoring**: Economy mode saves 162mW
- **Future direct-wire**: program_external + BAR access for custom mesh programming

### Immediate Next Steps

1. **Deploy actual ESN weights at batch=8 in Economy mode** — combine three
   discoveries for real transport prediction: 50→128→1 at ~400μs/sample
2. **Multi-observable readout** — predict D*, viscosity, thermal conductivity
   in one forward pass. Multi-output is free.
3. **Build deep FC network** — 50→256→256→1 exploiting the single-pass merge.
   Compare accuracy vs single-layer readout.
4. **program_external experimentation** — Generate custom FlatBuffer programs
   and inject via the raw API. Understand the routing configuration.
5. **BAR1 deep probe** — Map the 16GB address space to find NP SRAM regions.
   Goal: direct memory-mapped weight writes (bypass set_variable overhead).

---

## Research Context: What Others Are Doing

### Sandia NeuroFEM (2025)
- Solved Poisson equation on Loihi 2 using spiking dynamics
- Achieved "meaningful numerical accuracy" with ideal scaling
- Key insight: map sparse matrix to synapses, use neural dynamics as solver
- **Relevance**: Direct model for our Tier 3A exploration

### KTH Akida vs NVIDIA Comparison (2025)
- Compared AKD1000 and NVIDIA GPU for SNN workloads
- Found AKD1000 dominates in power efficiency for small models
- **Relevance**: Confirms our ESN deployment strategy (small model → NPU)

### Intel Loihi 2 Reservoir Computing (2025)
- Principled reservoir computing with Sigma-Pi neurons
- Separated memory buffering from feature expansion
- **Relevance**: More sophisticated reservoir than our ESN, but requires
  Loihi 2's programmable neuron model (unavailable on AKD1000)

### Nature: Principled Neuromorphic Reservoir Computing (2025)
- Polynomial feature expansion via multiplicative neuron interactions
- Demonstrated chaotic dynamics prediction (Lorenz, Hénon map)
- **Relevance**: Validates reservoir computing for dynamical systems
  prediction, exactly our VACF application

### BrainChip Performance Optimization (Nov 2025)
- 3.86× runtime improvement, 3.38× energy reduction via sparsity-aware training
- **Relevance**: Our ESN's natural sparsity (post-ReLU quantization) should
  benefit from similar optimization

---

## Addendum: Exp 020 NPU Characterization Campaign (Feb 26, 2026)

### Campaign Results

Ran comprehensive NPU characterization with 1800 trajectories (12 β-points × 150 traj)
using three specialized ESN models and six pipeline placements. All via NpuSimulator
(CPU f32 emulation of AKD1000) — ready for hardware validation.

### Models Trained and Validated

| Model | Architecture | Accuracy | Key Finding |
|---|---|---|---|
| Thermalization Detector | 10-in → 50 reservoir → 1 out | **87.5%** | Saves 61.8% of therm budget (3.15h projected) |
| Rejection Predictor | 5-in → 50 reservoir → 1 out | **96.2%** | Near-perfect but limited by high acceptance at 4⁴ |
| 6-Output Multi-Model | 8-in → 50 reservoir → 6 out | 33.3% phase (small N) | All 6 outputs finite, multi-output free confirmed |

### NPU Placement Discovery

| Placement | Best Use Case | Time Impact |
|---|---|---|
| Pre-thermalization (A) | Detect equilibrium early | **-3.15h** (biggest win) |
| Mid-trajectory (B) | Abort rejected trajectories | Potential at large lattices |
| Post-trajectory (C) | Phase classification (baseline) | Reference approach |
| Inter-beta steering (D) | Adaptive β selection | Needs more training data |
| Pre-run bootstrap (E) | Eliminate seed scanning | Warm-start from Exp 013/018 |
| All combined (F) | Maximum pipeline optimization | 87.5% accuracy, 390 traj saved |

### Characterization Metrics

- **Latency**: p50=331µs, p95=403µs, p99=520µs (simulator; hardware expected ~390µs)
- **Drift**: 0.0 over 50 batches (deterministic)
- **Mutation**: 0.015ms (simulator); 14ms target on hardware
- **Accuracy vs N**: 100% phase accuracy achieved with 10 training β-points

### Files

- Campaign binary: `barracuda/src/bin/npu_experiment_campaign.rs`
- Results: `/tmp/hotspring-runs/v0614/npu_campaign_results.jsonl`
- Full report: `experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md`
- Akida feedback: `wateringHole/handoffs/AKIDA_BEHAVIOR_REPORT_FEB26_2026.md`

---

## Addendum: Cross-Substrate ESN Comparison (Exp 021)

### NPU vs GPU vs CPU: Same Workload, Different Silicon

The cross-substrate benchmark (Exp 021) ran identical ESN inference on
CPU-f64, CPU-f32, GPU-f32, and NPU-simulator with reservoir sizes 8–1024.

**NPU advantage zone**: Streaming single-step inference at 2.8 μs/step.
GPU dispatch overhead (~3.5ms per submit cycle) makes it 1000× slower
than NPU for single-step screening. The NPU naturally maps to the
"examine each trajectory as it arrives" pattern.

**GPU advantage zone**: RS ≥ 512. At RS=1024 the GPU achieves 8.2× speedup
over CPU-f64. For future high-dimensional embedding (e.g., full Wilson
loop configuration vectors as ESN input), GPU-resident reservoir computing
becomes the right substrate.

**NPU capability envelope confirmed**:
- Threshold detection: 100% accuracy
- Streaming inference: 2.8 μs/step (matches CPU)
- Multi-output (1–8 heads): No latency penalty, max|Δ| < 3e-7
- Weight mutation: 141 μs per reload cycle
- QCD thermalization: 100% accuracy (38/38)
- Multi-observable anomaly scoring: RMSE = 0.003

**Precision finding**: f32→f64 divergence is RS-dependent. RS=100 shows
~7% relative error (precision "sweet spot" where errors compound without
enough neurons to dilute). RS ≤ 50 and RS ≥ 200 show < 3% error.

### Substrate Assignment for metalForge Pipeline

```
GPU (HMC physics, DF64)
  ↓ observable stream
NPU (2.8μs screening, threshold/anomaly/thermalization)
  ↓ flagged trajectories
CPU (f64 precision verification, readout arbitration)
```

For future large-reservoir applications:
```
GPU₁ (physics) → GPU₂ (large RS ESN) → NPU (lightweight screening)
```

### Files

- Benchmark: `barracuda/src/bin/cross_substrate_esn_benchmark.rs`
- Experiment log: `experiments/021_CROSS_SUBSTRATE_ESN_COMPARISON.md`
- Results JSONL: `/tmp/hotspring-runs/exp021/cross_substrate_results.jsonl`
