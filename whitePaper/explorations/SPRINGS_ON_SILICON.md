# Springs on Silicon — Cross-Domain NPU Science

**Date:** April 2026
**Status:** Living document
**License:** CC-BY-SA 4.0

The AKD1000 has 80 neural processors. The same 80 NPs that steer lattice
QCD trajectories also sentinel algal blooms and classify crop stress.
Neuromorphic hardware is domain-agnostic. This document maps how the same
NPU patterns manifest across physics, biology, and agriculture — organized
by pattern, not by spring.

---

## The Five Patterns

Every spring's NPU usage falls into one or more of five patterns. These
are not abstractions — they are the actual architectures running on
hardware, measured and validated.

```
                     ┌───────────────────────────────────┐
                     │         NPU Patterns              │
                     ├───────────────────────────────────┤
                     │ 1. Hybrid ESN                     │
                     │ 2. Microsecond Gatekeeper          │
                     │ 3. Streaming Sentinel              │
                     │ 4. Online Adaptation               │
                     │ 5. Precision Discipline             │
                     └───────────────────────────────────┘
                       ↑          ↑           ↑
                  hotSpring   wetSpring   airSpring
                  (physics)   (biology)   (agriculture)
```

---

## Pattern 1: Hybrid ESN

**Architecture:** tanh reservoir off-chip (CPU/GPU, f64), readout
on-chip (NPU, int4/int8).

The reservoir computes high-precision nonlinear dynamics. The readout
classifies. The split is natural: reservoirs need expressive
nonlinearities that the AKD1000 doesn't implement (tanh). Readouts
need speed that CPUs can't match (54 µs).

### hotSpring — Lattice QCD

| Component | Implementation |
|-----------|---------------|
| Reservoir | tanh, RS=50, spectral radius 0.9, f64 |
| Readout | InputConv(50,1,1) → FC(128) → FC(1), int4 |
| Task | Classify HMC trajectory: accept or reject? |
| Result | 80.4% accuracy, 5,978 calls over 24h |

Three variants: **ESN readout** (thermalization prediction), **SU(3) phase
classifier** (confinement/deconfinement), **WDM transport predictor**
(warm dense matter equation of state). All follow the same hybrid
architecture. All share the same InputConv → FC readout shape.

### wetSpring — Quorum Sensing

| Component | Implementation |
|-----------|---------------|
| Reservoir | tanh, RS=32, f64 |
| Readout | InputConv(8,1,1) → FC(64) → FC(3), int8 |
| Task | Classify QS phase: induction / competence / lag |
| Result | 100% CPU/NPU agreement on 1000 validation samples |

Bacterial quorum sensing oscillates between phases. The reservoir captures
the oscillation dynamics. The readout classifies which phase the population
is in. int8 because concentrations span wider dynamic range than lattice
observables.

### airSpring — Crop Stress

| Component | Implementation |
|-----------|---------------|
| Reservoir | tanh, RS=32, f64 |
| Readout | InputConv(32,1,1) → FC(64) → FC(4), int8 |
| Task | Classify crop condition: healthy / water-stress / heat / nutrient-deficient |
| Result | 48.7 µs inference, seasonal weight evolution via (1+1)-ES |

Multi-sensor input (soil moisture, leaf temperature, NDVI, humidity)
projected through the reservoir, classified on NPU. The reservoir adapts
seasonally; the readout weights are updated on-device via SRAM mutation.

### The Common Shape

All three share the same readout topology:

```
reservoir_state[N] → InputConv(N, 1, 1) → FC(width) → ... → FC(outputs)
```

The InputConv reshapes the reservoir state vector into a form the AKD1000
FC chain can process. This is not a convolution in the CNN sense — it is a
formatting layer that bridges CPU reservoir states to NPU readout weights.

**Standalone demo:** `cargo run --bin science_lattice_esn`

---

## Pattern 2: Microsecond Gatekeeper

**Architecture:** cheap NPU decision before expensive CPU/GPU/database
computation. The NPU does not do the heavy work — it decides whether the
heavy work is worth doing.

### hotSpring — HMC Trajectory Prescreen

*Should this trajectory continue?*

Each HMC trajectory on GPU costs minutes. The NPU takes 54 µs to predict
whether the trajectory will be accepted or rejected. If rejection is
predicted with high confidence, the trajectory is abandoned early, saving
all remaining compute.

**Result:** 63% of thermalization compute saved. The NPU's 54 µs decision
prevented minutes of wasted GPU time, thousands of times per day.

### wetSpring — LC-MS Spectral Triage

*Does this spectral peak need a library search?*

Mass spectrometry produces thousands of peaks per run. Library matching
is expensive (~0.3 ms per peak against a reference database). The NPU
classifies each peak in ~22 µs: "interesting" peaks get the full library
search; "background" peaks are skipped.

**Result:** 95.8% of peaks correctly triaged as background. Effective
throughput: ~45,000 spectra/s with NPU triage vs ~3,200/s without.
A **14x throughput multiplier** from a microsecond prefilter.

### airSpring — Irrigation Anomaly Detection

*Is this sensor reading anomalous?*

Agricultural sensors produce readings at 1-100 Hz. Most readings are
normal. The NPU classifies each reading in ~48 µs. Only anomalous
readings trigger the expensive response path (irrigation adjustment,
alert, logging, multi-sensor correlation).

**Result:** NPU energy consumption is 0.0009% of the active power cycle.
The decision is essentially free compared to the action it gates.

### The Common Architecture

```
Data source → NPU (µs decision) → expensive path (only if needed)
                                 → cheap path (discard / log / continue)
```

The gatekeeper pattern works whenever:
- Data arrives faster than the expensive path can process it
- Most data doesn't need the expensive path
- False negatives are acceptable (the expensive path catches what the NPU misses)

**Standalone demo:** `cargo run --bin science_spectral_triage`

---

## Pattern 3: Streaming Sentinel

**Architecture:** continuous NPU inference at the data arrival rate.
No batching, no buffering. One sample in, one decision out, next sample.

### wetSpring — Harmful Algal Bloom

Multi-channel environmental sensors (chlorophyll, phycocyanin, turbidity,
temperature, dissolved oxygen) feed continuously into the NPU. Every
23 µs, the sentinel classifies: bloom / pre-bloom / normal / instrument
fault. The classification is immediate — no batch window, no aggregation
delay.

For time-critical environmental monitoring, the difference between
"detected in 23 µs" and "detected after 256-sample batch window" can be
the difference between early intervention and contaminated water supply.

### airSpring — High-Cadence Sensor Loop

Rolling statistics (mean, variance, trend) computed over sliding windows
of sensor data. Each window update triggers an NPU classification. At
20,500 inferences per second, the NPU handles arbitrary sensor cadences
with headroom for multi-model classification (run crop stress AND
irrigation intent AND anomaly detection on the same data stream,
sequentially, within one sensor cycle).

### hotSpring — Multi-Observable Steering

Multiple lattice observables (plaquette, Polyakov loop, chiral condensate)
measured at each HMC step. Each observable feeds a separate readout head
on the NPU. The multi-head architecture runs 3 inferences sequentially
(~89 µs total for 3 heads) — still faster than a single GPU kernel launch.

### The Common Requirement

Inference latency must be less than inter-sample arrival time. The NPU's
single-sample latency (23-54 µs) is the relevant metric. Batch throughput
is irrelevant for streaming — what matters is: can the NPU finish before
the next sample arrives?

At 20,500 Hz, the NPU can sentinel at rates exceeding any environmental
sensor, any agricultural IoT device, and most scientific instrument
readout rates.

**Standalone demo:** `cargo run --bin science_bloom_sentinel`

---

## Pattern 4: Online Adaptation

**Architecture:** update readout weights on-device without full model
reprogram. The reservoir stays fixed. The readout evolves.

### airSpring — Seasonal Evolution

Crop stress signatures change with seasons. A model trained in spring
drifts by summer. airSpring uses **(1+1)-ES** (evolutionary strategy with
one parent, one offspring) to evolve readout weights on the NPU:

1. Current weights → inference on validation batch
2. Mutate weights (small Gaussian perturbation)
3. Mutated weights → inference on same batch
4. If mutant is better, keep it. Otherwise, discard.

The mutation is applied via `load_readout_weights` — direct SRAM write,
no full model reprogram. This takes microseconds, not the milliseconds
of a full program-load cycle.

### hotSpring — Multi-Head Reconfiguration

When the simulation transitions between phases (thermalization → production),
the readout objectives change. The multi-head readout is reconfigured by
swapping readout weight matrices — same SRAM mutation primitive, different
use case.

### rustChip — NpuEvolver / HybridEsn

The `NpuEvolver` and `HybridEsn` systems in `akida-driver` implement the
weight mutation primitives used by both springs:

- `mutate_weights(model_handle, slot, new_weights)` — SRAM-level weight update
- `swap_readout(model_handle, new_readout_weights)` — readout layer replacement
- `HybridEsn` manages the reservoir-on-CPU / readout-on-NPU split

### The Common Primitive

```
SRAM mutation: write new weights → immediate effect on next inference
```

No recompilation. No re-quantization. No model reload. The AKD1000's SRAM
is directly writable. Evolution happens at hardware speed.

**Standalone demo:** `cargo run --bin science_crop_classifier`

---

## Pattern 5: Precision Discipline

**Architecture:** the quantization ladder (f64 → f32 → int8 → int4) as
scientific methodology.

Each step down the ladder is validated against the step above. The gap
between precision levels reveals the information content of the problem:

| Domain | Precision | Quantization gap vs f64 | Interpretation |
|--------|-----------|------------------------|----------------|
| Physics (QCD) | int4 | ~1-3% | Narrow dynamic range: lattice observables are bounded |
| Biology (QS) | int8 | ~0.5-1% | Moderate range: concentrations span decades |
| Biology (spectra) | int8 | ~1-2% | Moderate range: intensity ratios |
| Agriculture | int8 | ~0.3-0.8% | Source noise dominates: sensors are inherently imprecise |
| Genomic | int8 | ~1-4% | Wide range: k-mer frequencies are heavy-tailed |

**Physics accepts int4** because the observables live in narrow numerical
bands. The plaquette on a 32^4 SU(3) lattice varies by ~0.01 around its
mean. int4 can represent this. If int4 accuracy degrades, that is itself a
physics signal — it means the observable has wider dynamics than assumed.

**Biology needs int8** because concentrations span orders of magnitude.
Quorum sensing signals range from nanomolar to millimolar. int4 loses the
tails. int8 preserves them.

**Agriculture tolerates int8** not because the models need it, but because
the sensor data was never f64-precise to begin with. Quantizing to int8
matches the source fidelity. The precision ladder confirms that the
bottleneck is the sensor, not the silicon.

The ladder is not a compression technique. It is a measurement instrument
for the information content of a scientific signal.

**Standalone demo:** `cargo run --bin science_precision_ladder`

---

## The 28 Zoo Models as a Map

Each model in rustChip's zoo connects to a science domain, a spring (or
BrainChip's MetaTF catalog), and one or more of the five patterns.

### Physics (ecoPrimals, hotSpring lineage)

| Model | Patterns | Key Metric |
|-------|----------|------------|
| EsnQcdThermalization | Hybrid ESN, Gatekeeper | 80.4% rejection accuracy |
| PhaseClassifierSu3 | Hybrid ESN | Confinement/deconfinement boundary |
| TransportPredictorWdm | Hybrid ESN | EOS regression from observables |
| AndersonRegimeClassifier | Hybrid ESN | Localization regime from W_c |

### Edge / NeuroBench

| Model | Patterns | Domain |
|-------|----------|--------|
| DsCnnKws | Streaming | Audio keyword spotting |
| EsnChaotic | Hybrid ESN | Chaotic time series (Mackey-Glass) |
| EcgAnomaly | Streaming Sentinel | Cardiac anomaly detection |

### Vision / Audio / Spatiotemporal (BrainChip MetaTF)

| Model | Patterns | Domain |
|-------|----------|--------|
| AkidaNetImagenet | Gatekeeper | Image classification |
| AkidaNetPlantvillage | Gatekeeper, Precision | Plant disease (agriculture) |
| CenterNetVoc | Gatekeeper | Object detection |
| DsCnnKws | Streaming | Keyword spotting |
| TennRecurrentSc12 | Streaming | Speech commands |
| TennSpatiotemporalDvs128 | Streaming | Event-driven (DVS) gesture |
| GxnorMnist | Precision | Binary neural network |
| YoloVoc / YoloWiderface | Gatekeeper | Detection |
| PointNetPlusModelnet40 | — | 3D point cloud |

### Standalone

| Model | Patterns | Purpose |
|-------|----------|---------|
| MinimalFc | Precision | Minimal proof-of-life |

The full zoo is registered in `crates/akida-models/src/zoo.rs` (28 variants).
Every model is runnable via `cargo run --bin preserve_{domain}`.

---

## The Thesis

Neuromorphic hardware is a scientific instrument. The AKD1000 does not
care whether the input vector represents lattice QCD observables, bacterial
quorum signals, LC-MS spectra, or soil moisture readings. It runs
InputConv → FC → output in 54 µs regardless.

The science is in the reservoir, the signal processing, the experimental
design. The silicon is the execution substrate. The five patterns —
hybrid ESN, gatekeeper, sentinel, adaptation, precision discipline — are
domain-independent. They work because the problems share a structure:
cheap classification gates expensive computation.

This is why one chip, one driver, one software stack serves three
springs across physics, biology, and agriculture. The domains are
different. The pattern is the same.

**The springs carry the full science. rustChip carries the silicon proof.**
**Run any demo:** `cargo run --bin science_{domain}`

---

## References

- [WHY_NPU.md](WHY_NPU.md) — The foundational neuromorphic argument
- [NPU_FRONTIERS.md](NPU_FRONTIERS.md) — What comes next
- [TANH_CONSTRAINT.md](TANH_CONSTRAINT.md) — Bounded ReLU vs tanh
- [hotSpring](https://github.com/syntheticChemistry/hotSpring) — Lattice QCD
- [wetSpring](https://github.com/syntheticChemistry/wetSpring) — Sentinel microbe
- [airSpring](https://github.com/syntheticChemistry/airSpring) — Agricultural IoT
- [sporePrint / primals.eco](https://primals.eco) — Public verification
