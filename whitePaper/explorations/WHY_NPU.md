# Why NPU? — The Neuromorphic Argument

**Date:** April 2026
**Status:** Living document
**License:** CC-BY-SA 4.0

This is not a datasheet. This is an argument for why neuromorphic hardware
changes what science can do — grounded in production measurements from three
independent research domains running on the same 80-neuron AKD1000 chip.

---

## 1. The Microsecond Decision

A GPU inference call takes milliseconds. A CPU classifier takes tens of
milliseconds. An AKD1000 inference takes **54 microseconds**.

That difference is not incremental. It is architectural. At 54 µs, you can
put a classifier *inside* a simulation loop — not beside it, not after it,
not in a separate pipeline stage. Inside. Between every step.

hotSpring proved this. Lattice QCD simulations generate Hybrid Monte Carlo
(HMC) trajectories on GPU. Each trajectory costs minutes of compute. The
question at every step: *should this trajectory continue, or should we
reject it and start over?* A bad trajectory wastes all the compute invested
so far. A good prediction saves everything.

An NPU running an Echo State Network readout answers that question in 54 µs.
The GPU doesn't notice it happened. The simulation loop doesn't stall. Over
24 hours of continuous operation, the AKD1000 made **5,978 live steering
decisions**, achieving **63% thermalization savings** and **80.4% rejection
prediction accuracy** on a 32^4 SU(3) lattice (Exp 022).

The physics didn't change. The algorithm didn't change. What changed was
that a decision that used to be too expensive to make at every step became
cheap enough to make at every step. The NPU didn't replace the GPU — it
made the GPU's work worth more.

**Standalone demo:** `cargo run --bin science_lattice_esn`
**Spring:** [hotSpring](https://github.com/syntheticChemistry/hotSpring)

---

## 2. Energy as a Constraint, Not a Cost

Each AKD1000 inference consumes approximately **1.4 microjoules**.

For context: a single GPU kernel launch consumes millijoules. A CPU
inference in PyTorch consumes tens of millijoules. The NPU inference is
three to four orders of magnitude cheaper in energy.

This matters not because electricity is expensive (it isn't, at lab scale),
but because **energy is a proxy for what you can afford to do everywhere**.
When a decision costs 1.4 µJ, you can make it:

- At every HMC step in a lattice simulation (hotSpring)
- At every sensor reading in an agricultural IoT loop (airSpring)
- At every spectral peak in an LC-MS run (wetSpring)
- At every frame in a DVS event stream

The constraint inverts. Instead of "can we afford to classify this?", the
question becomes "why would we not classify this?" The NPU makes the
decision layer thermodynamically negligible.

**Measured:** Economy mode draws ~120 mW continuous. At 18,500 inferences
per second, that is 6.5 µJ per inference including idle power. The
marginal energy per inference (above idle) is ~1.4 µJ.

---

## 3. The Hybrid Reservoir

The AKD1000 does not implement tanh. It implements bounded ReLU
(piecewise linear, int4/int8 quantized). This looks like a limitation.
It is an architectural choice.

An Echo State Network has two parts:

1. **The reservoir** — a recurrent network with fixed random weights.
   Its job is to project input signals into a high-dimensional nonlinear
   space. The dynamics are chaotic, sensitive, expressive. They need
   high-precision nonlinearities (tanh, typically in f64).

2. **The readout** — a simple linear layer (or shallow FC chain) that
   maps reservoir states to outputs. Its job is fast, cheap, repeated
   classification. It needs speed, not precision.

The hybrid architecture splits these naturally:

```
CPU/GPU (f64)              NPU (int4/int8)
┌──────────────┐           ┌──────────────┐
│  Reservoir   │           │  Readout     │
│  tanh(Wx+b)  │──state──> │  InputConv   │──output──> decision
│  RS=50       │           │  FC(128)     │
│  f64 dynamics│           │  FC(1)       │
└──────────────┘           └──────────────┘
   creative                   fast
   expressive                 cheap
   high-precision             hardware-quantized
```

The reservoir is the creative part. It captures the dynamics of the
physical system — lattice configurations, quorum sensing oscillations,
spectral shapes. It runs in f64 because the dynamics demand it.

The readout is the fast part. It turns reservoir states into yes/no
decisions. It runs on hardware because speed and energy matter more than
the last decimal of precision. The quantization gap (f64 readout vs int4
hardware) is measurable (~1-3% on physics benchmarks) and acceptable —
because the reservoir already did the hard work.

Three springs, three domains, same architecture:

| Spring | Reservoir | Readout | Precision |
|--------|-----------|---------|-----------|
| hotSpring (physics) | tanh, RS=50, f64 | InputConv(50)→FC(128)→FC(1) | int4 |
| wetSpring (biology) | tanh, RS=32-64, f64 | InputConv(N)→FC(64)→FC(3) | int8 |
| airSpring (agriculture) | tanh, RS=32, f64 | InputConv(32)→FC(64)→FC(4) | int8 |

**Exploration:** [TANH_CONSTRAINT.md](TANH_CONSTRAINT.md) — the full
analysis of bounded ReLU vs tanh and why the hybrid approach works.

---

## 4. Precision as a Ladder

The path from research to hardware is a quantization ladder:

```
f64  →  f32  →  int8  →  int4
 ↑       ↑       ↑        ↑
 │       │       │        │
 │       │       │        └── AKD1000 native (physics workloads)
 │       │       └─────────── AKD1000 native (biology workloads)
 │       └─────────────────── Software VirtualNPU validation
 └─────────────────────────── Research baseline (CPU/GPU)
```

Each step down the ladder is validated against the step above. The
springs treat this not as an engineering compromise but as **scientific
methodology** — each precision level reveals something about the problem:

**Physics (int4):** Lattice QCD observables have narrow dynamic range.
The plaquette, the Polyakov loop, the chiral condensate — they all live
in well-defined numerical bands. int4 suffices because the physics is
bounded. The quantization ladder *confirms* this: if int4 accuracy
degrades, it means the observable has wider dynamic range than assumed,
which is itself a physics result.

**Biology (int8):** Quorum sensing phase classification, spectral
triage, bloom detection — these have wider dynamic range than physics
(concentrations span orders of magnitude) but still bounded. int8
provides the headroom. Dropping to int4 degrades biology models by
~4-7% — the precision ladder reveals that biological signals genuinely
occupy more of the representable range.

**Agriculture (int8):** Soil moisture, leaf temperature, NDVI — the
sensor data is inherently noisy and low-precision. int8 matches the
source fidelity. Quantization does not degrade performance because the
input was never float64-precise to begin with.

The ladder is a scientific instrument. It doesn't just compress models —
it probes the information content of the problem domain.

**Standalone demo:** `cargo run --bin science_precision_ladder`

---

## 5. Streaming as a Native Mode

GPU inference has setup cost: kernel launch, memory transfer, context
switch. Even after warmup, the overhead is microseconds to milliseconds
per batch. GPUs amortize this by batching — process 32 or 256 samples
at once. This requires buffering, introduces latency, and breaks the
streaming abstraction.

NPU inference has no setup cost. The model is loaded once. Each
inference is a DMA write (input) and DMA read (output), with the NPU
processing in between. At 54 µs per inference, the NPU naturally
operates in streaming mode — one sample at a time, continuously.

This matters for three patterns the springs use:

**Sentinel (wetSpring):** Harmful algal bloom monitoring runs
continuously. Every 23 µs, a new environmental sensor reading goes
through the NPU. There is no batch. There is no buffer. The NPU is a
continuous classifier: data in, decision out, next sample. If a bloom
is detected, the response is immediate — not "after the batch finishes."

**Steering (hotSpring):** HMC trajectory evaluation must happen between
simulation steps. There is no batch of trajectories to evaluate — there
is one trajectory, right now, and the simulation is waiting for the
answer. The NPU's single-sample latency is the metric that matters.
Batch throughput is irrelevant.

**Edge IoT (airSpring):** Agricultural sensors produce readings at
1-100 Hz. The NPU can process 18,500 readings per second. It will
never be the bottleneck. This headroom enables rolling statistics,
multi-model classification, and seasonal weight adaptation — all
within the time budget of a single sensor cycle.

**Standalone demo:** `cargo run --bin science_bloom_sentinel`

---

## 6. What It Replaces

The NPU does not replace GPUs. It does not replace CPUs. It does not
replace training. It replaces **the decision layer** — the thin,
fast, cheap classifier that sits between expensive computations and
decides what to do next.

```
┌────────────┐     ┌───────────┐     ┌────────────────┐
│ GPU / CPU  │     │ NPU       │     │ GPU / CPU      │
│ (expensive │────>│ (decision │────>│ (expensive      │
│  compute)  │     │  54 µs)   │     │  next step)    │
└────────────┘     └───────────┘     └────────────────┘
  simulation         gatekeeper        more simulation
  LC-MS scan         triage            library search
  sensor poll        classify          alert / log
```

Without the NPU, the decision layer is either:
- **Missing** — every sample gets the expensive treatment (wasteful)
- **On CPU** — millisecond latency, blocks the pipeline
- **On GPU** — requires batching, introduces latency, wastes compute
  capacity that should be doing simulation

With the NPU:
- **hotSpring:** 63% of thermalization compute was saved because the NPU
  identified trajectories that would have been rejected anyway
- **wetSpring:** 95.8% of LC-MS library searches were skipped because the
  NPU identified peaks that didn't need matching (14x throughput)
- **airSpring:** NPU energy is 0.0009% of the active power cycle vs a
  Raspberry Pi — the decision layer is essentially free

The NPU is the gatekeeper. It makes expensive things cheaper by deciding
which expensive things are worth doing.

**Standalone demo:** `cargo run --bin science_spectral_triage`

---

## 7. The Evidence

Every claim in this document is grounded in measured hardware results
from at least one spring deployment:

| Claim | Measurement | Spring | Experiment |
|-------|-------------|--------|------------|
| 54 µs inference | metalForge Exp 001 | rustChip | `bench_latency` |
| 18,500 Hz throughput | metalForge Exp 001 | rustChip | `bench_latency` |
| 1.4 µJ/inference | metalForge Exp 001 | rustChip | `bench_clock_modes` |
| 37 MB/s DMA | metalForge Exp 001 | rustChip | `bench_dma` |
| 5,978 live calls | Exp 022 (24h campaign) | hotSpring | `npu_steering` |
| 63% thermalization savings | Exp 022 | hotSpring | `npu_steering` |
| 80.4% rejection accuracy | Exp 022 | hotSpring | `npu_steering` |
| 23 µs bloom sentinel | HAB validation | wetSpring | `validate_npu_bloom` |
| 45k spectra/s triage | Spectral pipeline | wetSpring | `validate_npu_spectra` |
| 48.7 µs crop classifier | IoT validation | airSpring | `validate_npu_eco` |
| 20.5 kHz streaming | IoT benchmark | airSpring | `validate_npu_eco` |

Software (`[SW]`) validation runs in parallel on the `SoftwareBackend`
(CPU f32 VirtualNPU). Every claim has a hardware-primary, software-
validated dual path. They are parallel and complementary — never
conflated.

---

## 8. The Invitation

This is a tool. The AKD1000 has 80 neural processors, 10 MB of SRAM,
and a PCIe Gen2 x1 link. It costs less than a GPU, consumes less than
a watt, and makes decisions in microseconds.

The interesting question is not "how fast is it?" — we measured that.
The interesting question is: **what would you do with a microsecond
decision?**

Every field that has an expensive computation gated by a cheap
classification — lattice QCD, mass spectrometry, genomics, agriculture,
environmental monitoring, process control — can use this. The same chip.
The same 80 NPs. The same Rust driver.

The science is in the springs. The silicon is in rustChip. The future
is in [NPU_FRONTIERS.md](NPU_FRONTIERS.md).

**All code, all models, all explorations: AGPL-3.0-or-later (scyBorg triple).**
**Run it yourself:** `cargo run --bin science_lattice_esn`

---

## References

- [TANH_CONSTRAINT.md](TANH_CONSTRAINT.md) — Bounded ReLU vs tanh analysis
- [VFIO_VS_KMOD.md](VFIO_VS_KMOD.md) — Why VFIO is the primary path
- [RUST_AT_SILICON.md](RUST_AT_SILICON.md) — The Rust-native silicon journey
- [GPU_NPU_PCIE.md](GPU_NPU_PCIE.md) — GPU-NPU PCIe data paths
- [NPU_ON_GPU_DIE.md](NPU_ON_GPU_DIE.md) — NPU integrated on GPU die
- [SPRINGS_ON_SILICON.md](SPRINGS_ON_SILICON.md) — Cross-domain NPU patterns
- [NPU_FRONTIERS.md](NPU_FRONTIERS.md) — The creative frontier
- [hotSpring](https://github.com/syntheticChemistry/hotSpring) — Lattice QCD
- [wetSpring](https://github.com/syntheticChemistry/wetSpring) — Sentinel microbe
- [airSpring](https://github.com/syntheticChemistry/airSpring) — Agricultural IoT
- [sporePrint / primals.eco](https://primals.eco) — Public verification
