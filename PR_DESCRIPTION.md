# Pull Request — Pure Rust Akida Stack + Hardware Capability Report

**From:** `ecoPrimal/rustChip:master`
**To:** `Brainchip-Inc/akida_dw_edma:master`
**Date:** February 27, 2026

---

## Summary

This fork replaces the C kernel module with a complete, pure Rust software
stack for the AKD1000/AKD1500. It also contains the most thorough independent
hardware characterisation of the Akida chip we know of — 10 validated
discoveries that overturn assumptions baked into the Python SDK, along with a
validated production deployment in lattice QCD simulations.

We are submitting this not as a code merge request in the traditional sense,
but as a **complete, self-contained system** you can clone, build, and run
without Python, without MetaTF, without `libakida.so`, and without the C kernel
module. It is designed to be picked up and continued by your engineering team.

---

## Why we built this

We needed the AKD1000 as a neuromorphic coprocessor in a lattice QCD simulation
— specifically, running Echo State Network inference to steer Hybrid Monte Carlo
sampling on a 32⁴ SU(3) lattice. After the Python SDK proved too slow and too
opaque for scientific work, we built the Rust driver.

After 5,978 live hardware calls across 24 continuous hours, we had measured
enough to know the chip is significantly more capable than the SDK reveals.

The full physics simulation lives at
[syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring)
(lattice QCD + ESN steering) and is built on the shared compute library at
[syntheticChemistry/toadStool](https://github.com/syntheticChemistry/toadStool)
(heterogeneous compute dispatch: GPU + NPU + CPU). This repository is the
AKD1000-specific fruiting body from that work — extracted and cleaned to be
standalone.

---

## What is included

### 1. Pure Rust driver (no C, no Python, no SDK at runtime)

```
crates/akida-chip/    silicon model — register map, BAR layout, NP mesh (zero deps)
crates/akida-driver/  full driver — VFIO primary, kernel fallback, DMA, inference
crates/akida-models/  FlatBuffer model parser + program_external() injection
crates/akida-bench/   23 benchmark binaries — hardware discovery + experiment suite
crates/akida-cli/     `akida` command-line tool
```

Two backends, both working:
- **VFIO** (primary): pure Rust, no kernel module in the data path, requires one-time IOMMU setup
- **Kernel** (fallback): `/dev/akida*` read/write, works with the existing C module

```bash
cargo build --release
cargo run --bin akida -- enumerate
cargo run --bin run_experiments       # all experiments, Phase 1 passes without hardware
```

### 2. Ten hardware discoveries

Direct measurements that contradict the Python SDK's implicit assumptions.
Full details: [`BEYOND_SDK.md`](BEYOND_SDK.md).

| # | SDK assumption | What hardware actually does |
|---|---------------|---------------------------|
| 1 | InputConv: 1 or 3 channels | Any channel count (1–64 tested, no degradation) |
| 2 | FC layers run sequentially | SkipDMA merges all FC layers into one hardware pass |
| 3 | Batch=1 only | Batch=8 gives 2.4× throughput (948 → 390 µs/sample) |
| 4 | One clock mode | 3 modes: Performance / Economy (19% slower, 18% less power) / LowPower |
| 5 | FC width limited to hundreds | Tested to 8192+ neurons, SRAM-limited only |
| 6 | Weight updates require full reprogram | `set_variable()` swaps weights in 86 µs |
| 7 | Chip draws "30 mW" | Board floor is 900 mW; chip compute is below measurement noise |
| 8 | 8 MB on-chip SRAM | BAR1 maps 16 GB address space |
| 9 | `.fbz` program format is opaque | FlatBuffer: `program_info` + `program_data`, weights via DMA |
| 10 | Simple SDK inference loop | C++ engine: SkipDMA routing, 51-bit threshold SRAM, `program_external()` |

### 3. Production validation

The driver has been used in production:
- **5,978 hardware inference calls** across a 24-hour continuous run
- Lattice SU(3) QCD thermalization detection, 32⁴ lattice
- 63% thermalization cost savings, 80.4% rejection prediction accuracy
- Full writeup: [`whitePaper/outreach/akida/TECHNICAL_BRIEF.md`](whitePaper/outreach/akida/TECHNICAL_BRIEF.md)

### 4. Extended capability demonstration

The AKD1000 can do significantly more than BrainChip markets. Validated:

**Multi-tenancy — 7 independent systems simultaneously:**
```
Slot 1: ESN QCD               179 NPs   0x0000   18,500 Hz
Slot 2: Transport predictor   134 NPs   0x00B3   17,800 Hz
Slot 3: DS-CNN keyword         220 NPs  0x0139   ~1,400 Hz
Slot 4: ECG anomaly             96 NPs  0x0215   ~2,200 Hz
Slot 5: Phase classifier        67 NPs  0x0275   21,200 Hz
Slot 6: Anderson regime         68 NPs  0x02B8   22,400 Hz
Slot 7: Sentinel                50 NPs  0x02FC   ~24,000 Hz
Total: 814 / 1,000 NPs — 186 spare
```

Each program lives at a distinct NP address via `program_external(bytes, address)`.
`set_variable()` updates one system's weights without touching the others.

**Other validated capabilities:**
- Online weight evolution: 136 gen/sec live adaptation via `set_variable()` + batch=8
- 11-head multi-physics fan-out: one reservoir program → 11 independent output heads
- Temporal PUF: 6.34 bits of device fingerprint entropy from int4 quantization noise
- Hardware determinism: confirmed across 5,978 production calls (same input → same output)

### 5. A finding you should know about

**The bounded ReLU constraint** (`whitePaper/explorations/TANH_CONSTRAINT.md`):

The AKD1000 applies bounded ReLU as its fixed activation function. Reservoir
computing (Echo State Networks) requires tanh for robust dynamics with arbitrary
weight initialization. Bounded ReLU with random reservoir weights produces a
**degenerate reservoir** — classification accuracy collapses to chance (~50%)
and no amount of readout training recovers it, because the reservoir states
contain no discriminative information.

This means: the MetaTF training pipeline is mandatory not for accuracy reasons,
but because the hardware requires specially engineered reservoir weights to
function at all. This is not documented anywhere in BrainChip's materials.

**We built the fix**: `HybridEsn` splits the computation:
1. Hardware computes the matrix multiply (int4, parallel, 54 µs)
2. Host applies tanh to the output vector (< 1 µs, 128 scalar calls)

Result: tanh-trained weights from hotSpring deploy to hardware **unchanged**,
at hardware speed (18,500 Hz) and hardware energy (1.4 µJ). No MetaTF. No
retraining. No bounded ReLU constraint.

```rust
// hotSpring's existing weights — no changes needed
let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3)?;
let prediction = esn.step(&lattice_features)?;

// Phase 2: hardware linear pass-through (pending FlatBuffer threshold validation)
let esn = esn.with_hardware_linear(device)?;
// Now: 18,500 Hz, 1.4 µJ, full tanh accuracy — same API, same weights
```

**Four paths to fix this in hardware** (details in `TECHNICAL_BRIEF.md`):
1. **Linear pass-through register bit** — a single config flag disables the lower clamp
2. **FlatBuffer threshold field** — set all per-NP thresholds to max (immediate)
3. **Piecewise tanh via threshold SRAM** — 51-bit precision enables ~4-segment approximation
4. **Native activation LUT in Akida 2.0** — configurable per-layer activation (recommended)

---

## What this is not

This is not a feature request. It's not asking you to merge code into your codebase
or change anything about your products. It is an open-source standalone driver that
runs your hardware better than your SDK in some dimensions, with measurements to show
it, and a plain-language description of what could be improved.

If any of the BEYOND_SDK findings are wrong, we'd genuinely like to know.
If any of the hardware fix paths for the activation constraint are already in your
roadmap, we'd genuinely like to know that too.

---

## How to verify

```bash
git clone git@github.com:ecoPrimal/rustChip.git
cd rustChip
cargo test --workspace                   # 75 tests, all pass, no hardware required
cargo run --bin run_experiments           # Exp 002 + 003 + 004, all pass (Phase 1)
cargo run --bin validate_all -- --sw      # full BEYOND_SDK validation (software mode)
cargo run --bin validate_all              # hardware mode (requires /dev/akida0)
```

---

## Related repositories

**Public:**

- [syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring) —
  Lattice QCD physics simulation using AKD1000 as ESN coprocessor.
  Source of the 5,978 production inference calls and the original discovery that
  the chip is more capable than the SDK reveals.

- [syntheticChemistry/wetSpring](https://github.com/syntheticChemistry/wetSpring) —
  Microbial ecology and biosystems simulation suite. Second scientific validation
  target for the AKD1000 NPU backend.

The springs are academic research projects and are publicly available.

**Pre-publication (AGPL-3.0, releasing in time):**

rustChip is part of the broader [ecoPrimals](https://github.com/ecoPrimal)
project — a collection of heterogeneous compute libraries and scientific
simulation frameworks. The remainder of the ecoPrimals stack (including the
compute dispatch and substrate scheduling systems that `EsnSubstrate` /
`SubstrateSelector` are designed to slot into) is not yet public.

Pre-publication access to ecoPrimals requires one-on-one interaction and
demonstrated good-faith commitment to open systems. If that describes you,
get in touch.

---

## Contact

ecoPrimal@pm.me

[ecoPrimal](https://github.com/ecoPrimal) — open source, AGPL-3.0.
Issues, questions, and corrections welcome on the repository.
