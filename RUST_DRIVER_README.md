# rustChip

Pure Rust software stack for BrainChip Akida neuromorphic processors (AKD1000, AKD1500).

Forked from [Brainchip-Inc/akida_dw_edma](https://github.com/Brainchip-Inc/akida_dw_edma).
C kernel module → deprecated (see [DEPRECATED.md](DEPRECATED.md)).
All active development is in the crates and directories below.

No Python. No C++ SDK. No MetaTF. No kernel module required.

---

## What this is

A fruiting body from the [ecoPrimals](https://github.com/ecoPrimals) project —
self-contained, carries everything it needs to replicate, designed to be handed
to the BrainChip engineering team as a complete, standalone working system.

It emerged from `toadStool` and `hotSpring`, the shared compute library and physics
simulation suites behind five scientific validation runs (lattice QCD, microbial
ecology, atmospheric physics, neural architectures, uncertainty quantification).
The AKD1000 was used in production physics simulation — 5,978 live hardware calls,
24 hours, lattice SU(3). This is the distillation of what we learned.

---

## Repository structure

```
rustChip/
│
├── crates/                     Rust source — the primary deliverable
│   ├── akida-chip/             silicon model: register map, NP mesh, BAR layout (zero deps)
│   ├── akida-driver/           full driver: VFIO primary, kernel fallback, DMA, inference
│   │   └── src/hybrid.rs       HybridEsn: substrate-agnostic ESN executor (tanh + hardware)
│   ├── akida-models/           FlatBuffer model parser + program_external() injection
│   ├── akida-bench/            benchmark suite: 10 BEYOND_SDK discoveries + experiments
│   └── akida-cli/              `akida` command-line tool
│
├── specs/                      Technical specification — read before coding
│   ├── AI_CONTEXT.md           entry point for AI coding assistants and new devs
│   ├── SILICON_SPEC.md         AKD1000/AKD1500 silicon capabilities, confirmed measurements
│   ├── DRIVER_SPEC.md          driver architecture, backend selection, safety rules
│   ├── PHASE_ROADMAP.md        Phase A–E sovereign driver progression
│   └── INTEGRATION_GUIDE.md   how to integrate with hotSpring / toadStool
│
├── baseCamp/                   Model zoo, novel systems, extended capabilities
│   ├── README.md               landscape: which models, which zoos, which conversions
│   ├── models/                 individual model docs (physics, edge, custom)
│   ├── systems/                novel multi-system architectures
│   │   ├── README.md           7-system NP packing table + answers to "how many?"
│   │   ├── multi_tenancy.md    7 programs at distinct NP addresses simultaneously
│   │   ├── online_evolution.md 136 gen/sec live weight adaptation via set_variable()
│   │   ├── npu_conductor.md    11-head multi-physics fan-out from one program
│   │   ├── hybrid_executor.md  software NPU on hardware NPU — HybridEsn architecture
│   │   ├── hw_sw_comparison.md capability matrix: AKD1000 vs SoftwareBackend
│   │   ├── chaotic_attractor.md Lorenz/Rössler/MSLP tracking on-chip
│   │   ├── temporal_puf.md     hardware fingerprinting via int4 quantization noise
│   │   ├── adaptive_sentinel.md autonomous domain-shift detection + self-recovery
│   │   ├── neuromorphic_pde.md Poisson/Heat equation solving via FC chains
│   │   └── physics_surrogate.md 4-domain GPU+NPU co-located physics ensemble
│   ├── models/edge/beyond_sdk/ extended capabilities beyond BrainChip's SDK claims
│   ├── conversion/             how to get arbitrary models into rustChip format
│   └── zoos/                   landscape survey: MetaTF, NeuroBench, SNNTorch, Norse
│
├── metalForge/                 Hardware experimentation — live measurement protocols
│   ├── README.md               experiment philosophy and status tracker
│   ├── experiments/
│   │   ├── 001_BASELINE_CHARACTERIZATION.md  ✅ 10 BEYOND_SDK discoveries
│   │   ├── 002_MULTI_TENANCY.md              Phase 1 ✅ | Phase 2 (hw co-loading)
│   │   ├── 003_BEYOND_CLAIMED.md             extended SDK capability validation
│   │   └── 004_HYBRID_TANH.md               Phase 1 ✅ | Phase 2 (FlatBuffer path)
│   └── npu/akida/              measurement logs, register probes, hardware profiles
│
├── whitePaper/                 Analysis and outreach
│   ├── README.md               index
│   ├── explorations/           deep-dive technical writeups
│   │   ├── TANH_CONSTRAINT.md  the bounded ReLU finding — impact on hotSpring
│   │   ├── VFIO_VS_KMOD.md     why VFIO beats the C kernel module
│   │   ├── GPU_NPU_PCIE.md     P2P DMA: GPU → NPU without CPU copy
│   │   └── RUST_AT_SILICON.md  long-term pure-Rust substrate vision
│   └── outreach/akida/         material for BrainChip engineering team
│       ├── TECHNICAL_BRIEF.md  10 discoveries + production use + novel systems
│       ├── BENCHMARK_DATASHEET.md  full measurement dataset
│       └── README.md           outreach index
│
├── docs/                       Stable docs (also accessible from whitePaper/outreach/)
├── BEYOND_SDK.md               the most important document — read first
├── CHANGELOG.md                change history
└── DEPRECATED.md               migration guide from C kernel module
```

---

## Quick start

```bash
cd rustChip/
cargo build --release

# List devices
cargo run --bin akida -- enumerate

# Run all hardware experiments (Phase 1 — software simulation, no hardware needed)
cargo run --bin run_experiments

# Run full benchmark suite (hardware required, validates BEYOND_SDK discoveries)
cargo run --bin validate_all -- --sw  # software mode (always available)
cargo run --bin validate_all          # hardware mode (/dev/akida0)

# Individual benchmarks
cargo run --bin bench_latency        # 54 µs / 18,500 Hz
cargo run --bin bench_batch          # batch=8 sweet spot
cargo run --bin bench_exp002_tenancy # multi-tenancy: 7-system NP packing
cargo run --bin bench_exp004_hybrid_tanh  # hybrid tanh: Approach B validation
```

---

## Backend selection

```text
Primary — VFIO (no kernel module):
  cargo run --bin akida -- bind-vfio 0000:a1:00.0   # once, requires root
  cargo run --bin akida -- enumerate                 # no root needed after

Fallback — C kernel module (if installed):
  sudo insmod akida-pcie.ko
  cargo run --bin akida -- enumerate                 # opens /dev/akida*
```

VFIO provides full DMA, IOMMU isolation, works on any kernel version.

---

## Measured results (AKD1000, PCIe x1 Gen2, Feb 2026)

| Metric | Measured |
|--------|----------|
| DMA throughput, sustained | 37 MB/s |
| Single inference | 54 µs / 18,500 Hz |
| Batch=8 inference | 390 µs/sample / 20,700 /s |
| Energy per inference | 1.4 µJ |
| Online weight swap (`set_variable()`) | 86 µs |
| Production calls (Exp 022, 24 h lattice QCD) | 5,978 |
| Multi-system NP packing (7 systems) | 814 / 1,000 NPs |
| Temporal PUF entropy | 6.34 bits |

---

## The 10 hardware discoveries

Full details in [`BEYOND_SDK.md`](BEYOND_SDK.md).

| # | SDK claim | Actual hardware |
|---|-----------|-----------------|
| 1 | InputConv: 1 or 3 channels only | Any channel count (1–64 tested) |
| 2 | FC layers run independently | All FC layers merge via SkipDMA (single HW pass) |
| 3 | Batch=1 only | Batch=8 amortises PCIe: 948→390 µs/sample (2.4×) |
| 4 | One clock mode | 3 modes: Performance / Economy / LowPower |
| 5 | Max FC width ~hundreds | Tested to 8192+ neurons (SRAM-limited only) |
| 6 | Weight updates require reprogram | `set_variable()` updates live (~86 µs optimal) |
| 7 | "30 mW" chip power | Board floor 900 mW; chip compute below noise floor |
| 8 | 8 MB SRAM limit | BAR1 exposes 16 GB address space |
| 9 | Program binary is opaque | FlatBuffer: `program_info` + `program_data`; weights via DMA |
| 10 | Simple inference engine | C++ engine: SkipDMA, 51-bit threshold SRAM, `program_external()` |

---

## Novel capabilities (beyond SDK claims)

Full details in [`baseCamp/systems/README.md`](baseCamp/systems/README.md).

**Answer to "how many systems can one chip handle?"**: 7 simultaneously.

| Capability | What it means |
|-----------|--------------|
| [Multi-tenancy](baseCamp/systems/multi_tenancy.md) | 7 independent programs at distinct NP offsets — 814/1,000 NPs used |
| [Online evolution](baseCamp/systems/online_evolution.md) | 136 gen/sec live weight adaptation via `set_variable()` |
| [NPU conductor](baseCamp/systems/npu_conductor.md) | 11 physics outputs from one reservoir forward pass (SkipDMA) |
| [Hybrid executor](baseCamp/systems/hybrid_executor.md) | Hardware matrix multiply + host tanh = full tanh accuracy at hardware speed |
| [Temporal PUF](baseCamp/systems/temporal_puf.md) | Device fingerprinting via int4 quantization noise (6.34 bits entropy) |
| [Adaptive sentinel](baseCamp/systems/adaptive_sentinel.md) | Autonomous domain-shift detection + self-recovery in 6 seconds |

---

## Key finding: the Tanh Constraint

The AKD1000 uses bounded ReLU as its activation function. This silently constrains
Echo State Networks — random reservoir initialization fails entirely under bounded
ReLU, requiring MetaTF re-optimization. This is undocumented.

**The fix**: `HybridEsn` splits the computation: hardware does the matrix multiply
(int4, 54 µs), host applies tanh to the result (< 1 µs). Full tanh accuracy at
hardware speed. No MetaTF required. No retraining.

```rust
use akida_driver::{HybridEsn, EsnSubstrate};

// hotSpring's existing tanh-trained weights — drop-in
let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3)?;
let prediction = esn.step(&features)?;  // 18,500 Hz, 1.4 µJ
```

Full analysis: [`whitePaper/explorations/TANH_CONSTRAINT.md`](whitePaper/explorations/TANH_CONSTRAINT.md)

---

## Driver roadmap

```
Phase A: Python SDK → Rust FFI wrapper          ✅ done (external)
Phase B: C++ Engine → Rust FFI to libakida.so   ✅ done (external)
Phase C: Direct ioctl/mmap on /dev/akida0        ✅ done (Feb 26, 2026)
Phase D: Pure Rust VFIO driver (this repo)       ✅ active — primary path
Phase E: Rust akida_pcie kernel module           🔲 queued
```

---

## AKD1500 compatibility

All BEYOND_SDK findings transfer directly to AKD1500 (same Akida 1.0 IP).
One constant changes in `akida-chip/src/pcie.rs`: `AKD1500 = 0xA500`.

---

## Scientific context

rustChip emerged from using the AKD1000 as a neuromorphic coprocessor in lattice
QCD simulations. The chip ran Echo State Network inference to steer HMC sampling —
5,978 live calls over 24 hours, achieving 63% thermalization savings and 80.4%
rejection prediction accuracy on a 32⁴ SU(3) lattice.

That work lives at [syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring).
The full technical writeup is in [`whitePaper/outreach/akida/TECHNICAL_BRIEF.md`](whitePaper/outreach/akida/TECHNICAL_BRIEF.md).

---

## For BrainChip engineers

Start here:
1. [`BEYOND_SDK.md`](BEYOND_SDK.md) — the 10 discoveries
2. [`whitePaper/outreach/akida/TECHNICAL_BRIEF.md`](whitePaper/outreach/akida/TECHNICAL_BRIEF.md) — what the hardware actually does
3. [`baseCamp/systems/README.md`](baseCamp/systems/README.md) — what more it can do
4. [`whitePaper/explorations/TANH_CONSTRAINT.md`](whitePaper/explorations/TANH_CONSTRAINT.md) — the one thing to fix in hardware

---

## License

AGPL-3.0-or-later.
The original C kernel module files at the repository root are GPL-2.0 (BrainChip Inc.).
