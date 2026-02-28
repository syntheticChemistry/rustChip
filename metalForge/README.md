# metalForge — Silicon Characterization & Direct-Wire Engineering

**Parent:** rustChip
**Purpose:** Concrete hardware characterization, register probing, cross-substrate
engineering, and experiment logs. Where `whitePaper/` is the write-up,
metalForge is the metal.

---

## Philosophy

Every claim in this repository originated in hardware. metalForge is where
hardware meets code: register read values, DMA timing measurements, BAR layout
probes, clock mode sweeps. Not benchmarks in the marketing sense — probes in
the scientific sense.

The GPU work in ecoPrimals proved this approach: DF64 arithmetic achieving 9.9×
native f64 performance on consumer cards started as a direct hardware measurement,
not a theoretical analysis. The 10 BEYOND_SDK discoveries followed the same method:
test the SDK claim against silicon, document the delta, build the Rust adapter.

---

## Hardware Inventory

| Substrate | Device | PCIe Slot | Key Spec | Status |
|-----------|--------|-----------|----------|--------|
| **NPU** | BrainChip AKD1000 | `08:00.0` | 78 NPs, 8 MB SRAM, ~30 mW chip | VFIO bound, production |
| **NPU** (alt node) | BrainChip AKD1000 | `i9-12900K node` | ESN classifiers, online evolution | Phase D validated |
| **GPU (primary)** | NVIDIA RTX 3090 | `01:00.0` | 24 GB, DF64 3.24 TFLOPS | Production QCD HMC |
| **GPU (secondary)** | NVIDIA Titan V ×2 | `05:00.0` | 12 GB, GV100, native f64 1:2 | Oracle validation |
| **CPU** | AMD Threadripper 3970X | — | 32C/64T, 256 GB DDR4 | Orchestration |

---

## Directory Structure

```
metalForge/
├── README.md                         ← this file
├── npu/
│   └── akida/
│       ├── README.md                 ← NPU characterization overview
│       ├── HARDWARE_PROFILE.md       ← NP types, capabilities, SRAM layout
│       ├── REGISTER_PROBE_LOG.md     ← raw BAR0 read log (confirmed values)
│       ├── benchmarks/
│       │   ├── README.md             ← how to run, what each bench measures
│       │   └── exp001_baseline.md    ← baseline characterization results
│       └── scripts/                  ← probe scripts (see akida-bench crate)
└── experiments/
    └── 001_BASELINE_CHARACTERIZATION.md
```

---

## Relationship to Other Components

| Component | Role | metalForge interaction |
|-----------|------|----------------------|
| `crates/akida-bench/` | Rust benchmark binaries | metalForge documents what each binary measures and reproduces |
| `docs/` | Raw measurement documents | metalForge is the experimental layer; docs/ is the distilled findings |
| `whitePaper/` | Write-ups and analysis | metalForge provides raw data; whitePaper/ provides interpretation |
| `specs/SILICON_SPEC.md` | Register map | metalForge Register Probe Log is the raw source for confirmed entries |

---

## Running the Forge

```bash
# Enumerate hardware (no root if VFIO bound)
cargo run --bin akida -- enumerate
cargo run --bin akida -- info 0

# Run baseline characterization (Experiment 001)
cargo run --bin enumerate
cargo run --bin bench_dma
cargo run --bin bench_latency

# Full BEYOND_SDK reproduction suite
cargo run --bin bench_channels
cargo run --bin bench_fc_depth
cargo run --bin bench_batch
cargo run --bin bench_clock_modes
cargo run --bin bench_fc_width
cargo run --bin bench_weight_mut
cargo run --bin bench_bar

# Compare against reference values in experiments/001_BASELINE_CHARACTERIZATION.md
```

---

## Experiment Log

| Experiment | Date | Description | Status |
|------------|------|-------------|--------|
| [001](experiments/001_BASELINE_CHARACTERIZATION.md) | Feb 27, 2026 | Baseline characterization: DMA, latency, channels, batch, clock modes | ✅ |
