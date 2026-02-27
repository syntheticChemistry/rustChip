# Changelog

## [Unreleased] — Phase 1 experiment validation + novel systems (Feb 27, 2026)

### Key finding
- **Tanh constraint discovered**: AKD1000 bounded ReLU breaks ESN reservoir dynamics with
  random initialization. Documented in `whitePaper/explorations/TANH_CONSTRAINT.md`.
  Fix: `HybridEsn` — hardware matrix multiply + host tanh recovery. No MetaTF required.
- **NP address correction**: 7-system packing table had off-by-4–44 hex address errors
  (cumulative rounding). Corrected to exact cumulative sums in all docs and bench binaries.

### Added — crates

**`crates/akida-driver`**
- `src/hybrid.rs` — `HybridEsn`, `EsnSubstrate` trait, `EsnWeights`, `SubstrateSelector`,
  `SubstrateMode`, `SubstrateInfo`. Substrate-agnostic ESN executor for hotSpring/toadStool.
  - `SubstrateMode::PureSoftware` — CPU f32 + tanh (always available, 800 Hz)
  - `SubstrateMode::HardwareLinear` — Approach B: scale trick + host tanh (Phase 1 emulated,
    Phase 2 hardware dispatch pending `metalForge/experiments/004_HYBRID_TANH`)
  - `SubstrateMode::HardwareNative` — bounded ReLU for MetaTF-designed weights
  - `ScaleTrickConfig` — auto-computes ε via 3σ statistical bound
  - `HardwareEsnExecutor::step_linear_emulated()` — working Approach B math (not a stub)
- `examples/vfio_bind.rs` — VFIO bind/unbind helper + status check

**`crates/akida-models`**
- `examples/program_external.rs` — demonstrates `program_external()` NP address semantics

**`crates/akida-bench`**
- `bench_exp002_tenancy.rs` — Exp 002 Phase 1: NP layout, address isolation, reload fidelity,
  weight mutation isolation, 2→4→7 packing progression (all ✅)
- `bench_exp004_hybrid_tanh.rs` — Exp 004 Phase 1: Approach B accuracy, linear region check,
  throughput comparison, ε sweep, determinism (all ✅)
- `run_experiments.rs` — unified runner: Exp 002 + 003 (E3.1+E3.6) + 004, structured pass/fail
  with hardware/software substrate notes. Run: `cargo run --bin run_experiments`
- `bench_multi_tenancy.rs` — multi-tenancy simulation (N systems, round-robin throughput)
- `bench_online_evolution.rs` — 136 gen/sec evolution simulation
- `bench_hw_sw_parity.rs` — HW vs SW capability matrix: throughput, energy, activation

### Added — docs / baseCamp / metalForge

**`specs/`**
- `AI_CONTEXT.md`, `SILICON_SPEC.md`, `DRIVER_SPEC.md`, `PHASE_ROADMAP.md`, `INTEGRATION_GUIDE.md`

**`baseCamp/systems/`**
- `README.md` — 7-system NP packing table (814/1,000 NPs, corrected addresses)
- `multi_tenancy.md`, `online_evolution.md`, `npu_conductor.md` — novel NPU architectures
- `hybrid_executor.md` — HybridEsn design doc
- `hw_sw_comparison.md` — AKD1000 vs SoftwareBackend capability matrix
- `chaotic_attractor.md`, `temporal_puf.md`, `adaptive_sentinel.md` — novel applications
- `neuromorphic_pde.md`, `physics_surrogate.md` — physics computing on NPU

**`baseCamp/models/edge/beyond_sdk/`**
- `akidanet_beyond.md`, `kws_beyond.md`, `ecg_beyond.md`, `dvs_beyond.md`, `detection_beyond.md`
- Extended capabilities for each BrainChip SDK claimed use case

**`metalForge/experiments/`**
- `002_MULTI_TENANCY.md` — updated: Phase 1 results section added, corrected NP addresses
- `003_BEYOND_CLAIMED.md` — extended SDK capability validation protocol
- `004_HYBRID_TANH.md` — updated: Phase 1 results added, Approach B implemented

**`whitePaper/`**
- `explorations/TANH_CONSTRAINT.md` — full analysis of bounded ReLU constraint + fix
- `explorations/VFIO_VS_KMOD.md`, `explorations/GPU_NPU_PCIE.md`, `explorations/RUST_AT_SILICON.md`
- `outreach/akida/TECHNICAL_BRIEF.md` — updated with Discovery 11 (bounded ReLU) + hardware fix paths
- `outreach/akida/BENCHMARK_DATASHEET.md` — updated Section 10: activation constraint + hybrid

### Changed
- `README.md` — full rewrite: complete directory structure, novel systems, quick-start section,
  HybridEsn example, "For BrainChip engineers" section
- `specs/AI_CONTEXT.md` — added baseCamp/metalForge patterns, HybridEsn guidance

---

## [Initial] — divergent evolution from Brainchip-Inc/akida_dw_edma

### Added

**`crates/akida-chip`** — silicon model crate (no dependencies)
- `pcie`: vendor/device IDs for AKD1000 (`0x1E7C:0xBCA1`) and AKD1500 (`0x1E7C:0xA500`)
- `bar`: BAR layout (BAR0 16 MB, BAR1 16 GB NP mesh window, BAR3 32 MB)
- `regs`: BAR0 register map — confirmed addresses from direct probing + C++ symbol analysis
- `mesh`: NP mesh topology (5×8×2, 78 functional, SkipDMA routing model)
- `program`: FlatBuffer `program_info` / `program_data` format (reverse-engineered)

**`crates/akida-driver`** — full pure Rust driver
- VFIO backend: complete DMA (mlock, IOVA mapping, scatter-gather), BAR0 MMIO,
  inference trigger/poll, power measurement via hwmon
- Kernel backend: `/dev/akida*` read/write (fallback when C module present)
- Userspace backend: BAR mmap, development/register probing
- `vfio::bind_to_vfio()` / `unbind_from_vfio()` — replace C `install.sh`
- `vfio::iommu_group()` — IOMMU group discovery from sysfs
- Runtime capability discovery: `MeshTopology`, `ClockMode`, `BatchCapabilities`,
  `WeightMutationSupport`, `PcieConfig` — all from sysfs, nothing hardcoded
- Phase C sovereign driver: direct ioctl/mmap on `/dev/akida0` (Feb 26, 2026)

**`crates/akida-models`** — FlatBuffer model layer
- `.fbz` parser (FlatBuffer + Snappy)
- `program_external()` path: direct program binary injection, bypass SDK compilation
- Model zoo: ESN readout, transport predictor, phase classifier

**`crates/akida-bench`** — BEYOND_SDK reproduction suite
- `bench_channels` — Discovery 1: any input channel count works (1–64)
- `bench_fc_depth` — Discovery 2: FC chains merge via SkipDMA (8 layers ≈ 2 layers)
- `bench_batch` — Discovery 3: batch=8 sweet spot (390 µs/sample, 2.4× speedup)
- `bench_clock_modes` — Discovery 4: Economy = 19% slower, 18% less power
- `bench_fc_width` — Discovery 5: PCIe-dominated below 512 neurons
- `bench_weight_mut` — Discovery 6: weight mutation overhead ~14 ms
- `bench_dma` — Production: 37 MB/s sustained DMA
- `bench_latency` — Production: 54 µs / 18,500 Hz single inference
- `bench_bar` — Discovery 8: BAR layout probe (16 GB BAR1)

**`crates/akida-cli`** — `akida` command-line tool
- `akida enumerate` — list all devices with capabilities
- `akida info <addr>` — detailed single-device info including IOMMU group
- `akida bind-vfio <addr>` — bind to vfio-pci
- `akida unbind-vfio <addr>` — unbind and re-bind to akida driver
- `akida iommu-group <addr>` — show IOMMU group and /dev/vfio path

**Docs**
- `BEYOND_SDK.md` (root + `docs/`) — 10 hardware discoveries, raw measurements
- `docs/HARDWARE.md` — NP mesh architecture, BAR layout, per-NP capabilities
- `docs/TECHNICAL_BRIEF.md` — production use in lattice QCD (Exp 022)
- `docs/BENCHMARK_DATASHEET.md` — complete measurement dataset
- `DEPRECATED.md` — migration guide from C kernel module to Rust VFIO path

### Changed

- `akida-pcie-core.c` and related C files: marked deprecated. Kept at root
  for upstream reference; not part of the Rust build.

### Removed

- Dependency on Python SDK (MetaTF) — replaced by `akida-models` FlatBuffer parser
- Dependency on C++ libakida.so — replaced by direct VFIO + register access
- Dependency on kernel module for operation — VFIO backend requires no C code

---

## Origin — Brainchip-Inc/akida_dw_edma (master)

The original repository contained:
- `akida-pcie-core.c` — Linux PCIe driver wrapping DesignWare eDMA controller
- `install.sh` — kernel module build and load script
- `build_kernel_w_cma.sh` — custom kernel build for CMA support (AKD1500)

These files are preserved at the repository root unchanged.
