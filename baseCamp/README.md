# baseCamp — Model Zoo, Conversions, and Domain Briefings

**Date:** April 30, 2026
**Status:** 28-model zoo (21 BrainChip + 4 physics + 2 NeuroBench + 1 hand-built). Pure Rust
            conversion pipeline (import/quantize/serialize/compress). Nature Preserve
            with 7 domain application patterns. CLI model inspection and conversion.
            Systems scaffolded into Rust modules with SRAM access infrastructure.

---

## Scaffolded Systems (Rust Modules)

The following systems are scaffolded as Rust modules with SRAM access infrastructure:

| Module | Path | Key types |
|--------|------|-----------|
| Program builder | [`crates/akida-models/src/builder.rs`](../crates/akida-models/src/builder.rs) | `ProgramBuilder`, `LayerSpec`, `QuantConfig`, `EsnProgramBuilder` |
| Multi-tenancy | [`crates/akida-driver/src/tenancy.rs`](../crates/akida-driver/src/tenancy.rs) | `MultiTenantDevice`, `ProgramSlot`, `load_at_offset()`, `verify_isolation()` |
| Online evolution | [`crates/akida-driver/src/evolution.rs`](../crates/akida-driver/src/evolution.rs) | `NpuEvolver`, `WeightPatch`, `EvolutionConfig`, `FitnessEvaluator` |
| Temporal PUF | [`crates/akida-driver/src/puf.rs`](../crates/akida-driver/src/puf.rs) | `PufSignature`, `PufConfig`, `measure_puf()`, `puf_entropy()`, `puf_hamming_distance()` |
| Adaptive sentinel | [`crates/akida-driver/src/sentinel.rs`](../crates/akida-driver/src/sentinel.rs) | `DriftMonitor`, `DriftAlert`, `DriftConfig`, `AdaptiveRecovery` |

**SRAM infrastructure:** `SramAccessor` provides direct BAR0/BAR1 SRAM read/write. `NpuBackend` trait has `verify_load()`, `mutate_weights()`, `read_sram()`. `Capabilities::from_bar0()` reads NP count, SRAM size, mesh topology from BAR0 registers. `probe_sram` binary provides interactive SRAM diagnostics.

---

## What baseCamp is

In ecoPrimals, baseCamp is where the technology meets the science domain.
For rustChip, the domain is neuromorphic inference — so baseCamp is about **models**:

- Which models exist (the landscape survey in `zoos/`)
- Which models we've rebuilt in Rust or can load via `akida-models` (in `models/`)
- How to get arbitrary models into rustChip format (in `conversion/`)
- The ecoPrimals-specific models — ESN, physics classifiers — that extend the standard zoo

The pattern is consistent with the other springs:
- hotSpring's baseCamp documents physics papers and what was reproduced
- wetSpring's baseCamp documents biology domains and what was validated
- rustChip's baseCamp documents **model architectures** and what runs on silicon

---

## Guides

Start here depending on what you want to do:

| Document | Audience |
|----------|----------|
| [**QUICKSTART.md**](../QUICKSTART.md) | Clone → build → parse in 5 commands, no Python needed |
| [**ZOO_GUIDE.md**](ZOO_GUIDE.md) | Full model zoo (28 models), Rust conversion pipeline, ecosystem integration |
| [**preserve/README.md**](preserve/README.md) | Nature Preserve — 7 domain application patterns bridging zoo to science |
| [**SCIENTIFIC_DEPLOYMENT.md**](SCIENTIFIC_DEPLOYMENT.md) | Deploying NPU for scientific/data workloads — architecture, results, patterns |
| [**RUST_NPU_ECOSYSTEM.md**](RUST_NPU_ECOSYSTEM.md) | What the Rust NPU ecosystem enables and how rustChip fits in |
| [**spring-profiles/README.md**](spring-profiles/README.md) | Every NPU workload from ecoPrimals springs, with reproduction instructions |
| [**GUIDESTONE_CERTIFICATION.md**](GUIDESTONE_CERTIFICATION.md) | guideStone verification status — the 5-property checklist for reproducible NPU compute |

---

## Model Zoo Landscape

| Zoo | Models | Status | Rust path |
|-----|--------|--------|-----------|
| [BrainChip MetaTF](zoos/brainchip_metatf.md) | 21 exported, all parse | **Fully validated** | `akida-models` parse + zoo-status |
| [NeuroBench](zoos/neurobench.md) | 8 benchmarks (keyword, gesture, ECG, chaos) | 1 exported (DS-CNN) | `akida-models` + benchmark adapter |
| [SNNTorch](zoos/snntorch.md) | SNN framework, Leaky Integrate-and-Fire | Conversion path documented | Quantize → .fbz conversion |
| [Norse](zoos/third_party.md) | SNN primitives, event-driven | Architecture analysis | Not directly compatible |
| [BindsNET](zoos/third_party.md) | Bio-inspired SNN, STDP | Architecture analysis | Weights extractable |
| [ecoPrimals Physics](models/physics/) | 4 models, hardware-validated | **Production-deployed** | Direct `.fbz` injection |

---

## Models Index

### Physics (ecoPrimals native — validated on real AKD1000)

| Model | Architecture | Task | Status |
|-------|-------------|------|--------|
| [ESN Readout](models/physics/esn_readout.md) | InputConv(50→128) → FC(128→1) | Lattice QCD thermalization steering | ✅ **5,978 live calls** |
| [Phase Classifier](models/physics/phase_classifier.md) | InputConv(3→64) → FC(64→2) | SU(3) confined/deconfined | ✅ 100% test accuracy |
| [Transport Predictor](models/physics/transport_predictor.md) | InputConv(6→128) → FC(128→3) | D*/η*/λ* from observables | ✅ All outputs finite |
| [Anderson Regime Classifier](models/physics/anderson_classifier.md) | InputConv(4→64) → FC(64→3) | Localized/diffusive/critical | ✅ (groundSpring Exp 028) |

### BrainChip MetaTF Zoo (21 models — all parsed by rustChip)

| Category | Models | Status |
|----------|--------|--------|
| Image classification | AkidaNet (1.0, 18), PlantVillage, VWW, MobileNet, GXNOR MNIST | ✅ Exported + parsed |
| Face analysis | AkidaNet FaceID, VGG UTK Face | ✅ Exported + parsed |
| Object detection | CenterNet VOC, YOLO VOC, YOLO WiderFace | ✅ Exported + parsed |
| Segmentation | Akida UNet Portrait 128 | ✅ Exported + parsed |
| Gesture/video | ConvTiny Gesture, ConvTiny Handy, TENN ST (DVS128, Eye, Jester) | ✅ Exported + parsed |
| Audio/speech | DS-CNN KWS, TENN Recurrent (SC12, UORED) | ✅ Exported + parsed |
| 3D point cloud | PointNet++ ModelNet40 | ✅ Exported + parsed |

Full catalog with I/O shapes and sizes: [ZOO_GUIDE.md](ZOO_GUIDE.md)

### Edge Intelligence (NeuroBench benchmark models)

| Model | Architecture | Task | Status |
|-------|-------------|------|--------|
| [DS-CNN KWS](models/edge/ds_cnn_kws.md) | Depthwise separable CNN | Keyword spotting (35 words) | ✅ Exported + parsed (41 KB) |
| [ECG Anomaly](models/edge/ecg_anomaly.md) | FC + threshold | ECG anomaly detection | 📋 Architecture analysis |
| [DVS Gesture](models/edge/dvs_gesture.md) | Event-based CNN | DVS128 gesture (11 classes) | ✅ Exported as ConvTiny Gesture |
| [Chaotic ESN](models/edge/chaotic_esn.md) | Reservoir + readout | MSLP chaotic prediction | 📋 Extends ecoPrimals ESN |
| [AkidaNet](models/edge/akidanet.md) | MobileNet-style SNN | ImageNet classification | ✅ Exported (5,269 KB) |

### Rust-Native Models (generated by `akida convert`, no Python)

| Model | Architecture | Purpose | Status |
|-------|-------------|---------|--------|
| ESN Multi-Head (3-out) | InputConv(50)→FC(128)→FC(1) | Multi-observable readout | ✅ Converted + parsed |
| ESN 3-Head Transport | InputConv(50)→FC(64)→FC(3) | Transport coefficient prediction | ✅ Converted + parsed |
| Streaming Sensor 12ch | InputConv(1)→FC(256)→FC(128)→FC(12) | 12-channel sensor fusion | ✅ Converted + parsed |
| Adaptive Sentinel | InputConv(64)→FC(128)→FC(1) | Domain-shift drift detection | ✅ Converted + parsed |

### Custom (hand-built via program_external())

| Model | Description | Status |
|-------|-------------|--------|
| [Minimal FC](models/custom/minimal_fc.md) | 50→1 FC, hand-crafted FlatBuffer | ✅ program_external() confirmed |
| [ESN Stub](models/custom/esn_stub.md) | Skeleton for custom reservoir programs | 📋 Template |

---

## Science Demos — Standalone NPU Proof Layer

Five self-contained binaries that reproduce peer-reviewed NPU science claims
without external data or hardware. Each demo is a standalone organism: derivative
of the ecoPrimals springs but fully self-sufficient within rustChip.

| Binary | NPU Pattern | Science Domain | Spring Origin |
|--------|-------------|----------------|---------------|
| `science_lattice_esn` | Hybrid ESN | Lattice QCD thermalization steering | hotSpring |
| `science_bloom_sentinel` | Streaming Sentinel | Harmful algal bloom detection | wetSpring |
| `science_spectral_triage` | Microsecond Gatekeeper | LC-MS spectral peak triage | wetSpring |
| `science_crop_classifier` | Online Adaptation | Seasonal crop stress via (1+1)-ES | airSpring |
| `science_precision_ladder` | Precision Discipline | f64 → f32 → int8 → int4 degradation | cross-domain |

Run any demo: `cargo run --bin science_lattice_esn`

For the narrative context behind these patterns, see
[`whitePaper/explorations/WHY_NPU.md`](../whitePaper/explorations/WHY_NPU.md) and
[`whitePaper/explorations/SPRINGS_ON_SILICON.md`](../whitePaper/explorations/SPRINGS_ON_SILICON.md).

---

## The Model Pipeline

```
External source                    Conversion path             rustChip
───────────────────────────────────────────────────────────────────────

BrainChip MetaTF zoo ──────────────────────────────────────────┐
  (.fbz files, Python SDK)                                      │
                                                                ▼
SNNTorch / Norse / BindsNET ──── quantize ──── compile ── akida-models
  (PyTorch SNN models)           (float→int4)  (.fbz)    (parse + load)

ecoPrimals physics ──────────── direct ──────────────────── akida-models
  (ESN weights, int4 quantized)  (.fbz injection           (program_external)
                                  via program_external)

Hand-crafted programs ───────── FlatBuffer ─────────────── akida-models
  (custom physics architectures)  (build_program_info)     (inject directly)
                                                                │
                                                                ▼
                                                          akida-driver
                                                          (VFIO inference)
```

---

## Reading Order

**Starting from scratch (what is Akida, what can it run?):**
1. [ZOO_GUIDE.md](ZOO_GUIDE.md) — the complete zoo guide
2. `zoos/brainchip_metatf.md` — detailed BrainChip catalog
3. `models/physics/esn_readout.md` — the one we validated in production

**I want to deploy an NPU for science or data:**
1. [SCIENTIFIC_DEPLOYMENT.md](SCIENTIFIC_DEPLOYMENT.md) — architecture, results, patterns
2. `spring-profiles/README.md` — reproducible spring workloads
3. `models/physics/` — all four validated physics models

**What can Rust do that Python can't?:**
1. [RUST_NPU_ECOSYSTEM.md](RUST_NPU_ECOSYSTEM.md) — ecosystem analysis
2. `systems/multi_tenancy.md` — 7 concurrent models (not possible via Python SDK)
3. `systems/online_evolution.md` — 136 gen/sec weight evolution

**Verification and reproducibility:**
1. [GUIDESTONE_CERTIFICATION.md](GUIDESTONE_CERTIFICATION.md) — current status per property
2. `../specs/GUIDESTONE.md` — what guideStone means for NPU compute
3. `SCIENTIFIC_DEPLOYMENT.md` §6 — reproducibility table

**Adding a new model from an existing framework:**
1. `conversion/from_pytorch.md` — the general path
2. `conversion/from_snntorch.md` — if using SNN framework
3. `conversion/from_scratch.md` — if hand-building

---

## What rustChip adds beyond the standard zoo

| Addition | Description |
|----------|-------------|
| **Online weight mutation** | swap 3 classifiers via set_variable() at 86 µs (not in any zoo) |
| **ESN temporal streaming** | reservoir state maintained across calls at 18.5K Hz |
| **Physics-calibrated models** | ESN trained on real SU(3) lattice data, not benchmark datasets |
| **Batch-aware loading** | models designed around batch=8 amortisation sweet spot |
| **program_external() injection** | bypass compilation entirely — load any hand-built program |
| **VFIO-native inference** | no C kernel module; models run on any kernel with VFIO support |
