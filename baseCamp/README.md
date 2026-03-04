# baseCamp — Model Zoo, Conversions, and Domain Briefings

**Date:** February 27, 2026
**Status:** Rust model infrastructure active; physics models validated;
            vision/audio/edge models: analysis complete, conversion tooling queued;
            systems scaffolded into Rust modules with SRAM access infrastructure

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

## Model Zoo Landscape

| Zoo | Models | Status | Rust path |
|-----|--------|--------|-----------|
| [BrainChip MetaTF](zoos/brainchip_metatf.md) | 30+ models (vision, audio, detection) | Python-only | `akida-models` conversion |
| [NeuroBench](zoos/neurobench.md) | 8 benchmarks (keyword, gesture, ECG, chaos) | Mixed (PyTorch/Akida) | `akida-models` + benchmark adapter |
| [SNNTorch](zoos/snntorch.md) | SNN framework, Leaky Integrate-and-Fire | PyTorch-based | Quantize → .fbz conversion |
| [Norse](zoos/third_party.md) | SNN primitives, event-driven | JAX/PyTorch | Not directly compatible |
| [BindsNET](zoos/third_party.md) | Bio-inspired SNN, STDP | PyTorch | Weights extractable |
| [ecoPrimals Physics](models/physics/) | ESN, phase classifier, transport predictor | **Rust native** | Direct `.fbz` injection |

---

## Models Index

### Physics (ecoPrimals native — validated on real AKD1000)

| Model | Architecture | Task | Status |
|-------|-------------|------|--------|
| [ESN Readout](models/physics/esn_readout.md) | InputConv(50→128) → FC(128→1) | Lattice QCD thermalization steering | ✅ **5,978 live calls** |
| [Phase Classifier](models/physics/phase_classifier.md) | InputConv(3→64) → FC(64→2) | SU(3) confined/deconfined | ✅ 100% test accuracy |
| [Transport Predictor](models/physics/transport_predictor.md) | InputConv(6→128) → FC(128→3) | D*/η*/λ* from observables | ✅ All outputs finite |
| [Anderson Regime Classifier](models/physics/anderson_classifier.md) | InputConv(4→64) → FC(64→3) | Localized/diffusive/critical | ✅ (groundSpring Exp 028) |

### Edge Intelligence (NeuroBench benchmark models)

| Model | Architecture | Task | Status |
|-------|-------------|------|--------|
| [DS-CNN KWS](models/edge/ds_cnn_kws.md) | Depthwise separable CNN | Keyword spotting (35 words) | 📋 Analysis + conversion plan |
| [ECG Anomaly](models/edge/ecg_anomaly.md) | FC + threshold | ECG anomaly detection | 📋 Analysis |
| [DVS Gesture](models/edge/dvs_gesture.md) | Event-based CNN | DVS128 gesture (11 classes) | 📋 Analysis |
| [Chaotic ESN](models/edge/chaotic_esn.md) | Reservoir + readout | MSLP chaotic prediction | 📋 Extends ecoPrimals ESN |
| [AkidaNet 0.5](models/edge/akidanet.md) | MobileNet-style SNN | ImageNet top-1 65% | 📋 Analysis |

### Custom (hand-built via program_external())

| Model | Description | Status |
|-------|-------------|--------|
| [Minimal FC](models/custom/minimal_fc.md) | 50→1 FC, hand-crafted FlatBuffer | ✅ program_external() confirmed |
| [ESN Stub](models/custom/esn_stub.md) | Skeleton for custom reservoir programs | 📋 Template |

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
1. `zoos/brainchip_metatf.md` — the official zoo
2. `zoos/neurobench.md` — hardware-benchmarked models
3. `models/physics/esn_readout.md` — the one we validated in production

**Adding a new model from an existing framework:**
1. `conversion/from_pytorch.md` — the general path
2. `conversion/from_snntorch.md` — if using SNN framework
3. `conversion/from_scratch.md` — if hand-building

**Understanding the ecoPrimals models:**
1. `models/physics/` — all four validated physics models
2. `whitePaper/outreach/akida/TECHNICAL_BRIEF.md` — production results

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
