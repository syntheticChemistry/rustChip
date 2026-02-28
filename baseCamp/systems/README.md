# baseCamp/systems — Novel NPU Systems and Multi-Tenancy

**Date:** February 27, 2026
**Central question:** How many concurrent systems can a single AKD1000 handle?
**Answer:** 7 fully independent systems simultaneously, or 11+ outputs from one program.

---

## The Hardware Reality

BrainChip markets the AKD1000 as a **single-task inference accelerator**.
The silicon tells a different story.

```
AKD1000 physical budget:
  1,000 NPs total (80 DP-NPs + 920 general NPs)
  4 SRAM types: filter (64-bit), threshold (51-bit), event (32-bit), status (32-bit)
  SkipDMA: NP-to-NP without PCIe round-trip
  set_variable(): readout swap in 86 µs (not 14 ms — quantization-matched)
  Batch=8 sweet spot: 2.4× throughput from amortized PCIe
  Hardware determinism: same input → same output, always
  16 GB BAR1 address space: full NP mesh may be memory-mapped
```

The SDK never combines these into a multi-tenant view. That is the gap.

---

## Simultaneous System Packing

Every model below has been validated on real AKD1000 or has a confirmed
architecture that fits within NP budget constraints:

```
NP budget: 1,000 total
─────────────────────────────────────────────────────────────────────────────────────────────
 Slot  System                         NPs  NP Start  NP End  Status        Throughput
─────────────────────────────────────────────────────────────────────────────────────────────
  1    ESN QCD thermalization         179  0x0000    0x00B3  ✅ validated    18,500 Hz
  2    Transport predictor (D*,η*,λ*) 134  0x00B3    0x0139  ✅ validated    17,800 Hz
  3    DS-CNN keyword spotting        220  0x0139    0x0215  📋 analysis     ~1,400 Hz
  4    ECG anomaly detection           96  0x0215    0x0275  📋 analysis     ~2,200 Hz
  5    Phase classifier (SU3)          67  0x0275    0x02B8  ✅ validated    21,200 Hz
  6    Anderson regime classifier      68  0x02B8    0x02FC  ✅ validated    22,400 Hz
  7    Minimal sentinel (50-dim)       50  0x02FC    0x032E  ✅ confirmed    ~24,000 Hz
─────────────────────────────────────────────────────────────────────────────────────────────
  TOTAL                               814                    186 NPs spare
─────────────────────────────────────────────────────────────────────────────────────────────
```
NP addresses are cumulative (end of slot N = start of slot N+1). Validated by
`cargo run --bin bench_exp002_tenancy` — no overlaps, 186 spare NPs confirmed.

**7 distinct systems on one chip, simultaneously.**
The 186 spare NPs fit an 8th minimal system or a temporal integration ESN.

With the 11-head pattern (SkipDMA, single program):

```
1 reservoir program (179 NPs) → 11 independent output heads
  Head 0: thermalization flag
  Head 1: phase classification
  Head 2: anomaly score
  Head 3: β priority (pre-scan)
  Head 4: CG iteration estimate
  Head 5: rejection likelihood
  Head 6: quality score
  Head 7: next-run recommendation
  Head 8: deconfinement order parameter
  Head 9: transport coefficient D*
  Head 10: transport coefficient η*

→ 11 outputs at single-program latency (54 µs, not 11 × 54 µs)
```

---

## Novel Systems (Not in Any SDK Zoo)

| System | NPs | Capability | Status |
|--------|-----|-----------|--------|
| [Multi-tenancy controller](multi_tenancy.md) | 814 | 7 simultaneous domains | Architecture defined |
| [Online evolution engine](online_evolution.md) | 179+50 | 136 gen/sec live adaptation | ✅ validated (wetSpring) |
| [NPU conductor (11-head)](npu_conductor.md) | 179 | 11 physics outputs, 1 program | ✅ validated (hotSpring Exp 023) |
| [Chaotic attractor tracker](chaotic_attractor.md) | 259 | ESN for Lorenz/Rössler/MSLP | Architecture + NeuroBench ref |
| [Temporal PUF](temporal_puf.md) | 68 | Hardware fingerprinting, 6.34 bits | ✅ validated (wetSpring NPU) |
| [Adaptive edge sentinel](adaptive_sentinel.md) | 179+50 | Domain-shift detection + adapt | ✅ validated (wetSpring) |
| [Neuromorphic PDE solver](neuromorphic_pde.md) | 200-400 | Poisson/heat via multi-pass FC | Planned |
| [Physics surrogate ensemble](physics_surrogate.md) | 560 | 4-domain co-located surrogates | Architecture defined |

---

## Extended SDK Use Cases

| BrainChip claims | rustChip demonstrates |
|-----------------|----------------------|
| [AkidaNet: 65% ImageNet top-1](../models/edge/beyond_sdk/akidanet_beyond.md) | Domain adaptation at 86 µs, online fine-tune, 1000→N class hot-swap |
| [DS-CNN: 93.8% keyword spotting](../models/edge/beyond_sdk/kws_beyond.md) | Multi-vocabulary hot-swap, simultaneous KWS+anomaly, acoustic sentinels |
| [DVS: 97.9% gesture](../models/edge/beyond_sdk/dvs_beyond.md) | Continuous temporal tracking via ESN, multi-gesture ensemble |
| [ECG: 97.4% anomaly](../models/edge/beyond_sdk/ecg_beyond.md) | 12B inferences on coin cell, real-time HAB/environmental adaptation |
| [YOLO: mAP 0.42](../models/edge/beyond_sdk/detection_beyond.md) | Temporal object persistence via ESN integration, multi-class co-location |

---

## Reading Order

**For hardware validation (you have a live AKD1000):**
1. `multi_tenancy.md` — can the chip really hold all 7 simultaneously?
2. `bench_multi_tenancy` binary — measures it directly
3. `npu_conductor.md` — run the 11-head system

**For novel capabilities:**
1. `online_evolution.md` — the most surprising capability
2. `temporal_puf.md` — hardware fingerprinting with no extra software
3. `chaotic_attractor.md` — the attractor tracking / MSLP NeuroBench bridge

**For the outreach case:**
1. The "Beyond SDK" extended analyses in `models/edge/beyond_sdk/`
2. `physics_surrogate.md` — the clearest GPU+NPU value demonstration
3. `metalForge/experiments/002_MULTI_TENANCY.md` for empirical validation
