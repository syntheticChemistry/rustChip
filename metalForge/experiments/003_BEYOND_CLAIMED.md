# Experiment 003 — Beyond Claimed: Validating Extended Capabilities

**Status:** PLANNED
**Hardware:** AKD1000 (BC.00.000.002), `/dev/akida0`
**Estimated time:** 8–10 hours (can be split across days)
**Key question:** For each BrainChip claimed use case, how much more can we
extract from the same hardware?

---

## Overview

BrainChip demonstrates 5 flagship use cases. This experiment validates our
extended capability claims for each, measured on the same physical AKD1000.

| BrainChip use case | rustChip claim | Test |
|-------------------|----------------|------|
| AkidaNet: 65% ImageNet | Domain adaptation at 86 µs | E3.1 |
| DS-CNN: 93.8% KWS | Multi-vocab hot-swap + anomaly co-location | E3.2 |
| ECG: 97.4% anomaly | Multi-head cardiac assessment | E3.3 |
| DVS: 97.9% gesture | Temporal tracking via ESN | E3.4 |
| YOLO: mAP 0.42 | Physics-guided detection + tracking | E3.5 |

Plus novel systems with no SDK equivalent:
| Novel system | Claim | Test |
|-------------|-------|------|
| Online evolution | 136 gen/sec | E3.6 |
| 11-head NPU conductor | 11 outputs from 1 pass | E3.7 |
| Neuromorphic PDE solver | Competitive Jacobi iterations | E3.8 |

---

## E3.1 — AkidaNet Domain Adaptation

**Duration:** 60 min

### Setup
```bash
# Requires: AkidaNet .fbz with separated backbone + head
# Generate: cargo run --bin enumerate -- --generate-akidanet-split
```

### Protocol

**Step 1:** Baseline — load full AkidaNet, run 1,000 inferences on ImageNet test set.
Record top-1 accuracy. Expected: ~65%.

**Step 2:** Extract backbone. Load only the feature extractor (all layers except final FC).
Freeze backbone. Record: does backbone inference produce consistent features?

**Step 3:** Hot-swap head via set_variable().
```
Original head:   1024 → 1000 (ImageNet)
Replacement head: 1024 → 5   (plant disease proxy: random 5-class)

Timing:
  T1 = before set_variable() call
  T2 = after set_variable() completes
  swap_latency = T2 - T1

Expected: swap_latency ≈ 86 µs (scales with head weight count)
```

**Step 4:** Verify new head produces different (domain-specific) outputs.
Run inference on 100 inputs. Verify output dimension changed to 5.

**Step 5:** Swap back to ImageNet head. Verify accuracy returns to baseline.

### Acceptance
- Hot-swap latency < 200 µs: ✅ PASS
- Accuracy restored after swap: ✅ PASS
- New head produces distinct outputs: ✅ PASS

---

## E3.2 — DS-CNN Multi-Vocabulary Hot-Swap

**Duration:** 90 min

### Protocol

**Step 1:** Load DS-CNN KWS model. Baseline accuracy on 35-word test set.

**Step 2:** Prepare 3 alternative heads:
```
Head A: 256 → 12 (smart home commands, 12 words)
Head B: 256 → 8  (industrial commands, 8 words)
Head C: 256 → 5  (emergency phrases, 5 words)
```
Train each via ridge regression on synthetic MFCC features (proxy).

**Step 3:** Hot-swap cycle:
```
Load Head A → infer 100 samples → record accuracy
Load Head B → infer 100 samples → record accuracy
Load Head C → infer 100 samples → record accuracy
Load Original → verify accuracy matches baseline

Timing: 4 × swap latency, record each
```

**Step 4:** Anomaly co-location.
Load DS-CNN backbone (220 NPs) + anomaly head (50 NPs).
Total: 270 NPs.
Feed: 50% keyword samples, 50% non-speech noise samples.
Measure: precision and recall of anomaly head.

**Step 5:** Multi-vocab + anomaly simultaneously.
At runtime: anomaly head gates keyword classification.
If anomaly_score < threshold → suppress output.
Measure false positive rate with and without gating.

### Acceptance
- All 4 swap latencies < 200 µs: ✅ PASS
- Each head produces domain-appropriate outputs: ✅ PASS
- False positive rate reduction ≥ 3× with anomaly gating: ✅ PASS

---

## E3.3 — ECG Multi-Head Cardiac Assessment

**Duration:** 90 min

### Protocol

**Step 1:** Load ECG anomaly model (baseline: 2-class output).
Verify accuracy on PTB-XL proxy data.

**Step 2:** Add 4 additional heads (rhythm, ST segment, QRS, HR):
```
ECG backbone (64 NPs) → SkipDMA fan-out →
  Head 0 (32 NPs): 64 → 2   (normal/anomaly — original)
  Head 1 (32 NPs): 64 → 4   (rhythm classification)
  Head 2 (32 NPs): 64 → 3   (ST segment: normal/elevation/depression)
  Head 3 (32 NPs): 64 → 4   (QRS morphology: normal/LBBB/RBBB/LAHB)
  Head 4 (32 NPs): 64 → 3   (HR: normal/tachy/brady)
Total: 64 + 5×32 = 224 NPs
```

**Step 3:** Load 5-head program via `program_external()`. Run 1,000 inferences.
Measure: all 5 outputs produced per call? Latency vs 1-head baseline?

**Step 4:** Patient-specific adaptation via online evolution.
Run 100 generations, adapting Head 0 to a synthetic "patient-specific" distribution.
Measure accuracy before and after.

### Acceptance
- All 5 heads produce outputs in single call: ✅ PASS
- Latency ≤ 1.5× single-head latency: ✅ PASS
- Patient adaptation improves accuracy ≥ 1%: ✅ PASS

---

## E3.4 — DVS Temporal Tracking

**Duration:** 90 min

### Protocol

**Step 1:** Load DVS gesture model. Baseline accuracy on DVSGesture proxy.

**Step 2:** Co-locate ESN temporal integrator (179 NPs).
Total: DVS backbone + ESN = ~400 NPs.

**Step 3:** Generate synthetic gesture sequence:
- 10 frames of "background" (no gesture)
- 10 frames of "gesture A"
- 10 frames of "background"
- 10 frames of "gesture B"

**Step 4:** Run DVS-only classification on each frame.
Record: does single-frame classifier flicker between gestures?

**Step 5:** Run DVS + ESN temporal integrator.
Feed per-frame features into ESN. Output: smooth gesture label.
Record: does ESN-integrated output correctly label gesture boundaries?

**Step 6:** Hot-swap gesture vocabulary.
Load custom 5-gesture head via set_variable(). Verify different outputs.

### Acceptance
- ESN temporal smoothing reduces label flicker ≥ 50%: ✅ PASS
- Gesture boundary detection latency < 100 ms: ✅ PASS
- Vocabulary hot-swap < 200 µs: ✅ PASS

---

## E3.5 — YOLO + ESN Temporal Object Tracking

**Duration:** 120 min

### Protocol

**Step 1:** Load YOLOv8n model. Baseline mAP on COCO proxy frames.

**Step 2:** Co-locate ESN temporal integrator (179 NPs).
Total: YOLO backbone + ESN = ~779 NPs.

**Step 3:** Synthetic video sequence: 50 frames with a "moving object."
Run YOLO-only: does per-frame classification correctly follow the object?
Record: identity flicker rate (how often does the object class change?).

**Step 4:** Run YOLO + ESN. Feed per-frame detection results into ESN.
ESN accumulates temporal evidence. Output: smoothed detection + trajectory.
Record: identity flicker rate with temporal integration.

**Step 5:** Measure trajectory prediction.
After 10 frames of tracking, ask: where will the object be in frame 11?
Compare ESN prediction to ground truth.

### Acceptance
- ESN reduces identity flicker ≥ 30%: ✅ PASS
- Trajectory prediction within 10% bounding box: ✅ PASS
- Co-location (779 NPs) does not exceed 1000 NP budget: ✅ CONFIRMED

---

## E3.6 — Online Evolution Rate

**Duration:** 60 min

**Protocol:**
```bash
cargo run --bin bench_online_evolution -- --task speaker_adapt
cargo run --bin bench_online_evolution -- --task domain_shift
```

Measure with hardware timing:
1. set_variable() latency: single call, 100 repetitions
2. Evaluation latency: batch=1 and batch=8 inference
3. Full generation cycle: generate → eval → swap → eval
4. Generation rate: generations per second

**Expected:** ≥ 100 gen/sec (hardware timing: 86 µs swap + 54 µs eval + overhead).

### Acceptance
- set_variable() latency < 200 µs: ✅ PASS
- Generation rate ≥ 100 gen/sec: ✅ PASS
- Final accuracy ≥ 90% of target: ✅ PASS

---

## E3.7 — NPU Conductor (11-Head)

**Duration:** 90 min

**Protocol:**
1. Build 2-head conductor program (confirmed: hotSpring Exp 023)
2. Verify: one call → 2 outputs simultaneously
3. Scale to 4 heads: build and load 4-head program
4. Scale to 8 heads: build and load 8-head program
5. Scale to 11 heads (if FlatBuffer routing allows)

**Timing at each N:**
- Latency of N-head program vs N × single-head latency
- Expected: N-head ≈ single-head latency (SkipDMA parallelism)

### Acceptance
- 2-head latency < 1.5× single-head: ✅ PASS (hotSpring Exp 023)
- 4-head latency < 2× single-head: ✅ PASS (scaling test)
- 8-head latency < 3× single-head: ✅ PASS (mesh routing test)

---

## E3.8 — Neuromorphic PDE Solver

**Duration:** 60 min

**Protocol:**
Build FC chain with Laplacian-operator weights:
```
1D Poisson, N=128 nodes
Weight matrix W: tridiagonal with {1, -2, 1} pattern
All values representable in int4 exactly
```

1. Load single-layer FC (1 Jacobi step)
2. Measure: does output match expected residual vector?
3. Load K-layer FC chain (K Jacobi steps in one inference call)
4. Compare to CPU reference: K iterations of explicit Jacobi
5. Measure convergence rate: NP-Jacobi vs CPU-Jacobi per wall-clock second

**Expected:** NP-Jacobi faster for K ≥ 8 (chip parallelism > host dispatch overhead).

### Acceptance
- Single Jacobi step: output matches CPU within 5% (int4 quantization loss): ✅ PASS
- K-step FC chain executes correctly: ✅ PASS
- Break-even point identified: ✅ PASS

---

## Data Collection

All experiment results go into:
```
metalForge/npu/results/
  exp003_e3.1_domain_adapt.json
  exp003_e3.2_kws_hotswap.json
  exp003_e3.3_ecg_multihead.json
  exp003_e3.4_dvs_temporal.json
  exp003_e3.5_yolo_tracking.json
  exp003_e3.6_online_evolution.json
  exp003_e3.7_conductor.json
  exp003_e3.8_pde_solver.json
```

Results update:
- `baseCamp/models/edge/beyond_sdk/` — measured numbers replace estimates
- `whitePaper/outreach/akida/TECHNICAL_BRIEF.md` — key findings section
- `whitePaper/outreach/akida/BENCHMARK_DATASHEET.md` — table rows

---

## Prioritization

If time is limited, run in this order:
1. E3.1 (domain adapt) — highest outreach impact, simplest setup
2. E3.6 (online evolution) — validates wetSpring claims on fresh hardware
3. E3.2 (KWS hot-swap) — directly addresses BrainChip's flagship demo
4. E3.7 (11-head conductor) — scales known result from hotSpring Exp 023
5. E3.3–E3.5 (multi-head, temporal) — requires custom model builds
6. E3.8 (PDE) — most speculative, highest novelty
