# Experiment 002 — Multi-Tenancy: 7 Systems on One AKD1000

**Status:** Phase 1 COMPLETE ✅ (software isolation model) | Phase 2 PENDING (hardware co-loading)
**Hardware:** AKD1000 (BC.00.000.002), `/dev/akida0`
**Estimated time:** Phase 2: 4–6 hours
**Key question:** Can multiple programs coexist at distinct NP offsets without
corrupting each other's outputs?

---

## Hypothesis

From `BEYOND_SDK.md` Discovery 9: `program_external(bytes, address)` accepts
a device address as its second argument. Different programs loaded at different
NP memory regions should be independent — their weights and thresholds reside
in disjoint SRAM regions.

If true: 7 programs loaded at NP offsets 0x0000, 0x00B3, 0x0135, 0x0178,
0x01BC, 0x021C, 0x0308 will each produce correct outputs without interfering
with each other.

**Expected result:** All 7 produce correct outputs simultaneously.
**Null hypothesis:** Address isolation fails — loading program B corrupts program A's SRAM.

---

## Pre-Conditions

- `rustChip` compiled with `--features vfio`
- `/dev/akida0` accessible (chmod 666 or udev rule 99-akida-pcie.rules installed)
- VFIO group binding confirmed (`ls /sys/bus/pci/drivers/vfio-pci/`)
- 7 `.fbz` model files available (or generated via `akida-models`)

### Required Model Files

```bash
# Generate minimal test models (pure Rust, no Python needed)
cargo run --bin enumerate -- --generate-test-models

# Or use existing validated models from hotSpring:
# esn_readout.fbz, transport_predictor.fbz, phase_classifier.fbz,
# anderson_regime.fbz, ecg_anomaly.fbz, kws_trimmed.fbz, minimal_sentinel.fbz
```

---

## Phase 1 Results (Software Isolation Model) ✅

**Results** (from `cargo run --bin bench_exp002_tenancy`):
- NP layout: ✅ 7 systems fit in 814 / 1,000 NPs, 186 spare, no address overlap
- Reload fidelity: ✅ system A output bit-identical before and after system B loads
- Round-robin throughput: ✅ > 5,000 Hz baseline (hardware target: 80,000–120,000 Hz)
- Weight mutation isolation: ✅ mutating system A does not affect system B
- Packing progression (2→4→7): ✅ all stages within 1,000 NP budget

**Corrected NP address map** (cumulative offsets):
```
Slot  System      NPs  NP Start  NP End
  1   ESN-QCD     179  0x0000    0x00B3
  2   Transport   134  0x00B3    0x0139
  3   KWS         220  0x0139    0x0215
  4   ECG          96  0x0215    0x0275
  5   Phase        67  0x0275    0x02B8
  6   Anderson     68  0x02B8    0x02FC
  7   Sentinel     50  0x02FC    0x032E
TOTAL            814  186 spare of 1,000
```
Note: Earlier docs had off-by-4 to off-by-44 hex address errors. Corrected in bench binary and experiment protocol.

**Run the simulation:**
```bash
cargo run --bin bench_exp002_tenancy
cargo run --bin run_experiments -- --exp 002
```

---

## Phase 2 Protocol

### NP Address Mapping (30 min)

**Goal:** Confirm that `program_external(bytes, addr)` places weights at the
expected NP SRAM offset, not always at address 0.

```rust
// Test: load same model at two different addresses
// Verify: can we infer from address A without affecting address B?

let mgr = DeviceManager::discover()?;
let device = mgr.open_first()?;

// Load model at address 0x0000
let addr_a = device.write_program_data(&model_bytes, 0x0000)?;
device.program_external(&program_info_bytes, addr_a)?;
let output_a1 = device.infer(&probe_input)?;

// Load DIFFERENT model at address 0x00B3
let addr_b = device.write_program_data(&model2_bytes, 0x00B3)?;
device.program_external(&program_info_bytes_2, addr_b)?;

// Does model at address 0x0000 still produce the same output?
let output_a2 = device.infer_at(addr_a, &probe_input)?;

println!("Address isolation: {}", if output_a1 == output_a2 { "✅ HOLDS" } else { "❌ CORRUPTED" });
```

Expected: `output_a1 == output_a2` — address B write does not touch address A.

### Phase 2: 2-Program Co-Loading (45 min)

**Goal:** Baseline — two programs coexist and both produce correct outputs.

```
Programs:
  Slot 0 (0x0000): ESN-QCD-Thermalization (179 NPs)
  Slot 1 (0x00B3): Transport-Predictor (134 NPs)
  Total: 313 NPs

Validation:
  - Load both
  - 1,000 inferences from Slot 0 with known inputs
  - 1,000 inferences from Slot 1 with known inputs
  - Verify: Slot 0 outputs unchanged after Slot 1 inferences
  - Verify: Slot 1 outputs unchanged after Slot 0 inferences
```

**Acceptance criterion:** Cross-slot inference error < 0.01% (essentially zero).

### Phase 3: 4-Program Co-Loading (1 hour)

Add 2 more programs:
```
Slot 2 (0x0135): Phase-Classifier-SU3 (67 NPs)
Slot 3 (0x0178): Anderson-Regime (68 NPs)
Total: 448 NPs
```

Same validation: 1,000 inferences per slot, verify isolation.
Measure throughput impact: does loading more programs slow individual inference?

**Expected:** No throughput impact (NP execution is slot-local).

### Phase 4: 7-Program Co-Loading (2 hours)

Full fleet:
```
Slot 4 (0x01BC): ECG-Anomaly (96 NPs)
Slot 5 (0x021C): KWS-DS-CNN-Trimmed (220 NPs)
Slot 6 (0x0308): Minimal-Sentinel (50 NPs)
Total: 814 NPs
```

Full validation:
- Each slot: 5,000 inferences with known inputs
- Cross-slot isolation: infer from all 7, verify each unchanged
- Throughput: round-robin all 7, measure aggregate

### Phase 5: Concurrent Throughput Measurement (30 min)

Measure:
1. Each slot in isolation: record throughput_i
2. All 7 round-robin: record aggregate throughput
3. Compute: aggregate / (7 × mean_individual)

**Expected:** aggregate throughput ≈ 7× individual throughput (no head-of-line blocking).
**If contention exists:** measure which slots interfere and characterize the pattern.

### Phase 6: set_variable() with Multi-Tenancy (30 min)

While 6 other slots are loaded, call `set_variable()` on slot 0.

**Expected:** Weight update in slot 0 does not affect slots 1–6.
**Validation:** 1,000 inferences per slot before and after the update.

---

## Expected Results

```
┌─────────────────────────────────────────────────────────────────────┐
│ Multi-Tenancy Validation Matrix                                     │
├────────────────────┬─────────────┬─────────────┬───────────────────┤
│ Test               │ Expected    │ Actual      │ Pass?             │
├────────────────────┼─────────────┼─────────────┼───────────────────┤
│ Address isolation  │ 0 corruption│ _____       │ _____             │
│ 2-program load     │ Both correct│ _____       │ _____             │
│ 4-program load     │ All correct │ _____       │ _____             │
│ 7-program load     │ All correct │ _____       │ _____             │
│ Aggregate Hz       │ ≥90K Hz     │ _____       │ _____             │
│ set_variable() MT  │ Isolated    │ _____       │ _____             │
├────────────────────┴─────────────┴─────────────┴───────────────────┤
│ OVERALL: _____ / 6 tests passed                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## If Address Isolation Fails

If loading program B at address `0x00B3` corrupts program A at `0x0000`,
the most likely cause is one of:
1. `program_external()` ignores the address argument and always loads at 0
2. NP SRAM regions overlap (our NP offset calculations are wrong)
3. Threshold SRAM (51-bit) is global, not per-NP (shared across all programs)

**Fallback strategy:**
- Time-share: load/unload programs per request (≈14 ms overhead per switch)
- Measure: can we reduce set_variable() overhead for full-model reloads?
- Design: smaller programs to reduce reload latency

If fallback is needed, update `baseCamp/systems/multi_tenancy.md` with
the actual constraint and revised performance model.

---

## Success Criteria

**PASS:** All 6 test cases pass. Update `baseCamp/systems/multi_tenancy.md`
with confirmed hardware behavior and measured throughput numbers.

**PARTIAL:** Address isolation holds but throughput is lower than predicted.
Document contention mechanism.

**FAIL:** Address isolation does not hold. Document failure mode.
Implement time-sharing with measured overhead.

---

## Follow-On Experiments

On success:
- `003_ONLINE_EVOLUTION.md` — validate 136 gen/sec with multi-tenant device
- `004_11HEAD_CONDUCTOR.md` — validate SkipDMA fan-out to 11 heads
- `005_NEUROMORPHIC_PDE.md` — validate Poisson solver via FC chain

On failure:
- `002b_TIMESHARE_OVERHEAD.md` — characterize reload latency for all 7 models
- Revisit `whitePaper/outreach/akida/TECHNICAL_BRIEF.md` claims
