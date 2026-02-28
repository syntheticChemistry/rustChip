# Experiment 004 — Hybrid Tanh: Hardware Linear + Host Tanh Activation

**Status:** Phase 1 COMPLETE ✅ (software simulation) | Phase 2 PENDING (hardware dispatch)
**Hardware:** AKD1000 (BC.00.000.002), `/dev/akida0`
**Estimated time:** Phase 2: 2–3 hours
**Key question:** Can we configure the AKD1000 to produce pre-activation values
(the linear FC output before bounded ReLU), enabling the host to apply tanh?

**What unlocks on success:**
- hotSpring's tanh-trained weights deploy directly to hardware (no MetaTF, no retraining)
- Accuracy gap closes from 3.6% → ~0.5%
- `HybridEsn::with_hardware_linear()` becomes fully functional
- Hardware speed (18,500 Hz) + tanh accuracy (89.7%) simultaneously

---

## Background

The AKD1000 applies bounded ReLU as its fixed activation function. The reservoir
dynamics that make ESNs work (echo state property) rely on tanh for robust
behavior with arbitrary weight initialization.

`bench_hw_sw_parity` measured:
- Random weights + bounded ReLU: ~50% accuracy (chance) — degenerate reservoir
- Random weights + tanh: ~100% accuracy — expressive reservoir
- Purpose-designed weights + bounded ReLU: 86.1% (hotSpring Exp 022, live hardware ✅)
- Purpose-designed weights + tanh: 89.7% (software reference ✅)

The hybrid approach: use hardware for the expensive matrix multiply, host for tanh.
Cost: < 1 µs extra per step. Benefit: 3.6% accuracy recovery, arbitrary weight reuse.

---

## Approach A: Zero-Threshold Activation via FlatBuffer

The threshold SRAM (51-bit, BEYOND_SDK Discovery 9) stores per-NP activation
thresholds. If we set all thresholds to their maximum value, the bounded ReLU
becomes `clamp(x, 0, MAX)` — effectively linear for any reasonable weight scale.

The `program_external()` API allows us to inject a custom FlatBuffer program
with explicitly set threshold values.

```
Target: CreateProgramInfoDirect(..., RegisterValue, ...)
The RegisterValue entries include threshold registers.
If threshold = 2^51 - 1 (max): activation never clamps → linear pass-through
```

**Step 1:** Build a minimal 128-NP FC program via FlatBuffer.
**Step 2:** Set all threshold SRAM values to `2^51 - 1`.
**Step 3:** Inject via `program_external()`.
**Step 4:** Run inference. Measure: does output match `W·x` (linear) or `relu(W·x)`?

```rust
// In metalForge/npu/ or bench binary:
fn build_linear_fc_program(input_dim: usize, reservoir_dim: usize) -> Vec<u8> {
    use akida_models::flatbuffer::ProgramBuilder;
    ProgramBuilder::new()
        .add_layer_fc(input_dim, reservoir_dim)
        .with_activation(ActivationMode::Linear)  // sets threshold to max
        .build_flatbuffer()
}

fn test_linear_activation(device: &mut AkidaDevice) -> Result<bool> {
    let probe_input = vec![1.0f32; 50];
    let identity_weights = identity_matrix(50); // W = I → output should equal input

    let program = build_linear_fc_program(50, 50);
    device.program_external(&program, 0x0000)?;
    device.set_variable("fc_weights", &identity_weights)?;

    let output = device.infer(&probe_input)?;
    // If linear: output ≈ probe_input (identity transform)
    // If bounded relu: output = relu(probe_input) = probe_input (all positive anyway)
    // Test with negative inputs to distinguish:
    let neg_input = vec![-1.0f32; 50];
    let neg_output = device.infer(&neg_input)?;
    // Linear: neg_output ≈ -1.0. Bounded ReLU: neg_output = 0.0.
    let is_linear = neg_output.iter().all(|&x| x.abs() > 0.5);
    Ok(is_linear)
}
```

---

## Approach B: Saturating Scale Trick

If threshold manipulation is not accessible, a second approach:

Scale all weights by a very small factor ε (e.g., 0.001) so that all
pre-activation values are in the "approximately linear" region of bounded ReLU:

```
bounded_relu(x) ≈ x  for x ∈ [0, 0.1]  (within 5% of linear)
```

If we quantize weights to int4 with max-abs scale such that the maximum
activation is < 0.1, the bounded ReLU region is effectively linear.

The host then applies: `tanh(output / ε)` to recover the tanh-scaled result.

```rust
fn scale_for_linear_region(weights: &[f32], target_max: f32) -> (Vec<f32>, f32) {
    let max_abs = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = target_max / max_abs;
    let scaled: Vec<f32> = weights.iter().map(|&w| w * scale).collect();
    (scaled, scale)
}

// On host, post-hardware:
fn recover_tanh(hw_output: &[f32], inv_scale: f32, state: &[f32], alpha: f32) -> Vec<f32> {
    hw_output.iter().zip(state.iter())
        .map(|(&hw, &s)| {
            let pre_activation = hw * inv_scale;
            (1.0 - alpha) * s + alpha * pre_activation.tanh()
        })
        .collect()
}
```

Tradeoff: reduces weight resolution (fewer int4 levels used), may lower accuracy.
But for the proof-of-concept this is achievable today.

---

## Approach C: Direct BAR0 Register Write

From `BEYOND_SDK.md` Discovery 9: `format_cnp_tnp_common_config_registers()` exists
in the C++ engine. This configures the NP's comparator/threshold registers.

If BAR0 register mapping is complete (`metalForge/experiments/005_BAR0_PROBE.md`),
we can write threshold registers directly:

```rust
// Direct register write to set threshold to MAX:
device.bar0_write(NP_THRESHOLD_REGISTER_OFFSET + np_index * NP_STRIDE, u64::MAX)?;
```

This is the cleanest approach but depends on BAR0 register mapping.

---

## Protocol

### Phase 1: Approach B — COMPLETE ✅ (software simulation validated)

**Results** (from `cargo run --bin bench_exp004_hybrid_tanh`):
- Scale trick math: ✅ Approach B produces non-degenerate reservoir states
- Determinism: ✅ bit-identical across two runs (consistent with BEYOND_SDK Discovery 10)
- ε formula: ✅ RMS pre-activation confirmed within hardware range
- **Key finding**: Approach B partially fixes the bounded ReLU constraint — it prevents
  degenerate collapse (the main issue with random weights + bounded ReLU) but does NOT
  fully recover tanh accuracy because the lower clamp (rectification) discards sign info
  for negative pre-activations. Approach A (FlatBuffer threshold override) is the full fix.
- `HardwareEsnExecutor::step_linear_emulated()` implements Approach B math in software
- `ScaleTrickConfig::from_weights()` uses 3σ statistical bound for automatic ε selection

**Run the simulation:**
```bash
cargo run --bin bench_exp004_hybrid_tanh
cargo run --bin run_experiments -- --exp 004
```

### Phase 2: Approach A (2 hours) — FlatBuffer threshold override

Use the scale trick to verify the concept works end-to-end:

1. Take hotSpring's validated ESN weights (w_in, w_res, w_out — tanh-trained)
2. Scale down by ε = 0.01 (puts all activations in [0, 0.01] linear region)
3. Load via `HybridEsn::with_hardware_native()` with scaled weights
4. Run inference, recover on host: `tanh(output / ε)`
5. Compare to software tanh baseline

Measure:
- Does the recovered output match tanh reference within 5%?
- What is the effective throughput (hardware inference + host recovery)?
- Is the accuracy on QCD data preserved (within 1% of 89.7%)?

### Phase 2: Approach A (2 hours) — FlatBuffer threshold override

1. Build minimal FC FlatBuffer with `ActivationMode::Linear` threshold values
2. Inject via `program_external()`
3. Probe with negative inputs: if linear, outputs are negative
4. On success: load hotSpring weights with no scaling
5. Run inference, apply tanh on host
6. Measure accuracy on QCD data

### Phase 3: Integration test (1 hour)

1. Enable `HybridEsn::with_hardware_linear()` fully (replace the TODO stub)
2. Run `bench_hw_sw_parity` with `--hw` flag
3. Measure: HardwareLinear mode vs PureSoftware mode
4. Expected: ≤ 1 µs extra overhead for tanh, accuracy within 0.5% of software

---

## Success Criteria

| Test | Expected | Pass if |
|------|----------|---------|
| Negative input test | output < 0 (linear) | output < -0.1 for -1.0 input |
| QCD accuracy (Approach B) | ≥ 88% (within 2% of 89.7%) | ≥ 87% |
| QCD accuracy (Approach A) | ≥ 89% (within 1% of 89.7%) | ≥ 88% |
| Latency overhead | < 2 µs extra | measured < 5 µs total |
| Weight compatibility | hotSpring weights load unchanged | no retraining required |

---

## Failure Modes

**If threshold registers are read-only after compilation:**
Use Approach B (scale trick). It adds ~1% accuracy cost vs Approach A.
Document: "linear-only requires small-weight initialization."

**If FlatBuffer threshold field is not respected:**
Probe BAR0 register map. The threshold SRAM must be writable post-load
(it's how `set_variable()` works — weights are writable post-load).
The same mechanism should apply to threshold values.

**If hardware adds noise that prevents tanh recovery:**
The AKD1000 is deterministic (BEYOND_SDK Discovery 10 ✅).
No stochastic behavior — recovery is exact if the linear region assumption holds.

---

## On Success: What Changes

1. `HybridEsn::with_hardware_linear()` — remove the TODO stub, full implementation
2. `SubstrateSelector::for_weights()` — uncomment the hardware discovery block
3. `hotSpring` — `HybridEsn` API is now a drop-in hardware accelerator
4. `toadStool` — `SubstrateSelector` dispatches to 18,500 Hz NPU automatically
5. Update `baseCamp/systems/hybrid_executor.md` — mark HardwareLinear mode as validated
6. Update `whitePaper/explorations/TANH_CONSTRAINT.md` — add measurement results
7. Update `whitePaper/outreach/akida/TECHNICAL_BRIEF.md` — new capability claim

The claim enabled: **hotSpring's physics models deploy to AKD1000 at 18,500 Hz with
no retraining, no MetaTF, no bounded ReLU compromise, and no accuracy loss.**
