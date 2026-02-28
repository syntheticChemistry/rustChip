# Transport Predictor — D*, η*, λ* from Plasma Observables

**Architecture:** InputConv(6→128) → FC(128→3)
**Status:** ✅ VALIDATED — all outputs finite, tested on AKD1000 (hotSpring Exp 022)
**Task:** Predict 3 reduced transport coefficients from 6 plasma observables
**Source:** hotSpring Murillo group WDM simulations + neuralSpring WDM surrogates

---

## The Problem

Warm Dense Matter (WDM) transport coefficients — diffusion D*, viscosity η*,
thermal conductivity λ* — are expensive to compute from first-principles MD
simulation. The surrogate approach: train a fast neural network on MD simulation
results, then use it as a real-time predictor.

On Akida: the surrogate runs at 18,500+ Hz, enabling real-time transport
prediction inside running simulations without breaking the GPU computation loop.

---

## Inputs (6 observables)

All normalized to [0, 1] or [-1, 1] from physical ranges:

| Index | Observable | Physical range | Notes |
|-------|-----------|---------------|-------|
| 0 | Temperature T | 0.01–100 eV | log-scaled, normalized |
| 1 | Density n | 10¹⁸–10²⁶ cm⁻³ | log-scaled |
| 2 | Coupling Γ | 0.01–100 | Γ = Ze²/(a k_B T) |
| 3 | Degeneracy θ | 0.001–10 | θ = T/T_F |
| 4 | Ion charge Z | 1–92 (H–U) | normalized by 92 |
| 5 | Mass ratio A/Z | 1–2.5 | normalized |

---

## Architecture

```
Input: float[6]  (6 plasma observables, normalized)
  │
  ▼
InputConv(in=6, out=128, kernel=1)    ← feature expansion
  │
  ▼
FC(in=128, out=3)                     ← multi-output regression
  │
  ▼
Output: float[3]  (D*, η*, λ* — all log-scaled, in physical units)
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Training samples | 50,000 MD simulation points |
| Source | Murillo group papers, hotSpring Exp 001–009 |
| Loss | MSE on log-scaled outputs |
| Optimizer | Adam, lr=0.0005, cosine annealing |
| Training precision | float64 (hotSpring CPU reference) |
| Deployment precision | int4 (post-training quantization) |

---

## Measured Performance

| Metric | Value |
|--------|-------|
| Throughput | 17,800 Hz (batch=8) |
| Latency | 56 µs |
| Energy | 1.5 µJ |
| Mean relative error D* | 3.1% |
| Mean relative error η* | 3.8% |
| Mean relative error λ* | 4.2% |
| All outputs finite | ✅ |

The 3–4% error is acceptable for the steering role the surrogate plays —
it guides simulation configuration selection, with full MD runs confirming
key results.

---

## GPU+NPU Co-location Pattern

This is the primary model demonstrating the GPU+NPU co-location architecture
described in `whitePaper/explorations/GPU_NPU_PCIE.md`:

```
GPU (RTX 4070 / Titan V):
  Running full MD simulation
  Needs: transport coefficients every N steps
  Sends: 6 float32 observables → CPU → NPU

NPU (AKD1000):
  Receives observables via DMA
  Runs transport predictor at 17,800 Hz
  Returns: 3 coefficients

CPU:
  Mediates GPU → NPU → GPU data flow
  Manages both devices
  Next step: P2P DMA bypasses CPU (Phase D+)
```

At 17,800 Hz NPU throughput vs GPU's 60 Hz simulation loop, the NPU
can service the GPU's transport coefficient requests with 99.7% idle time.

---

## Cross-Spring References

| Spring | Connection |
|--------|-----------|
| hotSpring | WDM MD simulations, transport coefficient reference |
| neuralSpring | Surrogate training, nW-01 MLP surrogate, nW-05 ESN classifier |
| groundSpring | Uncertainty quantification, WDM precision experiments 025–027 |
| rustChip | Hardware surrogate execution (this file) |
