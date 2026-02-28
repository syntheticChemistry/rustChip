# Anderson Regime Classifier

**Architecture:** InputConv(4→64) → FC(64→3)
**Status:** ✅ VALIDATED on real AKD1000 (groundSpring Exp 028)
**Task:** Classify Anderson localization regime: localized / diffusive / critical
**Source:** groundSpring Exp 028; spectral theory validated in Exp 008, 009, 012, 018

---

## The Problem

The Anderson localization model (tight-binding Hamiltonian with random disorder)
exhibits three regimes depending on disorder strength W:

| Regime | Disorder W | Eigenstates | Transport |
|--------|-----------|-------------|-----------|
| Diffusive | W < 4.0 (3D) | Extended | Normal diffusion |
| Critical | W ≈ W_c (16.26) | Multifractal | Sub-diffusive |
| Localized | W > W_c | Exponentially decaying | Insulating |

The classifier takes 4 spectral observables from a computed eigenspectrum
and classifies into one of three regimes — enabling automatic disorder-strength
estimation without brute-force eigenvalue computation.

---

## Inputs (4 spectral observables)

| Index | Observable | Notes |
|-------|-----------|-------|
| 0 | Level spacing ratio r | Mean r = ⟨r_n⟩ where r_n = min(δ_n, δ_{n+1})/max(...) |
| 1 | Participation ratio IPR | Inverse participation ratio, normalized |
| 2 | Spectral gap ratio | E_gap / E_bandwidth |
| 3 | DOS curvature | Second derivative of density of states at band center |

---

## Architecture

```
Input: float[4]  (4 spectral observables)
  │
  ▼
InputConv(in=4, out=64, kernel=1)
  │
  ▼
FC(in=64, out=3)
  │  softmax → {localized, diffusive, critical}
  ▼
Output: float[3]  (class probabilities)
```

---

## Measured Performance (groundSpring Exp 028)

| Metric | Value |
|--------|-------|
| Throughput | 22,400 Hz |
| Latency | 45 µs |
| Energy | 1.0 µJ |
| Accuracy (localized) | 99.2% |
| Accuracy (diffusive) | 99.1% |
| Accuracy (critical) | 87.3% |
| W_c estimate | 16.26 ± 0.95 |

The lower accuracy at the critical point is physically expected — the
multifractal eigenstates at criticality genuinely interpolate between the
two phases. The 87% accuracy at criticality correctly captures the
ambiguous nature of the transition.

---

## Critical Point Discovery

The 3D Anderson model at W_c = 16.26 ± 0.95 is confirmed across:
- wetSpring Exp 107–156 (3,100+ checks, Anderson-QS domain)
- groundSpring Exp 008, 009, 012, 018 (spectral theory validation)
- groundSpring Exp 028 (NPU hardware classification)

The NPU classifier correctly identifies W_c = 16.26 from spectral observables
alone, matching the theoretical prediction without being given W directly.

---

## Cross-Spring References

This model sits at the intersection of the entire ecoPrimals ecosystem:

| Spring | Connection |
|--------|-----------|
| wetSpring | Anderson-QS domain (papers 01, 06); quorum sensing geometry |
| hotSpring | Spectral theory (Lanczos, Anderson 3D), Exp 022 hardware platform |
| groundSpring | Training data + validation Exp 028 (primary source) |
| neuralSpring | ESN anomaly detection at Anderson transitions |
| airSpring | Soil moisture diffusion → Anderson-like disorder |
| rustChip | Hardware classification (this file) |

---

## Extension: Quantitative W Estimator

Instead of classifying regimes, a regression model could estimate W directly:

```
InputConv(4→64) → FC(64→1)
  output: float W (disorder strength estimate, 0.0–25.0)
```

This would enable real-time disorder-strength monitoring in:
- Biological systems (wetSpring quorum sensing geometry)
- Soil pore network analysis (airSpring no-till detection)
- Synthetic Anderson lattices (groundSpring metalForge)

Training data already exists from groundSpring's 28 experiments.
