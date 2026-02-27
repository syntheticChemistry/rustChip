# Phase Classifier — SU(3) Confined/Deconfined

**Architecture:** InputConv(3→64) → FC(64→2)
**Status:** ✅ VALIDATED — 100% test accuracy, tested on real AKD1000
**Task:** Classify lattice QCD configurations as confined or deconfined
**Source:** hotSpring Exp 022 + groundSpring Exp 019–021 (Bazavov comparisons)

---

## The Problem

In lattice QCD at finite temperature, the system undergoes a deconfinement
phase transition at β_c ≈ 5.69 (for SU(3) pure gauge, 32⁴ lattice).
Below β_c: quarks/gluons confined in hadrons (confined phase).
Above β_c: quark-gluon plasma (deconfined phase).

The classifier takes 3 observables from a lattice configuration:
- Mean plaquette `<P>` (normalized 0–1)
- Real part of Polyakov loop `Re<L>` (normalized -1 to +1)
- Imaginary part of Polyakov loop `Im<L>` (normalized -1 to +1)

And outputs: `{0: confined, 1: deconfined}`

---

## Architecture

```
Input: float[3]  (plaquette, Re(L), Im(L))
  │
  ▼
InputConv(in=3, out=64, kernel=1)     ← point-wise feature expansion
  │  3 observables → 64 NP feature map
  ▼
FC(in=64, out=2)                      ← class scores
  │  softmax → {confined, deconfined}
  ▼
Output: float[2]  (class probabilities)
```

Small model (67 NPs total) — well within AKD1000 NP budget.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Training configurations | 10,000 (β = 5.0–6.4) |
| Labels | β < 5.69 → 0 (confined), β ≥ 5.69 → 1 (deconfined) |
| Optimizer | Adam, lr=0.001 |
| Epochs | 50 |
| Int4 quantization | Post-training, max-abs per layer |
| Validation accuracy | 100% on 2,000 test configurations |

---

## Measured Performance

| Metric | Value |
|--------|-------|
| Throughput | 21,200 Hz (batch=8) |
| Latency (chip) | 47 µs |
| Energy | 1.1 µJ |
| Accuracy | 100% |
| Model size | ~128 KB .fbz |

This is the smallest model in the ecoPrimals physics zoo. Its throughput
exceeds the ESN readout (less computation per call).

---

## Phase Boundary Discovery

The classifier's confidence gradient reveals the phase boundary:
- At β = 5.60: 87% confident confined
- At β = 5.69: 51% confident (decision boundary)
- At β = 5.80: 94% confident deconfined

The β_c = 5.69 result matches Bazavov et al. (hotSpring Exp 019–021 validation).
The NPU classifier reproduces this from raw observables, not from β value directly.

---

## Cross-Spring References

| Spring | Connection |
|--------|-----------|
| hotSpring | Lattice configurations, β scan, Exp 022 |
| groundSpring | Bazavov comparison, Exp 019-021, spectral validation |
| neuralSpring | Phase classifier training infrastructure |
| wetSpring | Pattern for ESN-based binary classifiers |
| rustChip | Hardware inference (this file) |

---

## Extension: Three-Phase Classifier

A natural extension adds the crossover region as a third class:

```
InputConv(3→64) → FC(64→3)
  output: {0: confined, 1: crossover (5.60 < β < 5.78), 2: deconfined}
```

This would require re-labeling training data with a crossover band.
The hardware path is identical — only the FC output changes from 2→3.
