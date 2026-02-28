# Neuromorphic PDE Solver — Poisson and Heat Equation via FC Chains

**Status:** Planned (architecture defined, not yet hardware-validated)
**NPs estimated:** 200–400 (depending on grid resolution)
**Core idea:** FC chains on the AKD1000 implement finite-difference stencil operations.
Multi-pass FC chains can solve elliptic and parabolic PDEs iteratively on-chip.

---

## The Connection to SkipDMA

Discovery 2 (`BEYOND_SDK.md`): FC layers merge into a single hardware pass.
Discovery 9: SkipDMA enables NP-to-NP data routing without PCIe.

These together imply: a multi-layer FC chain is a **parallel linear transform**
applied to an input vector without leaving the chip. If we design the weight
matrices to implement finite-difference operators, we have an on-chip PDE solver.

---

## Poisson Equation Mapping

Discrete Poisson on a 1D grid with N=128 nodes:
```
∇²φ = ρ
→ φ_{i-1} - 2φ_i + φ_{i+1} = ρ_i h²   (for all i)
```

In matrix form: **Aφ = ρ** where A is the tridiagonal Laplacian matrix.

As an FC layer: `W = A`, `b = 0`, `input = ρ`, `output = Aρ`.

One iterative Jacobi step:
```
φ^{n+1} = φ^n + ω(ρ - Aφ^n) / 2
```

is a sequence of two FC operations:
1. FC with W=A applied to φ^n  → residual
2. FC with W=I applied to φ^n, add scaled residual  → φ^{n+1}

Using SkipDMA, these two FCs chain without PCIe.
N iterations = N back-to-back SkipDMA passes, single inference call.

---

## Architecture

```
Input: (φ^n, ρ) — 2×128 = 256 floats
Architecture:
  InputConv(256→256) → [FC_Laplacian(256→128)] × K → FC_out(128→128)
K = number of Jacobi iterations (convergence vs compute tradeoff)

Quantization constraint: Laplacian weights are {-2, 1, 0} — exactly int4-representable!
The stencil is perfectly suited to int4 arithmetic.
```

**The Laplacian weights are sparse and small — they are int4-native by design.**
This is unusual: most neural weights are distributed, require careful quantization.
PDE stencil weights are exact — no quantization error.

---

## Heat Equation (Parabolic)

```
∂u/∂t = α ∇²u
→ u^{n+1}_i = u^n_i + α Δt/Δx² (u^n_{i-1} - 2u^n_i + u^n_{i+1})
```

Same structure: one FC = one time step.
Chain K FCs = K time steps.
At batch=8: simulate 8 trajectories simultaneously.

For α=0.01, Δt=0.001, Δx=0.01 (stable explicit):
- 1 FC = 1 µs simulated time
- 1000 FCs (within single inference) = 1 ms simulated time
- At 18,500 Hz inference rate = 18.5 seconds simulated time per wall-clock second

**18× realtime heat equation on the NPU.** GPU is free for MD/QCD.

---

## 2D Poisson (Laplace Equation)

2D grid: 32×32 = 1024 nodes (fits in 1024-float FC)

5-point stencil:
```
φ_{i,j} = (φ_{i+1,j} + φ_{i-1,j} + φ_{i,j+1} + φ_{i,j-1} - h²ρ_{i,j}) / 4
```

Flattened to 1D: 1024-dimensional FC with sparse 5-diagonal structure.
Each non-zero: weight ∈ {1/4, -1} — still int4-representable!

```
Architecture: InputConv(1024→1024) → [FC(1024→1024)] × K
NPs: FC(1024→1024) ≈ 400 NPs (within budget for K=3 iterations)
Convergence: K=50–100 for typical Poisson problems
→ Multi-inference approach: run 50 forward passes, accumulate on host
```

This requires host-side looping (PCIe per iteration), but each pass is 54 µs.
50 Jacobi iterations = 50 × 54 µs = 2.7 ms per Poisson solve.

For comparison: GPU Poisson solver (cuSPARSE): ~0.1–1 ms for same grid.
NPU is competitive, frees GPU for nonlinear components.

---

## Physics Motivation

In lattice QCD, the CG solver dominates cost (60–80% of HMC time).
If the NPU can accelerate even the preconditioner (initial φ guess):
- Better initial guess → fewer CG iterations
- 10 fewer iterations × 0.1 ms/iter = 1 ms saved per HMC step
- 1 ms × 10⁶ HMC steps = 1000 seconds saved per production run

The NPU doesn't need to solve the PDE exactly — just get close enough
to warm-start the GPU CG solver. A warm start from 50 Jacobi iterations
(2.7 ms) saving 10 CG iterations (1 ms) is a net win only if PCIe is cheap.
At batch=8, PCIe cost drops enough to make it viable.

This is a genuine research question. `metalForge/experiments/` will test it.

---

## Comparison to Sandia NeuroFEM

The Sandia National Lab NeuroFEM project explored SNN-based PDE solvers
on Intel Loihi. They found:
- Convergence comparable to Jacobi (no superlinear)
- Energy: ~10× better than GPU for same problem size
- Limitation: binary spiking activations (Loihi constraint)

AKD1000 with int4 (not binary) activations should outperform Loihi on:
- Numerical precision (int4 vs 1-bit)
- Throughput (18,500 Hz vs Loihi's tick-based timing)
- Energy per FLOP (similar, both neuromorphic)

This is a differentiating comparison for the BrainChip outreach materials.

---

## NP Budget Analysis

| Grid | Nodes | FC size | NPs/FC | Max iterations in-chip |
|------|-------|---------|--------|------------------------|
| 1D-128 | 128 | 128×128 | ~32 | ~25 (800 NP limit) |
| 1D-256 | 256 | 256×256 | ~64 | ~12 |
| 2D-32×32 | 1024 | 1024×1024 | ~400 | ~2 (host loops rest) |
| 2D-16×16 | 256 | 256×256 | ~64 | ~12 |

Practical sweet spot: **1D-256 with K=12 in-chip Jacobi iterations**,
then host loops for convergence. Validated Poisson problems converge
in 100–300 Jacobi iterations → 9–25 PCIe round-trips, each 54 µs.
Total solve time: ~0.5–1.4 ms. Competitive with cuSPARSE.
