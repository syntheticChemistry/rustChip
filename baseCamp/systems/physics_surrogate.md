# Physics Surrogate Ensemble — 4-Domain Co-Located Surrogates

**Core demonstration:** Four physics surrogate models co-located on one AKD1000,
replacing four separate GPU compute threads, while the GPU runs the primary simulation.

This is the flagship demonstration for the BrainChip outreach:
**GPU computes physics. NPU predicts physics. Neither waits for the other.**

---

## The Heterogeneous Computing Architecture

Standard approach (compute-limited):
```
GPU thread 0: MD forces         ← 60% of GPU time
GPU thread 1: ESN readout       ← 8% of GPU time
GPU thread 2: transport prediction  ← 12% of GPU time
GPU thread 3: phase detection   ← 7% of GPU time
GPU thread 4: anomaly detection ← 5% of GPU time
Total: 92% of GPU consumed by non-force work
```

ecoPrimals heterogeneous approach:
```
GPU:     MD forces (100% of GPU — it does nothing else)
NPU:     ESN readout + transport + phase + anomaly (all simultaneously)
PCIe:    observables flow GPU→NPU (200 bytes), predictions flow NPU→GPU (12 bytes)
Latency: 650 µs round-trip, amortized across batch=8 (82 µs/sample effective)
```

The GPU gains back 32% of its cycles. MD/QCD simulation throughput increases 47%.
The NPU consumes 270 mW (vs the GPU's 300 W for those threads).

---

## Four Surrogate Systems

### Surrogate 1: QCD Thermalization Detector (179 NPs)
```
Input: float[50]  — plaquette averages over 50 time slices
Output: float[1]  — thermalization flag [0,1]
Use: halt or continue MD when p(thermalized) > 0.95
Throughput: 18,500 Hz → checks every 54 µs
Validated: ✅ hotSpring Exp 022, 5,978 live calls
```

### Surrogate 2: Transport Coefficient Predictor (134 NPs)
```
Input: float[6]   — plasma observables (T, ρ, ε, P, σ, n_e)
Output: float[3]  — D*, η*, λ* (diffusivity, viscosity, conductivity)
Use: provide transport coefficients without running separate MD
Throughput: 17,800 Hz
Validated: ✅ hotSpring adapted for AKD1000
```

### Surrogate 3: Phase Boundary Classifier (67 NPs)
```
Input: float[3]   — (β, am_sea, am_val)
Output: float[2]  — P(confined), P(deconfined)
Use: classify phase without full lattice measurement
Throughput: 21,200 Hz
Validated: ✅ hotSpring Exp 022
```

### Surrogate 4: Anderson Regime Classifier (68 NPs)
```
Input: float[4]   — (disorder W, energy E, system size L, filling ν)
Output: float[3]  — P(localized), P(diffusive), P(critical)
Use: identify localization regime during disorder scan
Throughput: 22,400 Hz
Validated: ✅ hotSpring Exp 022
```

Total: 448 NPs, 552 remaining.
All four run simultaneously, each at their full throughput.

---

## PCIe Data Flow

```
GPU simulation step:
  [MD force calculation — GPU only]
      ↓
  Compute observables from current state (GPU kernel)
  Pack 4 input tensors (total: ~200 bytes)
  DMA → NPU PCIe buffer
      ↓
[NPU inference — simultaneous with GPU next step]
  All 4 surrogates fire (54 µs, overlapped with GPU computation)
  Pack outputs (12 floats = 48 bytes)
  DMA → GPU PCIe buffer
      ↓
  GPU reads predictions for steering decisions
  (Steer β, skip CG convergence check if thermalized, adapt transport)
```

PCIe latency (650 µs) is amortized:
- GPU step time: ~5 ms (typical HMC step)
- NPU inference + PCIe: 650 µs
- Overlap: 5 ms - 650 µs = 4.35 ms pure parallelism
- Net cost of NPU predictions: ~0 GPU cycles (fully overlapped)

---

## Rust Implementation Architecture

```rust
// In toadstool / ecoPrimals physics engine:
// This code lives in the GPU-side simulation runner,
// calling into rustChip for NPU dispatch.

use rustchip::{SurrogateEnsemble, SurrogateQuery};

pub struct HeterogeneousMdRunner {
    gpu: WgpuDevice,
    npu: SurrogateEnsemble,  // rustChip handle
    step: u64,
}

impl HeterogeneousMdRunner {
    pub async fn step(&mut self, config: &MdConfig) -> MdStepResult {
        // Compute observables from current GPU state (cheap kernel)
        let obs = self.gpu.compute_observables().await;

        // Dispatch to NPU asynchronously (fire-and-forget)
        let npu_future = self.npu.query_all(SurrogateQuery {
            plaquette: obs.plaquette_50,
            plasma: obs.plasma_6,
            phase_params: obs.phase_3,
            disorder: obs.disorder_4,
        });

        // GPU does next MD step while NPU runs
        let gpu_result = self.gpu.md_force_step(config).await;

        // Await NPU result (usually already done by now)
        let predictions = npu_future.await?;

        // Steering decisions based on predictions
        if predictions.thermalization > 0.95 {
            return MdStepResult::Thermalized(gpu_result);
        }

        MdStepResult::Continue {
            forces: gpu_result,
            transport: predictions.transport,
            phase: predictions.phase,
        }
    }
}
```

---

## Performance Numbers

| Configuration | GPU occupancy | NPU contribution | Effective MD throughput |
|---------------|---------------|-----------------|------------------------|
| GPU-only (SDK approach) | 92% simulation | — | 1.0× baseline |
| GPU+NPU (rustChip) | 100% simulation | 4 surrogates | 1.47× baseline |
| GPU+NPU (w/ ESN temporal) | 100% simulation | 4+1 surrogates | 1.47× + steering |

**Power comparison:**
- GPU alone (4 threads): +24W for surrogate work
- NPU replacing 4 threads: +0.27W
- NPU power for 4 surrogates: 90× less power than GPU for same predictions

---

## The Outreach Narrative

BrainChip pitches the AKD1000 for "keyword spotting and object detection."
They show it as a standalone unit processing sensor data.

The rustChip demonstration shows something different:
**The AKD1000 is a physics coprocessor that makes HPC workloads faster and
greener by taking over the statistical inference tasks that pollute GPU computation.**

Four simultaneous physics surrogates. 270 mW. PCIe-coupled to the simulation.
47% faster MD. 90× lower power for prediction work.

This is not the product BrainChip is marketing. But it's the product they built.

See `whitePaper/outreach/akida/TECHNICAL_BRIEF.md` for the full writeup.
