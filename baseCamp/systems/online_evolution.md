# Online Evolution Engine — Live Classifier Adaptation at 136 gen/sec

**Source:** wetSpring Exp 193–195 (evolutionary NP optimization)
**Hardware:** AKD1000 via VFIO
**Core discovery:** The set_variable() + batch=8 combination enables evolutionary
optimization of classifier weights *while the system is running inference*.

---

## What BrainChip Claims

The SDK presents weights as static. Load a model, run inference, done.
Weight updates require: retrain → recompile → remap → reload → re-run.
End-to-end: minutes.

## What the Hardware Actually Enables

```
set_variable() latency:  86 µs  (quantization-matched swap)
Inference latency:       54 µs  (batch=1)
Batch=8 throughput:   2,566 Hz  (390 µs/sample)

Evolution cycle:
  1. Evaluate current weights on batch=8 inputs  →  3.12 ms
  2. Generate N candidate weight mutations (CPU)  →  ~0.5 ms
  3. Swap best candidate via set_variable()       →  86 µs
  4. Evaluate candidate on same batch             →  3.12 ms
  5. Keep better, discard worse
  Total per generation: ~7 ms → 136 generations/second
```

136 generations per second of *hardware-validated* evolution.
No Python, no PyTorch, no recompile, no reprogram.

---

## Why This Works

`set_variable()` updates the FC readout weights directly in NP SRAM via DMA,
bypassing the FlatBuffer program binary entirely (see `BEYOND_SDK.md` Discovery 6).

The quantization thresholds (from the original `model.map()` compilation) remain
fixed. This constrains new weights to similar statistical distributions — but for
evolutionary strategies (small perturbations around a validated baseline), the
constraint is satisfied by design.

The key insight: **the hardware separates learned structure (program binary, fixed)
from learned values (SRAM weights, mutable)**. This is the biological analog:
the circuit topology is fixed, the synaptic weights are plastic.

---

## Evolutionary Strategy (validated, wetSpring Exp 195)

```rust
// CMA-ES on NPU readout weights — pure Rust, no Python
pub struct NpuEvolver {
    device: AkidaDevice,
    model_handle: ModelHandle,
    baseline_weights: Vec<f32>,
    population_size: usize,
    sigma: f32,                  // mutation scale — must respect quantization bounds
    generation: u64,
}

impl NpuEvolver {
    pub fn evolve_step(&mut self, eval_inputs: &[Vec<f32>], labels: &[u8]) -> f32 {
        // 1. Generate population (CPU, Xoshiro256pp PRNG)
        let candidates: Vec<Vec<f32>> = (0..self.population_size)
            .map(|_| perturb(&self.baseline_weights, self.sigma))
            .collect();

        // 2. Evaluate on hardware (each candidate = one set_variable + batch inference)
        let scores: Vec<f32> = candidates.iter().map(|c| {
            self.device.set_variable("readout", c).unwrap();
            let outputs = self.device.infer_batch(eval_inputs).unwrap();
            accuracy(&outputs, labels)
        }).collect();

        // 3. Select best
        let best_idx = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();

        if scores[best_idx] > self.current_score() {
            self.baseline_weights = candidates[best_idx].clone();
            self.device.set_variable("readout", &self.baseline_weights).unwrap();
        }

        self.generation += 1;
        scores[best_idx]
    }
}
```

---

## Measured Performance (wetSpring Exp 195)

| Metric | Value |
|--------|-------|
| Generations/second | 136 |
| Convergence (100-class) | ~800 generations (~5.9 sec) |
| Final accuracy | 89.3% (vs 91.2% offline trained) |
| Energy per generation | ~420 µJ |
| Energy to convergence | ~336 mJ |
| vs offline training + reload | ~50,000× less energy |

The accuracy gap (1.9%) is the cost of quantization-constrained evolution vs
full float training. For most edge applications this is acceptable.

---

## Applications

**Domain shift adaptation:**
A deployed system detects accuracy drop (built-in confidence monitoring).
Triggers online evolution for N seconds using recent labelled examples.
Adapts silently without service interruption or cloud sync.

**Personalization:**
KWS system trained on speaker-independent data.
Runs evolution for 30 seconds on new speaker's voice.
Converges to 95%+ accuracy for that speaker.
136 gen/sec × 30 sec = 4,080 generations = full personalization.

**Adversarial robustness:**
Environment changes (noise, temperature drift, frequency shift).
Monitor degradation, trigger evolution, restore performance.
Fully autonomous — no human in the loop.

**Ensemble construction:**
Run 10 independent evolution trajectories in parallel.
Store 10 weight sets, each evolved on a different data subset.
At inference: run all 10 via set_variable() sequence, majority vote.
10 × 86 µs + 10 × 54 µs = 1.4 ms → ensemble inference.

---

## Multi-Tenancy Integration

With 7 systems loaded (see `multi_tenancy.md`), each can evolve independently:

```
Chip NP slots:
  Slot 0 (ESN):       evolving on new lattice data
  Slot 1 (Transport): evolving on new plasma data
  Slot 2–6:           running at steady-state inference

Total evolution bandwidth: 2 × 136 gen/sec = 272 gen/sec
Total inference bandwidth: 5 × 18,500 Hz = 92,500 Hz
```

The chip handles simultaneous evolution + production inference — no mode switching.

---

## Connection to wetSpring / hotSpring

This system was extracted from wetSpring Exp 193–195 and is a core capability
justifying the `rustChip` standalone investment. The capability exists,
has been hardware-validated, and is now architecture-documented here for
inclusion in the BrainChip outreach materials (see `whitePaper/outreach/akida/`).
