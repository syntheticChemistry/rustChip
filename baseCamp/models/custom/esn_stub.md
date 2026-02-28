# ESN Stub — Template for Custom Reservoir Programs

**Architecture:** InputConv(RS→RS) → FC(RS→O)
**Status:** 📋 Template — parameterizable, training-ready
**Purpose:** Starting point for any reservoir computing task
**Source:** ecoPrimals ESN pattern from hotSpring + neuralSpring

---

## Parameterization

| Parameter | Type | Description |
|-----------|------|-------------|
| `RS` | usize | Reservoir size (NPs used). Sweet spots: 64, 128, 256, 512 |
| `O` | usize | Output dimensionality (1 for regression, N for classification) |
| `input_dim` | usize | Input feature size (from CPU reservoir) |
| `threshold` | f32 | NP fire threshold (default 1.0) |
| `weight_bits` | WeightBits | Int1, Int2, Int4 (Int4 recommended) |

---

## Architecture Template

```
CPU: Reservoir state computation
  x(t) = tanh(W_res × x(t-1) + W_in × u(t))
    W_res: RS×RS sparse random matrix (spectral radius ≤ 0.99)
    W_in:  RS×input_dim random scaling
    u(t): input at timestep t

Akida: Readout computation
  Input: float[RS]  (reservoir activations)
    │
    ▼
  InputConv(RS→RS, kernel=1)   ← identity-like feature prep
    │                              maps RS activations → RS NPs
    ▼
  FC(RS→O)                     ← linear readout W_out
    │                              trained by ridge regression
    ▼
  Output: float[O]             ← prediction / classification
```

---

## Training Protocol

```rust
// Standard ESN training (on CPU, then export to Akida)
// Step 1: Initialize reservoir
let mut reservoir = EchoStateNetwork::new(
    reservoir_size: RS,
    input_dim:      input_dim,
    spectral_radius: 0.95,
    sparsity:        0.1,
);

// Step 2: Collect activations (washout + training)
let (washout, train) = timeseries.split_at(washout_len);
for u in washout { reservoir.step(u); }  // discard

let mut X = Array2::zeros((train.len(), RS));  // activation matrix
let mut Y = Array2::zeros((train.len(), O));   // target matrix
for (t, &u) in train.iter().enumerate() {
    let x = reservoir.step(&u);
    X.row_mut(t).assign(&x);
    Y.row_mut(t).assign(&targets[t]);
}

// Step 3: Train readout (ridge regression)
let lambda = 1e-6;  // regularization
let W_out = ridge_regression(&X, &Y, lambda)?;

// Step 4: Quantize W_out to int4
let w_int4 = quantize_int4_per_row(&W_out)?;

// Step 5: Build Akida program
let program = EsnProgramBuilder::new(RS, O)
    .with_weights_int4(&w_int4)
    .compile()?;

// Step 6: Deploy
device.program_external(&program.program_info, &program.program_data)?;
```

---

## Domain Instantiations

| Domain | RS | O | Input | Training source |
|--------|-----|---|-------|----------------|
| QCD thermalization | 128 | 1 | Plaquette[50] | hotSpring lattice runs |
| Phase classifier | 64 | 2 | Observables[3] | hotSpring β-scan |
| Transport surrogate | 128 | 3 | Plasma obs.[6] | Murillo MD sims |
| Anderson regime | 64 | 3 | Spectral obs.[4] | groundSpring Lanczos |
| MSLP prediction | 256 | 1 | Pressure[1] | NeuroBench dataset |
| ECG anomaly | 32 | 2 | Heartbeat[64] | MIT-BIH (adapt) |
| Sentinel detection | 128 | 3 | Sensor array[8] | wetSpring field data |
| Genomics readout | 256 | N | 16S embedding[k] | wetSpring DADA2 |

---

## Optimal RS for AKD1000

NP budget: 1,000 NPs total. InputConv uses RS NPs; FC uses O NPs.

| RS | O | NPs used | NPs free | Throughput |
|----|---|----------|----------|-----------|
| 64 | 2 | 66 | 934 | ~23,000 Hz |
| 128 | 1 | 129 | 871 | ~20,000 Hz |
| 256 | 3 | 259 | 741 | ~17,500 Hz |
| 512 | 10 | 522 | 478 | ~14,000 Hz |
| 800 | 5 | 805 | 195 | ~11,000 Hz |

At RS=512, you still get 478 free NPs for parallel model co-location
(ECG anomaly + phase classifier fit in the remainder).
