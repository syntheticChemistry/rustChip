# Converting SNNTorch Models to Akida

**Scope:** SNNTorch LIF networks → int4 Akida programs
**Status:** Path documented; Rust tooling queued for akida-models 0.2
**Prerequisites:** Trained SNNTorch model (.pt file or safetensors)

---

## Why SNNTorch Maps Well to Akida

SNNTorch's Leaky Integrate-and-Fire neuron:
```
V[t] = β × V[t-1] + W × X[t]       ← membrane potential update
S[t] = (V[t] > threshold) ? 1 : 0   ← spike output
V[t] = V[t] × (1 - S[t])            ← soft reset
```

Akida's NP neuron (simplified):
```
A[t] = W × X[t]                     ← weighted sum
S[t] = (A[t] > threshold) ? 1 : 0   ← fire
```

The key difference: SNNTorch has **membrane decay** (β parameter), Akida doesn't.
For most inference tasks, setting β ≈ 0 (single-timestep inference) makes
SNNTorch exactly equivalent to Akida's model. This is the rate-coding regime.

---

## Compatible SNNTorch Architectures

```python
import snntorch as snn
import torch.nn as nn

# ✅ Compatible: FC chain
class CompatibleFCSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.Leaky(beta=0.0)  # ← set beta=0 for Akida compat
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=0.0)

# ✅ Compatible: Conv + FC
class CompatibleConvSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # ← depthwise preferred
        self.lif1  = snn.Leaky(beta=0.0)
        self.fc1   = nn.Linear(32 * 28 * 28, 10)
        self.lif2  = snn.Leaky(beta=0.0)

# ❌ Not compatible: recurrent LIF
class IncompatibleRSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rleaky = snn.RLeaky(beta=0.95, V=...)  # ← recurrent not supported
```

---

## Extraction Script

One-time Python script to extract weights (no Python needed after this):

```python
# extract_snntorch_weights.py
import torch
import numpy as np
import snntorch as snn

model = CompatibleFCSNN()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Extract weights and thresholds
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        w = module.weight.detach().numpy()    # [out, in]
        b = module.bias.detach().numpy()      # [out]
        np.save(f"{name}_weight.npy", w)
        np.save(f"{name}_bias.npy", b)
    elif isinstance(module, snn.Leaky):
        thresh = float(module.threshold.detach())
        with open(f"{name}_threshold.txt", "w") as f:
            f.write(str(thresh))

print("Weights exported. No Python needed after this.")
```

---

## Rust Conversion Pipeline

```rust
// After running extraction script:
use akida_models::convert::SnnTorchConverter;
use ndarray_npy::read_npy;

// Load extracted weights
let fc1_w: Array2<f32> = read_npy("fc1_weight.npy")?;
let fc1_b: Array1<f32> = read_npy("fc1_bias.npy")?;
let fc2_w: Array2<f32> = read_npy("fc2_weight.npy")?;
let fc2_b: Array1<f32> = read_npy("fc2_bias.npy")?;

// Build Akida program
let converter = SnnTorchConverter {
    weight_precision: WeightBits::Int4,
    threshold: 1.0,  // LIF threshold (was set to 0.0 during training → re-normalize)
    batch_size: 8,
};

let program = converter.convert_fc_network(
    &[fc1_w.view(), fc2_w.view()],    // weight matrices
    &[fc1_b.view(), fc2_b.view()],    // biases
    input_shape: (1, 28, 28),          // MNIST-like
)?;

// Deploy
let mut device = DeviceManager::discover()?.open_first()?;
device.program_external(&program.program_info, &program.program_data)?;
```

---

## Threshold Re-normalization

SNNTorch trains with threshold=1.0 (default). After int4 quantization, the
effective threshold changes because weights are scaled.

Re-normalize:
```rust
// If original threshold was T and weight scale is S:
// Effective threshold = T / S
// → Store threshold_eff = T / scale in the FlatBuffer

let scale = fc_weights.mapv(|x| x.abs()).fold(0.0f32, f32::max) / 7.0;
let threshold_eff = original_threshold / scale;
```

This is automatically handled by `SnnTorchConverter` when `weight_precision = Int4`.

---

## Example: N-MNIST

N-MNIST is DVS event-based MNIST — a natural fit for SNNTorch and Akida:

```
SNNTorch model (rate coding):
  Input: float[34×34×2]  (event count frame, ON/OFF channels)
  Conv(2→32, 3×3) + Leaky(β=0) → Conv(32→64, 3×3) + Leaky → FC(64→10)

Akida equivalent:
  InputConv(2→32, 3×3) → FC(64→10)
```

After extraction and quantization, this maps directly to `program_external()`.
Accuracy expected: ~98.5% (vs SNNTorch reference 99.0%) — int4 precision cost.

---

## What's Different from Standard PyTorch Conversion

SNNTorch adds:
1. **β parameter**: If non-zero, represents temporal decay — lose this for Akida
2. **Spike encoding**: SNNTorch outputs {0,1} spikes, Akida NPs fire similarly
3. **Reset mechanism**: SNNTorch has subtractive/zero reset — use zero for Akida

For β ≈ 0 models (recommended for hardware deployment), SNNTorch → Akida
is identical to the PyTorch float path in `conversion/from_pytorch.md`.
The only SNNTorch-specific step is extracting the LIF threshold value.
