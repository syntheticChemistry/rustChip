# Converting PyTorch Models to Akida

**Scope:** Float PyTorch models → int4 Akida programs
**Status:** Path documented; Rust tooling queued for akida-models 0.2
**Prerequisites:** Trained PyTorch model, simple architecture (no LSTM/attention)

---

## Which PyTorch Models Are Compatible

Akida NPs implement:
1. **InputConv**: 2D depthwise + pointwise convolution (spatial downsampling)
2. **FullyConnected**: Standard FC layer with threshold
3. No recurrent cells, no attention, no normalization layers at inference time

Compatible architectures:
- FC networks (MLP): any depth
- Conv + FC (CNN): any standard CNN
- MobileNet-style (DWConv + PtConv): native Akida pattern
- ESN readouts (InputConv + FC): the ecoPrimals pattern

Not compatible without adaptation:
- LSTM / GRU / RNN (no recurrence on chip)
- Transformer / attention
- Residual connections (requires multi-head routing — AKD1500 path)
- BatchNorm at inference (fold into weights before quantization)

---

## Step-by-Step Conversion

### Step 1: Fuse BatchNorm into conv/FC weights

BatchNorm must be absorbed before quantization:

```python
# Python preprocessing (one-time; outputs weights only)
import torch

def fuse_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm parameters into conv weights."""
    std = (bn_var + eps).sqrt()
    w_fused = conv_weight * (bn_weight / std).reshape(-1, 1, 1, 1)
    b_fused = (conv_bias - bn_mean) * (bn_weight / std) + bn_bias
    return w_fused.numpy(), b_fused.numpy()

# Save fused weights
model.eval()
w, b = fuse_bn(...)
np.save("layer1_weights.npy", w)
np.save("layer1_biases.npy",  b)
```

After this step, you have plain numpy arrays — no PyTorch dependency needed.

### Step 2: Load numpy weights in Rust

```rust
// Planned: akida_models::convert::load_npy
// For now: use ndarray-npy crate

use ndarray_npy::read_npy;
use ndarray::Array2;

let weights: Array2<f32> = read_npy("layer1_weights.npy")?;
let biases:  Array1<f32> = read_npy("layer1_biases.npy")?;
```

### Step 3: Quantize

```rust
use akida_models::quantize::{quantize_int4_per_layer, pack_int4};

let w_int4 = quantize_int4_per_layer(weights.as_slice().unwrap());
let b_int4 = quantize_int4_per_layer(biases.as_slice().unwrap());

let w_packed = pack_int4(&w_int4);
let b_packed = pack_int4(&b_int4);
```

### Step 4: Build the Akida program

```rust
// Planned: akida_models::builder::ProgramBuilder
use akida_models::builder::{ProgramBuilder, LayerSpec, LayerType};

let program = ProgramBuilder::new()
    .input_conv(
        in_channels:  3,
        out_channels: 64,
        kernel_size:  3,
        weights:      &conv_w_packed,
    )
    .fully_connected(
        out_features: n_classes,
        weights:      &fc_w_packed,
        biases:       &fc_b_packed,
        threshold:    1.0,
    )
    .compile()?;
```

### Step 5: Run on hardware

```rust
let mut device = DeviceManager::discover()?.open_first()?;
device.program_external(&program.program_info, &program.program_data)?;

// Inference
let output = device.infer(&input_int8, &InferenceConfig::default())?;
```

---

## Accuracy Expectations

Int4 quantization typically loses 1–3% accuracy vs float32:

| Precision | Typical accuracy drop | Notes |
|-----------|----------------------|-------|
| Int8 | < 0.5% | Standard post-training quantization |
| Int4 | 1–3% | Akida native; acceptable for most tasks |
| Int4 + fine-tune | < 0.5% | QAT (Quantization-Aware Training) |

For the ecoPrimals physics models, we see near-zero loss because the
networks are small and the weight distributions are well-conditioned.

---

## Folding PyTorch Activations

PyTorch ReLU → Akida threshold:

| PyTorch | Akida equivalent |
|---------|-----------------|
| ReLU | Threshold at 0.0 |
| LeakyReLU | Not directly supported |
| ReLU6 | Threshold at 6.0 (clamped) |
| Sigmoid/Tanh | Not supported (use softmax at output only) |

Set `threshold` in the layer spec to match the PyTorch activation.

---

## Int4 Calibration

For models where max-abs quantization loses too much accuracy,
calibrate per-channel:

```rust
pub fn quantize_int4_per_channel(
    weights: &Array2<f32>  // [out_channels, in_features]
) -> (Array2<i8>, Vec<f32>) {  // (quantized, scales)
    let scales: Vec<f32> = weights.rows()
        .into_iter()
        .map(|row| row.mapv(|x| x.abs()).fold(0.0f32, f32::max))
        .collect();

    let quantized = Array2::from_shape_fn(weights.dim(), |(i, j)| {
        let s = if scales[i] > 0.0 { 7.0 / scales[i] } else { 0.0 };
        (weights[[i, j]] * s).round().clamp(-8.0, 7.0) as i8
    });

    (quantized, scales)
}
```

Per-channel quantization requires storing the scales alongside the program
and applying them to de-quantize outputs. This is supported in the
`program_external()` FlatBuffer format via the weight scale table.
