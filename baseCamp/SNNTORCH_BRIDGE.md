# SNNTorch / PyTorch → rustChip Conversion Bridge

## Overview

This document describes the conversion path from SNNTorch (spiking neural
network models in PyTorch) and standard PyTorch models to rustChip's `.fbz`
format for deployment on AKD1000 hardware.

## Architecture

```
SNNTorch model (Python)
   │
   ├─ torch.onnx.export() ──→ model.onnx ──→ akida import-onnx --weights model.onnx
   │                                          (Rust: onnx-rs → ImportedWeights → quantize → .fbz)
   │
   ├─ torch.save() ──→ weights.pt ──→ convert to .safetensors ──→ akida convert --weights w.safetensors
   │
   └─ snntorch.export() ──→ weights.npy ──→ akida convert --weights w.npy
```

## Conversion Paths

### Path A: ONNX Export (Recommended)

The cleanest path for standard CNN/DNN models. Works for any PyTorch model
that can be traced or scripted.

```python
import torch
import snntorch as snn
from snntorch import functional as SF

# 1. Define model
class SNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc1 = torch.nn.Linear(16 * 28 * 28, 10)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur1 = self.conv1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc1(spk1.flatten(1))
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2

# 2. Export to ONNX
net = SNN()
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(net, dummy, "snn_model.onnx",
                  input_names=["input"],
                  output_names=["spikes"],
                  opset_version=13)

# 3. Import in rustChip
# cargo run --bin akida -- import-onnx --weights snn_model.onnx
```

### Path B: Safetensors Export

For models where ONNX export is tricky (e.g., recurrent SNNs with state).

```python
import torch
from safetensors.torch import save_file

net = SNN()
net.eval()

# Extract state dict and save
tensors = {k: v for k, v in net.state_dict().items()}
save_file(tensors, "snn_weights.safetensors")

# Import in rustChip
# cargo run --bin akida -- convert \
#   --weights snn_weights.safetensors \
#   --arch "InputConv(16,3,1) FC(10)" \
#   --output snn_model.fbz
```

### Path C: NumPy Export

For SNNTorch models that use the built-in export utilities.

```python
import numpy as np

# After training, export each weight matrix
for name, param in net.named_parameters():
    np.save(f"weights/{name}.npy", param.detach().cpu().numpy())

# Import in rustChip (one layer at a time)
# cargo run --bin akida -- convert \
#   --weights weights/conv1.weight.npy \
#   --arch "InputConv(16,3,1)" \
#   --output conv1.fbz
```

## Hugging Face brainchip/* Models

BrainChip publishes pre-quantized models on Hugging Face that are already
in formats compatible with the Akida SDK. These can be loaded directly:

### Available Models

| Model | HF Path | Task | Size |
|-------|---------|------|------|
| AkidaNet ImageNet | `brainchip/akidanet_imagenet` | Classification | 5.0 MB |
| AkidaNet PlantVillage | `brainchip/akidanet_plantvillage` | Disease detection | 1.4 MB |
| AkidaNet Face ID | `brainchip/akidanet_faceidentification` | Verification | 2.8 MB |
| DS-CNN KWS | `brainchip/ds_cnn_kws` | Keyword spotting | 40 KB |
| YOLO VOC | `brainchip/yolo_voc` | Object detection | 4.3 MB |
| CenterNet VOC | `brainchip/centernet_voc` | Object detection | 2.8 MB |
| VGG UTK Face | `brainchip/vgg_utk_face` | Age estimation | 146 KB |
| UNet Portrait | `brainchip/akida_unet_portrait128` | Segmentation | 1.3 MB |

### Loading HF Models

If the Hugging Face model includes a `.fbz` artifact:

```bash
# Download
wget https://huggingface.co/brainchip/akidanet_imagenet/resolve/main/model.fbz

# Parse with rustChip
cargo run --bin akida -- parse model.fbz
```

If the model is in SavedModel/H5 format (requires Python conversion):

```python
import akida
from akida_models import fetch_file

model = akida.Model("brainchip/akidanet_imagenet")
model.save("akidanet_imagenet.fbz")
```

### Testing HF Models with rustChip

The model zoo already includes local copies of key HF models in
`baseCamp/zoo-artifacts/`. These are tested by:

1. `cargo run --bin model_zoo` — parses all zoo models
2. `cargo run --bin akida -- guidestone` — validates structure, checksums
3. The `preserve_*` binaries load and simulate inference per domain

## Quantization Notes

### SNNTorch → Akida Quantization

SNNTorch models typically use `float32` weights. Akida hardware uses
`int4` or `int8` quantized weights. The conversion requires:

1. **Post-training quantization (PTQ)**: Apply symmetric per-tensor or
   per-channel quantization after training. Use `akida_models::quantize`.

2. **Quantization-aware training (QAT)**: Use `QuantizeLinear` /
   `DequantizeLinear` ONNX ops or BrainChip's `quantizeml` library
   during training for best accuracy.

3. **Weight clipping**: AKD1000 int4 range is [-8, 7]. Weights outside
   this range after scaling are clipped, introducing quantization error.

### Precision Hierarchy

```text
PyTorch float32 → ONNX float32 → rustChip int8/int4
                                          ↓
                                  SoftwareBackend (f32 simulation)
                                          ↓
                                  AKD1000 hardware (int4 NP arithmetic)
```

Typical accuracy loss per stage:
- float32 → int8: 0.1–0.5% accuracy drop (PTQ)
- int8 → int4: 0.5–3.0% accuracy drop (depends on model)
- QAT int4: <0.5% accuracy drop vs float32 baseline

## Worked Example: MNIST SNN

Complete end-to-end example from training to hardware deployment:

```bash
# 1. Train in Python (snntorch_mnist_example.py)
python examples/snntorch_mnist.py --epochs 5 --export onnx

# 2. Import to rustChip
cargo run --bin akida -- import-onnx --weights mnist_snn.onnx

# 3. Convert to .fbz with quantization
cargo run --bin akida -- import-onnx --weights mnist_snn.onnx -o mnist_snn.fbz

# 4. Validate
cargo run --bin akida -- parse mnist_snn.fbz

# 5. Run on hardware (VFIO)
cargo run --bin preserve_vision
```

## Compatibility Matrix

| Source Framework | Export Format | rustChip Import | Status |
|-----------------|-------------|-----------------|--------|
| SNNTorch | ONNX | `import-onnx` | Supported |
| SNNTorch | .npy weights | `convert --weights` | Supported |
| PyTorch | ONNX | `import-onnx` | Supported |
| PyTorch | .safetensors | `convert --weights` | Supported |
| Keras/TF | SavedModel → ONNX | `import-onnx` | Via tf2onnx |
| BrainChip SDK | .fbz | Direct load | Native |
| BrainChip HF | .fbz artifact | Direct load | Native |
| Lava-DL | ONNX | `import-onnx` | Untested |

## Next Steps

- [ ] Add Python helper script (`tools/export_snn.py`) for automated conversion
- [ ] Implement calibrated quantization with representative dataset
- [ ] Add per-channel quantization for convolutions
- [ ] Test with real Hugging Face brainchip/* downloads
- [ ] Benchmark accuracy retention across the quantization pipeline
