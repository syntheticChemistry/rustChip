# SNNTorch and SNN Frameworks

**Status:** Reviewed February 2026. Conversion path documented; not yet implemented.

---

## SNNTorch

**URL:** https://github.com/jeshraghian/snntorch
**Paper:** Eshraghian et al. "Training Spiking Neural Networks Using Lessons From Deep Learning" (2023)
**Language:** Python / PyTorch
**License:** MIT
**Neurons:** Leaky Integrate-and-Fire (LIF), Synaptic Conductance (SConv2d), etc.

SNNTorch is the most popular SNN training framework. It produces PyTorch models
with spiking neuron dynamics. Getting these onto Akida requires:

```
SNNTorch model (PyTorch + LIF neurons)
  → extract float weights from LIF layers
    → quantize to int4 using QuantizeML or manual quantization
      → build Akida program binary (CNN2SNN or manual FlatBuffer)
        → load via akida-models + program_external()
```

### Compatibility Assessment

| SNNTorch component | Akida equivalent | Compatibility |
|-------------------|-----------------|---------------|
| `snn.Leaky` (LIF neuron) | FC layer with threshold | ✅ Direct weight extraction |
| `snn.RLeaky` (recurrent LIF) | ESN reservoir readout | ✅ Weight extraction + manual program |
| `snn.Conv2d` (spiking conv) | AKD1000 InputConv | ✅ With quantization |
| `snn.LSTM` | No direct Akida equivalent | ❌ Not mappable |
| Rate encoding (float→spikes) | Akida handles automatically | ✅ |
| TTFS encoding | Not standard in Akida SDK | ⚠️ Requires testing |
| BPTT (through time) training | Use post-training quantization | ✅ Extract weights after training |

### Conversion Path (planned for `akida-models`)

```rust
// Future: akida_models::convert::from_snntorch
// Input: path to SNNTorch weights file (.pt or .safetensors)
// Output: .fbz binary + metadata

pub struct SnnTorchConverter {
    weight_precision: WeightBits,  // Int1, Int2, Int4
    threshold: f32,
    batch_size: usize,
}

impl SnnTorchConverter {
    pub fn convert_fc_network(
        &self,
        weights: &[Array2<f32>],   // layer weight matrices
        biases:  &[Array1<f32>],   // layer biases
        input_shape: (usize, usize, usize), // (channels, height, width)
    ) -> Result<CompiledProgram> {
        // 1. Validate architecture maps to Akida (no LSTM, no attention)
        // 2. Quantize weights to int4: max-abs scaling per layer
        // 3. Build FlatBuffer program_info using known format
        // 4. Build program_data with quantized thresholds
        // 5. Return CompiledProgram with program_info + program_data + weights
        todo!("Phase: akida-models 0.2")
    }
}
```

### Example Models from SNNTorch that Map to Akida

| Model | Task | Architecture | Akida mapping |
|-------|------|-------------|---------------|
| MNIST FC-SNN | Digit classification | 784→256→10 LIF | FC chain, SkipDMA merge |
| N-MNIST Conv-SNN | Event camera MNIST | Conv+Pool+FC | InputConv + FC |
| SHD Audio | Heidelberg Digits | FC reservoir | ESN-like readout |
| NTIDIGITS | Audio keywords | Conv+LSTM | Conv only (drop LSTM) |

---

## Norse

**URL:** https://github.com/norse/norse
**Language:** Python / PyTorch / JAX
**Neurons:** LIF, ALIF, CuBaLIF (conductance-based), Izhikevich
**License:** MIT

Norse provides more biologically detailed neuron models than SNNTorch.
Compatibility with Akida is lower because:
- Akida's NPs implement a simplified event-based model (not conductance-based)
- Akida does not support the Izhikevich or AdEx neuron dynamics
- The quantization step is lossy for conductance-based models

**Practical path:** Extract rate-coded representations from Norse models,
quantize the effective weight matrix, then build Akida programs as if the
network were a standard FC or conv network. This loses the temporal dynamics
but preserves the learned representations.

---

## BindsNET

**URL:** https://github.com/BindsNET/bindsnet
**Language:** Python / PyTorch
**Neurons:** LIF, AdEx, Izhikevich, Hodgkin-Huxley
**Learning:** STDP (Spike-Timing Dependent Plasticity), R-STDP, BC-STDP
**License:** AGPL-3.0

BindsNET is the most biology-faithful framework. It supports STDP learning —
the same learning rule that BrainChip uses on-chip in the AKD1000's On-chip
Learning Engine (not yet accessible via external API).

**Interesting connection:** If BrainChip opens the on-chip learning register
path (Phase F in `specs/PHASE_ROADMAP.md`), BindsNET-trained STDP networks
could run their learning step on-chip rather than requiring PCIe weight uploads.

**Conversion path:** Same as SNNTorch — extract rate-coded weight matrices,
quantize, build Akida program. The STDP-trained weights are just float matrices
after training; the learning rule is irrelevant for inference deployment.

---

## Lava (Intel Loihi)

**URL:** https://github.com/lava-nc/lava
**Hardware:** Intel Loihi 2
**Language:** Python
**License:** BSD 3-Clause

Lava models run on Loihi 2, not Akida. However:
- The architecture concepts (LIF neurons, spike-based processing) overlap
- Models can be re-trained for Akida if architectures are compatible
- The comparison is scientifically interesting: Loihi 2 vs AKD1000 for same task

**Status:** Not a direct conversion target. Reference for architecture design.

---

## Summary: Framework → Akida Compatibility

```
Full compatibility (weight extraction + quantization → .fbz):
  SNNTorch   ─── FC, Conv → int4 → program_external()     ✅
  BindsNET   ─── FC, Conv → int4 → program_external()     ✅ (STDP weights)
  
Partial compatibility (architecture adaptation required):
  Norse      ─── Drop conductance dynamics → FC/Conv only  ⚠️
  Lava       ─── Re-train for Akida format               ⚠️
  
Not compatible:
  LSTM-based models (no recurrent cells in Akida FC NPs)   ❌
  Attention/Transformer-based SNN                          ❌
  Continuous-time (ODE-based) neurons                      ❌
```

The **ESN (Echo State Network)** architecture — InputConv + FC readout —
maps to Akida better than any other recurrent architecture because:
1. Reservoir weights are fixed after initialization (no BPTT, no LSTM)
2. Only the readout layer learns (FC weights, easily quantized to int4)
3. The architecture is exactly what ecoPrimals physics models use
