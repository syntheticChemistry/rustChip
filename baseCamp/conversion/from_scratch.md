# Building Akida Programs from Scratch in Rust

**Scope:** Hand-craft any Akida-compatible network entirely in Rust
**Status:** `program_external()` path confirmed (BEYOND_SDK Discovery 3)
           `akida-models::builder` queued for 0.2
**Prerequisites:** rustChip (no Python, no MetaTF, no SDK)

---

## Why from scratch

Four reasons to build from scratch rather than converting from PyTorch:

1. **Physics constraints**: The ecoPrimals models are designed from the domain
   up — ESN reservoir size chosen for AKD1000 NP budget, not PyTorch convenience

2. **program_external() speed**: Pre-built FlatBuffer binaries load in microseconds.
   No SDK, no CUDA, no Python interpreter needed at runtime.

3. **set_variable() adaptation**: Hand-built programs can be designed to have
   mutable weight slots from the start, enabling 86 µs weight swaps

4. **Novel architectures**: Any architecture that doesn't exist in the PyTorch
   world — e.g., direct physics-motivated weight initialization — requires
   building from scratch

---

## The Pattern

```rust
// 1. Define weights in float (train using any Rust method)
let (w_out, _loss) = ridge_regression(&activations, &targets, lambda: 1e-6);

// 2. Quantize to int4
let w_int4 = quantize_int4_per_layer(w_out.as_slice().unwrap());
let w_packed = pack_int4(&w_int4);

// 3. Build FlatBuffer binary
use akida_models::builder::ProgramBuilder;

let program = ProgramBuilder::new()
    .sdk_version("2.18.2")
    .input_conv(InputConvSpec {
        in_features:  128,
        out_features: 128,
        kernel_size:  1,
        weights:      &conv_packed,
        threshold:    1.0,
    })
    .fully_connected(FcSpec {
        in_features:  128,
        out_features: output_dim,
        weights:      &w_packed,
        biases:       &b_packed,
        threshold:    1.0,
    })
    .build()?;

// 4. Inject onto chip
let mut device = DeviceManager::discover()?.open_first()?;
device.program_external(&program.program_info, &program.program_data)?;
```

---

## Designing for set_variable() Mutability

To enable weight swapping without full reprogramming, mark weight slots
as mutable in the FlatBuffer:

```rust
let program = ProgramBuilder::new()
    .fully_connected(FcSpec {
        in_features:  128,
        out_features: 2,
        weights:      &w_packed_classifier_A,
        mutable:      true,   // ← Discovery 6: marks this slot for set_variable()
        slot_id:      0,      // ← handle for later updates
        threshold:    1.0,
    })
    .build()?;

// Later, swap to classifier B without reprogramming:
let var = ProgramVariable { slot_id: 0, data: &w_packed_classifier_B };
device.set_variable(&var)?;   // 86 µs (vs ~50 ms full reprogramming)
```

This is the multi-classifier hot-swap pattern from BEYOND_SDK Discovery 6.
ecoPrimals uses it to run 3 ESN classifiers on one loaded program,
switching between physics domains at 86 µs intervals.

---

## ESN Training in Pure Rust

The ecoPrimals ESN training loop — no Python required:

```rust
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct EsnBuilder {
    reservoir_size: usize,
    spectral_radius: f32,
    input_scale:     f32,
    sparsity:        f32,
    seed:            u64,
}

impl EsnBuilder {
    pub fn build(&self) -> EchoStateNetwork { ... }
}

pub struct EchoStateNetwork {
    w_res: Array2<f32>,   // sparse random reservoir
    w_in:  Array2<f32>,   // input weights
    state: Array1<f32>,   // current reservoir state
}

impl EchoStateNetwork {
    pub fn step(&mut self, input: &[f32]) -> &Array1<f32> {
        let u = Array1::from_slice(input);
        self.state = (self.w_res.dot(&self.state) + self.w_in.dot(&u))
            .mapv(f32::tanh);
        &self.state
    }

    pub fn collect_activations(
        &mut self,
        inputs: &[Vec<f32>],
        washout: usize,
    ) -> Array2<f32> {
        for inp in &inputs[..washout] { self.step(inp); }
        inputs[washout..].iter()
            .map(|inp| self.step(inp).to_owned())
            .collect::<Vec<_>>()
            .into()
    }
}

// Ridge regression for readout
pub fn ridge_regression(
    X: &Array2<f32>,  // [n_samples, reservoir_size]
    Y: &Array2<f32>,  // [n_samples, output_dim]
    lambda: f32,
) -> Array2<f32> {   // W_out: [output_dim, reservoir_size]
    // W = (X^T X + λI)^-1 X^T Y
    let xt  = X.t();
    let xtx = xt.dot(X);
    let reg = Array2::eye(xtx.nrows()) * lambda;
    let xtx_reg = xtx + reg;
    let xtx_inv = xtx_reg.inv().expect("matrix not invertible");
    xtx_inv.dot(&xt.dot(Y)).t().to_owned()
}
```

This is the complete Rust training loop. No PyTorch. No Python.
`train → quantize → program_external()` — pure Rust from raw data to silicon.

---

## FlatBuffer Format Reference

For direct FlatBuffer construction (bypassing `ProgramBuilder`), see:
- `crates/akida-chip/src/program.rs` — the confirmed Rust model
- `metalForge/npu/akida/REGISTER_PROBE_LOG.md` — raw format derivation
- `specs/SILICON_SPEC.md` — FlatBuffer section with byte layout

The key offsets (AKD1000, firmware 2.x):

```
Offset 0x00: 4 bytes — FlatBuffer size (LE uint32)
Offset 0x04: 4 bytes — table root offset
Offset 0x08: N bytes — version string (null-terminated UTF-8, "2.18.2")
Offset 0x10: M bytes — layer table (variable, depends on architecture)
  Each layer entry:
    0x00: 1 byte  — layer type (0x01=InputConv, 0x02=FC)
    0x01: 2 bytes — in_features
    0x03: 2 bytes — out_features
    0x05: 4 bytes — weight offset (relative to program_data start)
    0x09: 4 bytes — weight size (bytes)
    0x0D: 4 bytes — threshold (float32)
    0x11: 1 byte  — mutable flag
    0x12: 1 byte  — slot_id (if mutable)
```

*Offsets are confirmed for AKD1000 firmware 2.18.2; may vary for AKD1500.*

---

## From-Scratch Checklist

- [ ] Define architecture (NP count ≤ 1,000, no LSTM/attention)
- [ ] Initialize weights (random or physics-motivated)
- [ ] Train readout (ridge regression or gradient descent in Rust)
- [ ] Quantize to int4 (max-abs per layer, or per-channel for better accuracy)
- [ ] Build FlatBuffer via `ProgramBuilder` (or direct byte construction)
- [ ] Load onto chip: `device.program_external(&info, &data)?`
- [ ] Verify: `device.infer(&test_input, &config)?`
- [ ] Benchmark: `cargo run --bin bench_custom -- --model your_model.fbz`
