# Minimal FC — Hand-Built via program_external()

**Architecture:** FC(50→1)
**Status:** ✅ CONFIRMED — program_external() validated (BEYOND_SDK Discovery 3)
**Task:** Smoke test; demonstrates direct FlatBuffer program injection
**Source:** rustChip metalForge Exp 001 + BEYOND_SDK.md Discovery 3

---

## What this proves

Discovery 3 from `BEYOND_SDK.md`:

> program_external() works with hand-crafted FlatBuffer binaries.
> The chip does not validate program authenticity — any correctly-formatted
> FlatBuffer binary is accepted and executed.

This is the most important capability in rustChip: you don't need the MetaTF
Python SDK to put programs on the chip. If you can build a valid FlatBuffer
with the right layout, you can run it.

---

## The Minimal Program

The smallest valid Akida program:

```
Input: float[50]  (arbitrary 50-dimensional input)
  │
  ▼
FC(50→1)          ← 50 weight values, 1 bias, 1 threshold
  │
  ▼
Output: float[1]
```

This is 51 int4 weights (50 input weights + 1 bias) = 26 bytes packed.
With FlatBuffer overhead: ~2 KB total program binary.

---

## Building It

```rust
// akida_models::builder (planned 0.2) — sketch of what already works
// in the metalForge hand-craft experiments

use akida_models::flatbuf::ProgramInfoBuilder;

// 1. Define layer structure
let layer = LayerSpec {
    layer_type: LayerType::FullyConnected,
    in_features: 50,
    out_features: 1,
    weight_bits: 4,
    threshold: 1.0,
};

// 2. Build program_info (FlatBuffer header + layer table)
let program_info = ProgramInfoBuilder::new()
    .version("2.18.2")  // must match chip firmware expectation
    .layer(layer)
    .build()?;

// 3. Build program_data (quantized weights)
let weights_int4 = quantize_int4(&trained_weights);  // [50] weights
let bias_int4    = quantize_int4(&[trained_bias]);    // [1] bias
let program_data = pack_program_data(&weights_int4, &bias_int4)?;

// 4. Inject
device.write(&program_info)?;
device.dma_write_program(&program_data)?;
device.set_program()?;  // triggers program_external()
```

---

## Reverse Engineering the Format

The FlatBuffer layout was reverse-engineered from:
1. Akida Python SDK — intercepted `program_info` bytes during normal SDK use
2. Hex dump analysis — `metalForge/npu/akida/REGISTER_PROBE_LOG.md`
3. `crates/akida-chip/src/program.rs` — the confirmed Rust model

Key findings:
- Magic bytes: `08 00 00 00` at offset 0 (FlatBuffer standard)
- Version string at offset 8: null-terminated UTF-8
- Layer table at variable offset (table pointer at offset 4)
- Weight layout: int4 packed 2-per-byte, row-major order
- Threshold encoding: int4, same packing as weights

---

## Why This Matters for the Ecosystem

Every model in `baseCamp/models/` ultimately reduces to this pattern:
a sequence of FlatBuffer tables describing layers and weights.

`program_external()` means:
1. **Independence from MetaTF**: no Python SDK, no conda env, no CUDA dependency
2. **Architecture freedom**: any architecture that maps to Akida NPs can be expressed
3. **Dynamic adaptation**: weights can be swapped via `set_variable()` without rebuilding the program binary
4. **Rust-only pipeline**: train in Rust (neuralSpring primitives) → quantize in Rust → inject via `program_external()`

The minimal FC model is the proof-of-concept. The physics models above
are the production examples.
