# Model Conversion — Getting Any Model onto Akida in Rust

**Status:** Direct .fbz loading works today. Rust conversion tooling in progress.

---

## The Conversion Landscape

```
Source                      Path                            Status
──────────────────────────────────────────────────────────────────

Pre-compiled .fbz  ──────── Model::from_file() ──────────  ✅ Works today

Python MetaTF      ──────── .save("model.fbz") ────────────  ✅ Works today
  (requires Python SDK)     then Model::from_file()

SNNTorch weights   ──────── extract → quantize → builder ── 📋 Phase 0.2
  (.pt / .safetensors)

PyTorch float model ─────── prune → quantize → builder ──── 📋 Phase 0.3

Hand-built program ──────── ProgramBuilder → program_external() ✅ Confirmed
  (Rust weights array)

NumPy arrays (.npy) ──────── load → quantize → builder ──── 📋 Phase 0.2
  (any training framework)
```

---

## Choosing the Right Path

| Question | Answer → Path |
|----------|--------------|
| Do I have a `.fbz` file already? | Use `Model::from_file()` — done |
| Do I have a MetaTF-compiled model? | Same as above |
| Do I have PyTorch float weights and a simple architecture? | `conversion/from_pytorch.md` |
| Do I have a SNNTorch model with LIF neurons? | `conversion/from_snntorch.md` |
| Am I starting from scratch in Rust? | `conversion/from_scratch.md` |
| Do I need sub-ms custom behavior the SDK can't produce? | `conversion/from_scratch.md` |

---

## Quick Reference: Int4 Quantization

All Akida weights are quantized to int4 ([-8, 7] for weights, [0, 15] for activations).

```rust
// Max-abs quantization (used for all ecoPrimals models)
pub fn quantize_int4_per_layer(weights: &[f32]) -> Vec<i8> {
    let scale = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let inv_scale = if scale > 0.0 { 7.0 / scale } else { 0.0 };
    weights.iter().map(|&w| {
        (w * inv_scale).round().clamp(-8.0, 7.0) as i8
    }).collect()
}

// Pack 2 int4 values per byte (little-endian nibble packing)
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    values.chunks(2).map(|pair| {
        let lo = (pair[0] as u8) & 0x0F;
        let hi = if pair.len() > 1 { ((pair[1] as u8) & 0x0F) << 4 } else { 0 };
        lo | hi
    }).collect()
}
```

This is implemented in `crates/akida-models/src/quantize.rs` (Phase 0.2).
