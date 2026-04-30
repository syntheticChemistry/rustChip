# rustChip Quickstart

Clone to first model parse in under a minute. No Python, no SDK, no hardware needed.

## 1. Clone and build

```bash
git clone https://github.com/syntheticChemistry/rustChip.git
cd rustChip
cargo build --workspace
```

## 2. Generate a model (pure Rust — no Python)

```bash
cargo run -p akida-cli -- convert \
  --weights "random:6400" \
  --arch "InputConv(50,1,1) FC(128) FC(1)" \
  --output my_first_model.fbz \
  --bits 4
```

## 3. Parse and inspect it

```bash
cargo run -p akida-cli -- parse my_first_model.fbz
```

Output:

```
Akida Model: my_first_model.fbz
============================================================
File size         : 3647 bytes (3.56 KB)
Decompressed      : 3864 bytes (3.77 KB)
SDK version       : 2.19.1
Layers            : 3
============================================================
```

## 4. Use in Rust code

```rust
use akida_models::prelude::*;

let model = Model::from_file("my_first_model.fbz")?;
println!("version: {}", model.version());
println!("layers: {}", model.layer_count());
```

## 5. Run the validation suite

```bash
cargo test --workspace
```

## What you have now

- A pure Rust FBZ model parser that handles any Akida model
- A pure Rust model conversion pipeline (import, quantize, serialize, compress)
- Weight import from `.npy` and `.safetensors` (no Python at runtime)
- Int1/2/4/8 quantization with nibble packing
- CLI tools for model inspection and conversion

## Next steps

| Goal | Start here |
|------|-----------|
| Browse the full model zoo | [baseCamp/ZOO_GUIDE.md](baseCamp/ZOO_GUIDE.md) |
| Apply NPU to your science domain | [baseCamp/preserve/README.md](baseCamp/preserve/README.md) |
| Deploy for scientific workloads | [baseCamp/SCIENTIFIC_DEPLOYMENT.md](baseCamp/SCIENTIFIC_DEPLOYMENT.md) |
| Understand the ecosystem | [LEVERAGE.md](LEVERAGE.md) |
| Run on real Akida hardware | [specs/INTEGRATION_GUIDE.md](specs/INTEGRATION_GUIDE.md) |
| Verify reproducibility | `cargo run -p akida-cli -- guidestone` |
