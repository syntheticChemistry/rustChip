# Spring NPU Workload Profiles

Every ecoPrimals spring that uses neuromorphic hardware is cataloged here
with reproducible instructions for generating, parsing, and running the
model on AKD1000.

## Physics Models (validated on AKD1000)

| Spring | Model | Architecture | Input | Output | NPs | Throughput | Validated |
|--------|-------|-------------|-------|--------|-----|------------|-----------|
| hotSpring | ESN readout | InputConv(50→128)+FC(128→1) | 50-step plaquette history | thermalization score | 179 | 18,500 Hz | Exp 022 (5,978 live calls) |
| hotSpring | Phase classifier | InputConv(3→64)+FC(64→2) | plaquette+Re(L)+Im(L) | confined/deconfined | 67 | 21,200 Hz | Exp 022 (100% accuracy) |
| hotSpring | Transport predictor | InputConv(6→128)+FC(128→3) | 6 plasma observables | D\*/η\*/λ\* | 134 | 17,800 Hz | Exp 022 |
| groundSpring | Anderson classifier | InputConv(4→64)+FC(64→3) | 4 spectral observables | loc/diff/critical | 68 | 22,400 Hz | Exp 028 (W_c=16.26±0.95) |

## Edge/Streaming Models (planned)

| Spring | Model | Architecture | Input | Output | Status |
|--------|-------|-------------|-------|--------|--------|
| airSpring | Crop stress ESN | ESN+readout | sensor features | crop class | Documented (Exp 028-029) |
| wetSpring | HAB sentinel | ESN+readout | microbe features | bloom risk | Documented (Sub-thesis 04) |

## Reproducing Physics Models

The four physics models above have fully documented architectures and can be
reconstructed using `ProgramBuilder`:

```bash
# Generate physics model .fbz artifacts
source .zoo-venv/bin/activate
python scripts/export_physics.py --output baseCamp/zoo-artifacts/

# Parse with Rust
cargo run --bin akida -- parse baseCamp/zoo-artifacts/esn_readout.fbz
cargo run --bin akida -- parse baseCamp/zoo-artifacts/phase_classifier.fbz
cargo run --bin akida -- parse baseCamp/zoo-artifacts/transport_predictor.fbz
cargo run --bin akida -- parse baseCamp/zoo-artifacts/anderson_classifier.fbz
```

These models stand independent of the spring repos — you only need rustChip
and the export script.

## How These Connect to the BrainChip Model Zoo

The BrainChip zoo models (21 exported via `scripts/export_zoo.py`) validate
that rustChip's parser handles the full range of Akida architectures:
AkidaNet, MobileNet, YOLO, PointNet++, TENN, CenterNet, UNet, DVS, etc.

The spring models are custom architectures deployed on the same hardware,
demonstrating that rustChip handles both standard and bespoke models.

Together they prove: **every model we claim to run, we can parse, inspect,
and profile**.

## Model Documentation

Detailed per-model documentation lives in `baseCamp/models/`:

- `physics/esn_readout.md` — thermalization detector
- `physics/phase_classifier.md` — SU(3) phase boundary
- `physics/transport_predictor.md` — WDM transport coefficients
- `physics/anderson_classifier.md` — Anderson localization regime
- `edge/ds_cnn_kws.md` — keyword spotting (BrainChip reference)
- `edge/akidanet.md` — ImageNet classification
- `edge/dvs_gesture.md` — DVS gesture recognition

## Relationship to the Sovereign Compute Trio

```
barraCuda (GPU math)  →  &[f32]  →  rustChip (NPU inference)  →  &[f32]  →  application
       ↑                                    ↑
   coralReef (VFIO)                   toadStool (dispatch)
```

The springs use the full pipeline. rustChip handles the NPU segment.
See `specs/INTEGRATION_GUIDE.md` for the integration architecture.
