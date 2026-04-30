#!/usr/bin/env python3
"""Export ecoPrimals physics models to .fbz for rustChip validation.

These are the four custom models deployed on AKD1000 across springs:
  - ESN readout (hotSpring Exp 022)
  - Phase classifier (hotSpring Exp 022)
  - Transport predictor (hotSpring Exp 022)
  - Anderson classifier (groundSpring Exp 028)

Each model is reconstructed from its documented architecture using the Akida SDK,
with random int4 weights (the actual trained weights live in the spring repos).
The purpose is to validate that rustChip's parser handles these architectures
correctly — weight accuracy is not tested here.

Validation oracle only — not a rustChip runtime dependency.

Usage:
    source .zoo-venv/bin/activate
    python scripts/export_physics.py [--output baseCamp/zoo-artifacts/]
"""

import json
import os
import sys
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import akida


PHYSICS_MODELS = [
    {
        "name": "esn_readout",
        "spring": "hotSpring",
        "experiment": "Exp 022",
        "description": "Lattice QCD thermalization detector (ESN readout)",
        "input_shape": (10, 5, 1),
        "layers": [
            {"type": "InputConv", "in_channels": 1, "out_channels": 64, "kernel": 3},
            {"type": "InputConv", "in_channels": 64, "out_channels": 128, "kernel": 3},
            {"type": "FC", "in_features": 128, "out_features": 1},
        ],
    },
    {
        "name": "phase_classifier",
        "spring": "hotSpring",
        "experiment": "Exp 022",
        "description": "SU(3) confined/deconfined phase classifier",
        "input_shape": (3, 1, 1),
        "layers": [
            {"type": "InputConv", "in_channels": 3, "out_channels": 64, "kernel": 1},
            {"type": "FC", "in_features": 64, "out_features": 2},
        ],
    },
    {
        "name": "transport_predictor",
        "spring": "hotSpring",
        "experiment": "Exp 022",
        "description": "WDM transport coefficient predictor (D*, eta*, lambda*)",
        "input_shape": (6, 1, 1),
        "layers": [
            {"type": "InputConv", "in_channels": 6, "out_channels": 128, "kernel": 1},
            {"type": "FC", "in_features": 128, "out_features": 3},
        ],
    },
    {
        "name": "anderson_classifier",
        "spring": "groundSpring",
        "experiment": "Exp 028",
        "description": "Anderson localization regime classifier (loc/diff/critical)",
        "input_shape": (4, 1, 1),
        "layers": [
            {"type": "InputConv", "in_channels": 4, "out_channels": 64, "kernel": 1},
            {"type": "FC", "in_features": 64, "out_features": 3},
        ],
    },
]


def build_physics_model(spec):
    """Build a Keras model matching the physics architecture, convert to Akida .fbz."""
    import tf_keras as keras
    from quantizeml.models import quantize
    from cnn2snn import convert

    h, w, c = spec["input_shape"]
    inp = keras.layers.Input(shape=(h, w, c))
    x = inp

    for i, layer_spec in enumerate(spec["layers"]):
        if layer_spec["type"] == "InputConv":
            x = keras.layers.Conv2D(
                filters=layer_spec["out_channels"],
                kernel_size=layer_spec["kernel"],
                padding="same",
                name=f"conv_{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = keras.layers.ReLU(name=f"relu_{i}", max_value=6.0)(x)
        elif layer_spec["type"] == "FC":
            x = keras.layers.Flatten(name=f"flatten_{i}")(x)
            x = keras.layers.Dense(
                units=layer_spec["out_features"],
                name=f"dense_{i}",
            )(x)

    model = keras.Model(inputs=inp, outputs=x, name=spec["name"])

    n_samples = 100
    dummy_x = np.random.randn(n_samples, h, w, c).astype(np.float32)
    model.predict(dummy_x, verbose=0)

    from quantizeml.models import QuantizationParams
    qparams = QuantizationParams(weight_bits=8, activation_bits=8,
                                  input_weight_bits=8, per_tensor_activations=False)
    qmodel = quantize(model, qparams=qparams, samples=dummy_x, num_samples=min(50, n_samples))

    akida_model = convert(qmodel)
    return akida_model


def main():
    parser = argparse.ArgumentParser(description="Export ecoPrimals physics models")
    parser.add_argument(
        "--output", default="baseCamp/zoo-artifacts/",
        help="Output directory for .fbz files"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting {len(PHYSICS_MODELS)} physics models to {output_dir}\n")

    results = []
    for spec in PHYSICS_MODELS:
        t0 = time.time()
        try:
            akida_model = build_physics_model(spec)
            fbz_path = os.path.join(output_dir, f"{spec['name']}.fbz")
            akida_model.save(fbz_path)
            size = os.path.getsize(fbz_path)
            elapsed = round(time.time() - t0, 2)

            entry = {
                "name": spec["name"],
                "filename": f"{spec['name']}.fbz",
                "spring": spec["spring"],
                "experiment": spec["experiment"],
                "description": spec["description"],
                "input_shape": list(spec["input_shape"]),
                "file_size": size,
                "status": "ok",
                "export_time_s": elapsed,
            }

            if hasattr(akida_model, "input_shape"):
                entry["akida_input_shape"] = list(akida_model.input_shape)
            if hasattr(akida_model, "output_shape"):
                entry["akida_output_shape"] = list(akida_model.output_shape)

            results.append(entry)
            print(f"  OK  {spec['name']}.fbz ({size:,} bytes, {elapsed}s)")

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            results.append({
                "name": spec["name"],
                "status": "failed",
                "error": str(e),
                "export_time_s": elapsed,
            })
            print(f"  FAIL {spec['name']}: {e}")

    manifest_path = os.path.join(output_dir, "physics_manifest.json")
    manifest = {
        "generator": "scripts/export_physics.py",
        "akida_version": akida.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(PHYSICS_MODELS)} exported")
    print(f"Manifest: {manifest_path}")

    return 0 if ok == len(PHYSICS_MODELS) else 1


if __name__ == "__main__":
    sys.exit(main())
