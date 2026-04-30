#!/usr/bin/env python3
"""Export BrainChip Akida model zoo to .fbz files with ground-truth manifest.

Validation oracle only — this script is NOT a rustChip runtime dependency.
It uses the BrainChip Python SDK to generate ground-truth .fbz artifacts
and metadata so the Rust parser can be regression-tested against every
model BrainChip publishes.

Usage:
    source .zoo-venv/bin/activate
    python scripts/export_zoo.py [--output baseCamp/zoo-artifacts/]
"""

import json
import os
import sys
import time
import traceback
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import akida
import akida_models
from cnn2snn import convert


def get_pretrained_models():
    """Discover all *_pretrained() functions in akida_models."""
    names = [n for n in dir(akida_models) if n.endswith("_pretrained")]
    return sorted(names)


def export_model(func_name, output_dir):
    """Export a single pretrained model to .fbz and return metadata."""
    entry = {"function": func_name, "status": "failed"}
    t0 = time.time()

    try:
        fn = getattr(akida_models, func_name)
        result = fn()

        # Some models (YOLO) return (model, anchors) tuples
        if isinstance(result, tuple):
            keras_model = result[0]
        else:
            keras_model = result

        model_name = func_name.replace("_pretrained", "")
        entry["name"] = model_name

        if hasattr(keras_model, "input_shape"):
            entry["keras_input_shape"] = list(keras_model.input_shape)
        if hasattr(keras_model, "output_shape"):
            entry["keras_output_shape"] = list(keras_model.output_shape)

        akida_model = convert(keras_model)
        entry["akida_version"] = akida.__version__

        fbz_path = os.path.join(output_dir, f"{model_name}.fbz")
        akida_model.save(fbz_path)
        entry["filename"] = f"{model_name}.fbz"
        entry["file_size"] = os.path.getsize(fbz_path)

        summary = akida_model.summary
        if hasattr(summary, "layers"):
            entry["layer_count"] = len(summary.layers)
            entry["layers"] = []
            for layer in summary.layers:
                layer_info = {"name": str(layer.name)}
                if hasattr(layer, "output_shape"):
                    layer_info["output_shape"] = list(layer.output_shape)
                entry["layers"].append(layer_info)

        if hasattr(akida_model, "input_shape"):
            entry["input_shape"] = list(akida_model.input_shape)
        if hasattr(akida_model, "output_shape"):
            entry["output_shape"] = list(akida_model.output_shape)

        entry["status"] = "ok"
        entry["export_time_s"] = round(time.time() - t0, 2)
        print(f"  OK  {model_name}.fbz ({entry['file_size']:,} bytes, "
              f"{entry.get('layer_count', '?')} layers, {entry['export_time_s']}s)")

    except Exception as e:
        entry["error"] = str(e)
        entry["traceback"] = traceback.format_exc()
        entry["export_time_s"] = round(time.time() - t0, 2)
        print(f"  FAIL {func_name}: {e}")

    return entry


def main():
    parser = argparse.ArgumentParser(description="Export Akida model zoo to .fbz")
    parser.add_argument(
        "--output", default="baseCamp/zoo-artifacts/",
        help="Output directory for .fbz files and manifest"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    pretrained = get_pretrained_models()
    print(f"Exporting {len(pretrained)} pretrained models to {output_dir}\n")

    manifest = {
        "generator": "scripts/export_zoo.py",
        "akida_version": akida.__version__,
        "akida_models_version": akida_models.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": [],
    }

    ok = 0
    fail = 0
    for func_name in pretrained:
        entry = export_model(func_name, output_dir)
        manifest["models"].append(entry)
        if entry["status"] == "ok":
            ok += 1
        else:
            fail += 1

    manifest["summary"] = {"total": len(pretrained), "ok": ok, "failed": fail}

    manifest_path = os.path.join(output_dir, "zoo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nDone: {ok} ok, {fail} failed")
    print(f"Manifest: {manifest_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
