#!/usr/bin/env python3
"""
Convert Akida .fbz model to C++ binary files (program.h/program.cpp)

This script automates the model conversion pipeline:
1. Load trained .fbz model
2. Discover Akida hardware
3. Map model to hardware
4. Generate C++ header and source files

The generated files can be compiled directly into your C++ application.

Usage:
    python convert_model.py --model-path /path/to/model.fbz [--output-dir ../src/]

Requirements:
    - akida Python package installed
    - Akida hardware connected and visible via 'akida devices'
    - Model file must be a valid .fbz exported from Akida training
"""

import argparse
import sys
from pathlib import Path


def check_dependencies():
    """Check that required Python packages are installed"""
    try:
        import akida
    except ImportError:
        print("ERROR: akida package not found", file=sys.stderr)
        print("\nInstall it with:", file=sys.stderr)
        print("  pip install akida", file=sys.stderr)
        sys.exit(1)

    try:
        from akida.core.array_to_cpp import array_to_cpp
    except ImportError:
        print("ERROR: Could not import array_to_cpp from akida.core", file=sys.stderr)
        print("\nYou may need to update the akida package:", file=sys.stderr)
        print("  pip install --upgrade akida", file=sys.stderr)
        sys.exit(1)

    return akida, array_to_cpp


def discover_device(akida_module):
    """Find an available Akida device for model mapping"""
    devices = akida_module.devices()

    if not devices:
        print("\nERROR: No Akida devices found!", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("1. Check that Akida hardware is connected via PCIe", file=sys.stderr)
        print("2. Verify PCIe driver is loaded:", file=sys.stderr)
        print("   $ lsmod | grep akida", file=sys.stderr)
        print("3. Check device visibility:", file=sys.stderr)
        print("   $ akida devices", file=sys.stderr)
        print("4. Ensure you have permissions to access /dev/akida*", file=sys.stderr)
        sys.exit(1)

    # Use the first available device
    device = devices[0]
    print(f"Using device: {device.desc}")
    return device


def load_model(model_path, akida_module):
    """Load and validate the .fbz model file"""
    model_file = Path(model_path)

    if not model_file.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not model_file.suffix == '.fbz':
        print(f"WARNING: Expected .fbz file, got: {model_file.suffix}", file=sys.stderr)

    print(f"Loading model: {model_file}")

    try:
        model = akida_module.Model(str(model_file))
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    return model


def map_model(model, device):
    """Map the model to Akida hardware"""
    print("Mapping model to hardware...")

    try:
        # Map with hw_only=True to generate the hardware program
        model.map(device, hw_only=True)
    except Exception as e:
        print(f"ERROR: Failed to map model: {e}", file=sys.stderr)
        print("\nPossible causes:", file=sys.stderr)
        print("- Model is incompatible with the connected device", file=sys.stderr)
        print("- Model requires features not available in hardware", file=sys.stderr)
        sys.exit(1)

    print("Model mapped successfully")
    return model


def generate_cpp_files(model, output_dir, array_to_cpp_func):
    """Generate program.h and program.cpp from the model program"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating C++ files in: {output_path}")

    try:
        # Extract the program binary from the first sequence
        # (most models have a single sequence)
        program = model.sequences[0].program

        # Generate C++ files using the akida utility
        # This creates:
        # - program.h: Declares 'extern const uint8_t program[]' and 'extern const size_t program_len'
        # - program.cpp: Defines the program array with the binary data
        array_to_cpp_func(str(output_path), program, 'program')

    except Exception as e:
        print(f"ERROR: Failed to generate C++ files: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify files were created
    header_file = output_path / "program.h"
    source_file = output_path / "program.cpp"

    if not header_file.exists() or not source_file.exists():
        print("ERROR: C++ files were not created", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Created: {header_file}")
    print(f"✓ Created: {source_file}")
    print(f"\nProgram size: {len(program)} bytes")

    return header_file, source_file


def print_summary(model, model_path):
    """Print a summary of the conversion"""
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Model file: {model_path}")
    print(f"Model name: {model.name if hasattr(model, 'name') else 'N/A'}")

    # Try to get input/output shapes if available
    try:
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
    except:
        pass

    print("\nNext steps:")
    print("1. Verify program.h and program.cpp were created in the output directory")
    print("2. Build the C++ inference application:")
    print("   $ mkdir build && cmake src/ -B build && make -C build")
    print("3. Run inference:")
    print("   $ ./build/akida_inference")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Akida .fbz model to C++ binary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert model and place files in ../src/
  python convert_model.py --model-path my_model.fbz

  # Specify custom output directory
  python convert_model.py --model-path my_model.fbz --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--model-path',
        required=True,
        help='Path to the .fbz model file'
    )

    parser.add_argument(
        '--output-dir',
        default='../src/',
        help='Directory for generated C++ files (default: ../src/)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Akida Model to C++ Converter")
    print("="*60)

    # Check dependencies
    akida, array_to_cpp = check_dependencies()

    # Discover Akida device
    device = discover_device(akida)

    # Load model
    model = load_model(args.model_path, akida)

    # Map model to hardware
    model_mapped = map_model(model, device)

    # Generate C++ files
    generate_cpp_files(model_mapped, args.output_dir, array_to_cpp)

    # Print summary
    print_summary(model_mapped, args.model_path)


if __name__ == '__main__':
    main()
