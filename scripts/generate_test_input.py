#!/usr/bin/env python3
"""
Generate C++ test input files from numpy arrays

This script helps validate your C++ inference pipeline by:
1. Loading test input data from a numpy .npy file
2. Generating C++ header/source files (test_input.h/test_input.cpp)
3. Optionally running Python inference and saving expected output for comparison

The generated C++ files can be included in your inference application to
verify that the C++ pipeline produces the same results as Python.

Usage:
    python generate_test_input.py --input-file input.npy --input-shape 32,32,3 --input-type uint8

    # With expected output validation
    python generate_test_input.py --input-file input.npy --input-shape 32,32,3 \
                                   --input-type uint8 --model-path model.fbz

Requirements:
    - numpy
    - akida (if using --model-path for validation)
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def check_dependencies(need_akida=False):
    """Check that required packages are installed"""
    try:
        import numpy
    except ImportError:
        print("ERROR: numpy package not found", file=sys.stderr)
        print("\nInstall it with:", file=sys.stderr)
        print("  pip install numpy", file=sys.stderr)
        sys.exit(1)

    if need_akida:
        try:
            import akida
            from akida.core.array_to_cpp import array_to_cpp
        except ImportError:
            print("ERROR: akida package not found (required for --model-path)", file=sys.stderr)
            print("\nInstall it with:", file=sys.stderr)
            print("  pip install akida", file=sys.stderr)
            sys.exit(1)
        return numpy, akida, array_to_cpp

    try:
        from akida.core.array_to_cpp import array_to_cpp
    except ImportError:
        # If akida is not available, we need to implement array_to_cpp ourselves
        def array_to_cpp_fallback(output_dir, data, name):
            """Simple implementation of array_to_cpp for when akida is not available"""
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Convert to uint8 array
            if data.dtype != np.uint8:
                print(f"WARNING: Converting data from {data.dtype} to uint8", file=sys.stderr)
                data = data.astype(np.uint8)

            flat_data = data.flatten()

            # Generate header file
            header_file = output_path / f"{name}.h"
            with open(header_file, 'w') as f:
                f.write(f"#ifndef {name.upper()}_H\n")
                f.write(f"#define {name.upper()}_H\n\n")
                f.write("#include <cstdint>\n")
                f.write("#include <cstddef>\n\n")
                f.write(f"extern const uint8_t {name}[];\n")
                f.write(f"extern const size_t {name}_len;\n\n")
                f.write("#endif\n")

            # Generate source file
            source_file = output_path / f"{name}.cpp"
            with open(source_file, 'w') as f:
                f.write(f'#include "{name}.h"\n\n')
                f.write(f"const uint8_t {name}[] = {{\n")

                # Write data in rows of 12 bytes
                for i in range(0, len(flat_data), 12):
                    row = flat_data[i:i+12]
                    f.write("    " + ", ".join(f"0x{b:02x}" for b in row) + ",\n")

                f.write("};\n\n")
                f.write(f"const size_t {name}_len = sizeof({name});\n")

            print(f"✓ Created: {header_file}")
            print(f"✓ Created: {source_file}")

        array_to_cpp = array_to_cpp_fallback

    return numpy, None, array_to_cpp


def load_input_data(input_file, input_shape, input_type):
    """Load and validate input data from numpy file"""
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading input data: {input_path}")

    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"ERROR: Failed to load numpy file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse shape
    try:
        target_shape = tuple(int(x) for x in input_shape.split(','))
    except Exception as e:
        print(f"ERROR: Invalid shape format: {input_shape}", file=sys.stderr)
        print("Expected format: height,width,channels (e.g., 32,32,3)", file=sys.stderr)
        sys.exit(1)

    # Reshape if needed
    if data.shape != target_shape:
        print(f"Reshaping data from {data.shape} to {target_shape}")
        try:
            data = data.reshape(target_shape)
        except Exception as e:
            print(f"ERROR: Cannot reshape data: {e}", file=sys.stderr)
            sys.exit(1)

    # Convert type if needed
    dtype_map = {
        'uint8': np.uint8,
        'int8': np.int8,
        'float32': np.float32,
        'int32': np.int32
    }

    if input_type not in dtype_map:
        print(f"ERROR: Unsupported input type: {input_type}", file=sys.stderr)
        print(f"Supported types: {list(dtype_map.keys())}", file=sys.stderr)
        sys.exit(1)

    target_dtype = dtype_map[input_type]
    if data.dtype != target_dtype:
        print(f"Converting data from {data.dtype} to {target_dtype}")
        data = data.astype(target_dtype)

    print(f"Input data shape: {data.shape}")
    print(f"Input data type: {data.dtype}")
    print(f"Input data range: [{data.min()}, {data.max()}]")

    return data


def run_python_inference(model_path, input_data, output_dir, akida_module):
    """Run inference in Python and save expected output"""
    print("\nRunning Python inference for validation...")

    try:
        model = akida_module.Model(model_path)
        devices = akida_module.devices()

        if not devices:
            print("WARNING: No Akida device found, skipping validation", file=sys.stderr)
            return None

        device = devices[0]
        model.map(device)

        # Add batch dimension
        input_batch = np.expand_dims(input_data, axis=0)

        # Run inference
        output = model.predict(input_batch)

        # Remove batch dimension
        output = output.squeeze(0)

        # Save expected output
        output_path = Path(output_dir)
        output_file = output_path / "expected_output.npy"
        np.save(output_file, output)

        print(f"✓ Saved expected output: {output_file}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output type: {output.dtype}")

        # Print first few values
        flat_output = output.flatten()
        print(f"  First values: {flat_output[:5]}")

        return output

    except Exception as e:
        print(f"WARNING: Failed to run Python inference: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate C++ test input files from numpy arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate test input files
  python generate_test_input.py --input-file input.npy --input-shape 32,32,3 --input-type uint8

  # With validation against Python inference
  python generate_test_input.py --input-file input.npy --input-shape 32,32,3 \
                                 --input-type uint8 --model-path model.fbz --output-dir ../src/
        """
    )

    parser.add_argument(
        '--input-file',
        required=True,
        help='Path to numpy .npy file containing input data'
    )

    parser.add_argument(
        '--input-shape',
        required=True,
        help='Input shape as comma-separated dimensions (e.g., 32,32,3)'
    )

    parser.add_argument(
        '--input-type',
        required=True,
        choices=['uint8', 'int8', 'float32', 'int32'],
        help='Input data type'
    )

    parser.add_argument(
        '--model-path',
        help='Optional: Path to .fbz model for validation (runs Python inference)'
    )

    parser.add_argument(
        '--output-dir',
        default='../src/',
        help='Directory for generated C++ files (default: ../src/)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Akida Test Input Generator")
    print("="*60)

    # Check dependencies
    np, akida, array_to_cpp = check_dependencies(need_akida=args.model_path is not None)

    # Load input data
    input_data = load_input_data(args.input_file, args.input_shape, args.input_type)

    # Generate C++ files
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating C++ files in: {output_path}")
    array_to_cpp(str(output_path), input_data, 'test_input')

    # Optional: Run Python inference for validation
    if args.model_path and akida:
        run_python_inference(args.model_path, input_data, args.output_dir, akida)

    print("\n" + "="*60)
    print("Generation Complete")
    print("="*60)
    print("\nNext steps:")
    print("1. Include test_input.h in your C++ code")
    print("2. Use test_input[] and test_input_len for inference input")
    if args.model_path:
        print("3. Compare C++ output with expected_output.npy")
    print("="*60)


if __name__ == '__main__':
    main()
