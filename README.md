# Akida C++ Inference Deployment Package

## Overview

This package provides a **complete, working example** of deploying a trained Akida neural network model as a standalone C++ application on a Linux host. It bridges the gap between Python-based model training and production C++ deployment without requiring Python at runtime.

### What's Included

- **Complete C++ inference pipeline** (`src/inference.cpp`) with detailed comments explaining each API call
- **Linux system implementation** (`src/system_linux.cpp`) for required runtime hooks
- **Model conversion script** (`scripts/convert_model.py`) to transform `.fbz` models to C++ binaries
- **Test input generator** (`scripts/generate_test_input.py`) for validation against Python results
- **Ready-to-build CMake configuration** (`src/CMakeLists.txt`)

### Target Use Case

You have:
- ✅ A trained Keras model converted to Akida format (`.fbz` file)
- ✅ Validated the model works in Python using the Akida SDK
- ✅ An AKD1000 or AKD1500 device connected to a Linux host via PCIe

You want:
- 🎯 A standalone C++ application that runs inference **without Python**
- 🎯 Production-ready code you can integrate into your existing C++ codebase
- 🎯 Full control over the inference pipeline (batching, latency measurement, pre/post-processing)

## Quick Start

### Prerequisites

**Hardware:**
- AKD1000 or AKD1500 connected via PCIe to a Linux host (Ubuntu/Debian recommended)
- PCIe driver installed and device visible

**Software:**
- CMake 3.16 or later
- C++17 compatible compiler (GCC 7+, Clang 5+)
- Python 3.7+ with `akida` package (only needed for model conversion, not runtime)
- Trained Akida model file (`.fbz` format)

**Verification Steps:**

```bash
# 1. Check PCIe driver is loaded
lsmod | grep akida
# Should show: akida_dw_edma

# 2. Verify device is visible
akida devices
# Should list at least one device

# 3. Check permissions
ls -l /dev/akida*
# You should have read/write access (or run as root)
```

### Step 1: Deploy the Akida Engine

The Akida Engine is the C++ library that interfaces with the hardware. Deploy it using the Python SDK:

```bash
# Create a working directory
mkdir akida_deployment && cd akida_deployment

# Deploy the engine with host examples
akida engine deploy --dest-path . --with-host-examples
```

This creates an `engine/` directory containing:
- `api/` - C++ API headers
- `host/` - Host-specific drivers (PCIe)
- `infra/` - Infrastructure interfaces
- `cmake/` - Build system modules
- `test/` - Example programs
- `libakida.a` - Static library

**Verify the deployment:**

```bash
# Build and run the included example
cd engine/test/simple_conv_v2
mkdir build
cmake . -B build
make -C build
./build/simple_conv_v2

# Should output: "Success!"
cd ../../..  # Return to akida_deployment/
```

### Step 2: Place This Package

Copy this `customer_deployment_package/` directory into `engine/test/`:

```bash
# Assuming you received this package as a zip file
unzip customer_deployment_package.zip
cp -r customer_deployment_package engine/test/
```

**Why this location?** The CMake configuration uses relative paths to find the engine's build modules. Placing the package at `engine/test/customer_deployment_package/` ensures these paths resolve correctly.

Your directory structure should now look like:

```
akida_deployment/
└── engine/
    ├── api/
    ├── host/
    ├── infra/
    ├── cmake/
    ├── libakida.a
    └── test/
        ├── simple_conv_v2/
        └── customer_deployment_package/  ← This package
            ├── README.md
            ├── src/
            └── scripts/
```

### Step 3: Convert Your Model

Use the provided Python script to convert your trained `.fbz` model to C++ source files:

```bash
cd engine/test/customer_deployment_package/scripts

# Convert your model
python convert_model.py --model-path /path/to/your/model.fbz

# This generates:
#   ../src/program.h       - Header declaring the model binary
#   ../src/program.cpp     - Source containing the model data
```

**What happens during conversion:**
1. Loads your `.fbz` model using the Akida Python SDK
2. Discovers the connected Akida device
3. Maps the model to hardware (generates the binary program)
4. Exports the program as C++ arrays (`program[]` and `program_len`)

**Expected output:**

```
===========================================================
Akida Model to C++ Converter
===========================================================
Using device: NSoC/2.0
Loading model: /path/to/model.fbz
Mapping model to hardware...
Model mapped successfully
Generating C++ files in: ../src/
✓ Created: ../src/program.h
✓ Created: ../src/program.cpp

Program size: 524288 bytes

===========================================================
Conversion Summary
===========================================================
Model file: /path/to/model.fbz
Input shape: (32, 32, 3)
Output shape: (1, 1, 10)
...
```

### Step 4: Build the Inference Application

```bash
cd ..  # Return to customer_deployment_package/

# Create build directory and compile
mkdir build
cmake src/ -B build
make -C build
```

**Expected output:**

```
-- Configuring Akida Inference Demo
--   Executable: akida_inference
--   Engine root: /path/to/engine
--   CMake module path: /path/to/engine/cmake
...
[ 25%] Building CXX object CMakeFiles/akida_inference.dir/main.cpp.o
[ 50%] Building CXX object CMakeFiles/akida_inference.dir/inference.cpp.o
[ 75%] Building CXX object CMakeFiles/akida_inference.dir/system_linux.cpp.o
[100%] Building CXX object CMakeFiles/akida_inference.dir/program.cpp.o
[100%] Linking CXX executable akida_inference
```

### Step 5: Run Inference

```bash
./build/akida_inference
```

**Expected output:**

```
======================================
  Akida C++ Inference Demo
======================================

Discovering Akida devices...
Found 1 Akida device(s)

Using device: NSoC/2.0

Starting inference pipeline...

=== Step 1: Creating Hardware Device ===
Device created successfully
Device description: NSoC/2.0

=== Step 2: Programming Device ===
Program size: 524288 bytes
Device programmed successfully
Input shape: [32, 32, 3]
Output shape: [1, 1, 10]
Input format: Dense
Activation enabled: Yes

=== Step 3: Setting Batch Size ===
Batch size set to 1

=== Step 4: Preparing Input Tensor ===
Input data prepared: 3072 elements

=== Step 5: Convert to Sparse format if required ===
Model accepts dense input - no conversion needed

=== Step 6: Enqueueing Input ===
Input enqueued successfully

=== Step 7: Fetching Results ===
Results fetched successfully

=== Step 8: Dequantizing Output ===
Output dequantized to float

=== Step 9: Results ===
Output shape: [1, 1, 10]
Output dtype: float32
First 10 output values:
  [0]: 0.123456
  [1]: 0.234567
  ...
  [9]: 0.987654

Predicted class: 7 (score: 0.987654)

=== Inference Complete ===

======================================
  Inference completed successfully!
======================================
```

## Understanding the Code

### File Overview

| File | Purpose |
|------|---------|
| `main.cpp` | Entry point - discovers Akida devices via PCIe and calls `run_inference()` |
| `inference.h` | Header declaring the `run_inference()` function |
| `inference.cpp` | **Core file** - implements the complete inference pipeline with detailed comments |
| `system_linux.cpp` | Linux implementation of required runtime hooks (`msleep`, `time_ms`, `panic`, `kick_watchdog`) |
| `program.h` | Generated by `convert_model.py` - declares the model binary |
| `program.cpp` | Generated by `convert_model.py` - contains the model binary data |
| `CMakeLists.txt` | Build configuration |

### The Inference Pipeline (inference.cpp)

The core pipeline follows these steps:

#### 1. Create Hardware Device
```cpp
std::unique_ptr<akida::HardwareDevice> device = akida::HardwareDevice::create(driver);
```
Wraps the low-level driver with the high-level API.

#### 2. Program the Device
```cpp
akida::ProgramInfo program_info = device->program(program, program_len);
```
Loads the neural network model onto the chip. The `ProgramInfo` returned contains critical metadata about input/output shapes, data types, and memory requirements.

#### 3. Set Batch Size
```cpp
device->set_batch_size(1, false);
```
Configures how many samples are processed per inference call. Most deployment scenarios use batch size 1.

#### 4. Prepare Input Tensor
```cpp
auto dense_input = akida::Dense::create_view(
    input_data.data(),
    akida::Shape{height, width, channels},
    akida::DType::uint8
);
```
Wraps your input data in an `akida::Tensor`. Note that `create_view()` does **not** copy data - it creates a lightweight wrapper.

#### 5. Convert to Sparse (if needed)
```cpp
if (!program_info.input_is_dense()) {
    auto sparse_input = akida::conversion::to_sparse(*dense_input);
    input_tensor = std::move(sparse_input);
}
```
Some models require sparse input format. Check `program_info.input_is_dense()` and convert if necessary.

#### 6. Enqueue Input
```cpp
device->enqueue(*input_tensor);
```
Sends the input to the device and triggers inference. This is non-blocking.

#### 7. Fetch Results
```cpp
std::unique_ptr<akida::Tensor> output = device->fetch();
```
Blocks until inference completes and returns the output tensor.

#### 8. Dequantize (if needed)
```cpp
if (program_info.activation_enabled()) {
    final_output = device->dequantize(*dense_output);
}
```
If the model uses quantized activations, the output is in `int32` format and must be dequantized to `float`.

#### 9. Extract Results
```cpp
const float* data = final_output->data<float>();
```
Access the raw output data as a typed pointer.

### System Requirements (system_linux.cpp)

The Akida Engine requires the runtime to implement 4 functions declared in `infra/system.h`:

- **`msleep(uint32_t ms)`** - Sleep for milliseconds (uses POSIX `usleep`)
- **`time_ms()`** - Get current time in milliseconds (uses `clock_gettime`)
- **`kick_watchdog()`** - Service hardware watchdog (no-op on Linux host)
- **`panic(const char* fmt, ...)`** - Fatal error handler (prints to stderr and aborts)

These implementations work for any Linux host. For embedded targets (RTOS), you would implement these using your platform's APIs.

### Device Discovery (main.cpp)

```cpp
auto drivers = akida::get_drivers();
```

This function scans for Akida devices connected via PCIe. It returns a vector of `HardwareDriver` smart pointers.

**Note:** `akida::get_drivers()` is specific to the **host PCIe deployment**. For embedded systems, you would instead create chip-specific drivers:
- `akida::BareMetalDriver` for AKD1000
- `akida::Akd1500SpiDriver` for AKD1500 over SPI

## Optional: Generate Test Input for Validation

To validate that your C++ pipeline produces the same results as Python, use the test input generator:

```bash
cd scripts/

# Generate C++ input files from a numpy array
python generate_test_input.py \
    --input-file /path/to/test_input.npy \
    --input-shape 32,32,3 \
    --input-type uint8 \
    --model-path /path/to/model.fbz \
    --output-dir ../src/
```

This creates:
- `test_input.h` / `test_input.cpp` - C++ arrays with your test data
- `expected_output.npy` - Python inference results for comparison

**Modify `inference.cpp` to use the test input:**

```cpp
#include "test_input.h"  // Add this

// In run_inference(), replace the test pattern with:
std::vector<uint8_t> input_data(test_input, test_input + test_input_len);
```

Rebuild and run. Compare the C++ output values with `expected_output.npy`.

**Expected differences:**
- Small floating-point differences (< 1e-5) are normal due to rounding
- Larger differences may indicate issues with input preprocessing or format conversion

## Adapting for Your Application

### 1. Replace Test Input with Real Data

The current `inference.cpp` uses a simple test pattern. Replace it with your actual data source:

**Example: Load from image file (requires OpenCV)**

```cpp
#include <opencv2/opencv.hpp>

// In run_inference():
cv::Mat img = cv::imread("input.jpg");
cv::resize(img, img, cv::Size(32, 32));
cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

std::vector<uint8_t> input_data(img.data, img.data + (32 * 32 * 3));
```

**Example: Real-time sensor data**

```cpp
// Read from sensor API
SensorReading reading = sensor.read();
std::vector<uint8_t> input_data(reading.data, reading.data + reading.size);
```

### 2. Add Pre-Processing

Common pre-processing steps for image models:

```cpp
// Normalize to [0, 1]
std::vector<float> normalized(input_size);
for (int i = 0; i < input_size; i++) {
    normalized[i] = input_data[i] / 255.0f;
}

// Create tensor with float data
auto dense_input = akida::Dense::create_view(
    normalized.data(),
    shape,
    akida::DType::float32
);
```

### 3. Add Post-Processing

**Softmax (convert logits to probabilities):**

```cpp
std::vector<float> probabilities(output_size);
float sum = 0.0f;

for (int i = 0; i < output_size; i++) {
    probabilities[i] = std::exp(data[i]);
    sum += probabilities[i];
}

for (int i = 0; i < output_size; i++) {
    probabilities[i] /= sum;
}
```

**Threshold-based decisions:**

```cpp
const float CONFIDENCE_THRESHOLD = 0.8f;

if (probabilities[predicted_class] > CONFIDENCE_THRESHOLD) {
    printf("Confident prediction: class %d\n", predicted_class);
} else {
    printf("Low confidence - rejecting prediction\n");
}
```

### 4. Use the Higher-Level predict() API

For simpler use cases where you don't need fine-grained control, use `predict()`:

```cpp
// Combines enqueue() + fetch() in one call
auto output = device->predict(*input_tensor);
if (!output) {
    // Handle error
}
```

This is more convenient but gives less control (e.g., you can't measure enqueue/fetch separately).

### 5. Measure Inference Latency

```cpp
// Enable hardware clock counter
device->toggle_clock_counter(true);

// Run inference
device->enqueue(*input_tensor);
auto output = device->fetch();

// Read cycle count
uint64_t cycles = device->read_clock_counter();
device->toggle_clock_counter(false);

// Convert to time (assumes 375 MHz clock for AKD1000)
double latency_ms = (cycles / 375000.0);
printf("Inference latency: %.2f ms\n", latency_ms);
```

### 6. Batch Processing

For throughput-oriented applications, process multiple samples per batch:

```cpp
// Set batch size
device->set_batch_size(8, false);

// Prepare batched input (8 samples concatenated)
std::vector<uint8_t> batch_data(8 * input_size);
// ... fill with 8 samples ...

auto batch_input = akida::Dense::create_view(
    batch_data.data(),
    akida::Shape{8 * height, width, channels},  // Batch in first dimension
    akida::DType::uint8
);

// Process batch
device->enqueue(*batch_input);
auto batch_output = device->fetch();

// Extract individual results
// Output shape: [8 * out_height, out_width, out_channels]
```

## Troubleshooting

### No Devices Found

**Error:**
```
ERROR: No Akida devices found!
```

**Solutions:**

1. **Check PCIe driver:**
   ```bash
   lsmod | grep akida
   # Should show: akida_dw_edma

   # If not loaded:
   sudo modprobe akida_dw_edma
   ```

2. **Verify device visibility:**
   ```bash
   akida devices
   # Should list at least one device

   # If empty, check dmesg:
   dmesg | grep akida
   ```

3. **Check permissions:**
   ```bash
   ls -l /dev/akida*
   # Should have rw permissions

   # If not, either run as root or add udev rule:
   sudo usermod -a -G akida $USER
   # Then log out and back in
   ```

4. **Hardware connection:**
   - Verify the PCIe card is seated properly
   - Check `lspci | grep -i brain` to see if the card is detected

### File Lock Errors

**Error:**
```
ERROR: Failed to create hardware device
Device is busy (locked by another process)
```

**Solution:**
Only one process can access the device at a time. Close any other applications using the device:

```bash
# Check for processes using /dev/akida*
lsof /dev/akida*

# Kill them if safe
kill <pid>
```

### Build Errors

**Error:**
```
CMake Error: Could not find akida-model module
```

**Solution:**
The CMake module path is incorrect. Verify the package is placed at `engine/test/customer_deployment_package/`. The `CMakeLists.txt` expects this exact location.

If you must place it elsewhere, edit `src/CMakeLists.txt` line 21:

```cmake
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/<relative_path_to_engine>/cmake"
    ${CMAKE_MODULE_PATH}
)
```

**Error:**
```
fatal error: api/akida/hardware_device.h: No such file or directory
```

**Solution:**
The include path is incorrect. Verify that:
1. The engine was deployed with `akida engine deploy`
2. You're building from the correct directory
3. The `CMakeLists.txt` include path (line 63) correctly points to the engine root

### Runtime Errors

**Error:**
```
PANIC: Scratch memory too small
```

**Solution:**
The model requires more scratch memory than available on the device. This typically means:
- The model is too large for the device
- The batch size is too large

Try:
- Reducing batch size: `device->set_batch_size(1, false)`
- Using a smaller model
- Using a device with more memory

**Error:**
```
ERROR: Failed to program device - invalid ProgramInfo
```

**Solution:**
The model binary is incompatible with the device. This can happen if:
- The model was converted for a different chip (AKD1000 vs AKD1500)
- The `program.cpp` file is corrupted
- The model file was not properly converted

Re-run `convert_model.py` and ensure it completes without errors.

### Output Mismatch vs Python

**Issue:**
C++ inference results differ significantly from Python.

**Common causes:**

1. **Input format mismatch:**
   - Check that input data type matches (uint8 vs float)
   - Verify preprocessing is identical (normalization, scaling)
   - Ensure input shape is correct (HWC vs CHW)

2. **Sparse conversion:**
   - If `input_is_dense()` returns `false`, ensure you're converting to sparse
   - Check that the conversion parameters match Python

3. **Dequantization:**
   - Verify `activation_enabled()` check is correct
   - Ensure you're calling `dequantize()` when needed

4. **Test input data:**
   - Use `generate_test_input.py` to create test data from the exact same numpy array used in Python
   - Compare outputs element-by-element

**Acceptable differences:**
- Floating-point differences < 1e-5 are normal due to rounding
- Quantized models may have larger differences (< 1e-3) but should predict the same class

## Next Steps

### Integration Checklist

- [ ] Replace test input with your data source (file, camera, sensor)
- [ ] Add necessary preprocessing (normalization, resizing, color conversion)
- [ ] Add postprocessing (softmax, argmax, thresholding)
- [ ] Implement error handling and logging for production
- [ ] Add latency measurement and performance monitoring
- [ ] Test with diverse inputs to validate correctness
- [ ] Optimize batch size for your throughput requirements
- [ ] Add unit tests for the inference pipeline
- [ ] Document model-specific requirements (input format, class labels, etc.)

### Performance Optimization

1. **Minimize data copies:**
   - Use `create_view()` instead of `create()` when possible
   - Avoid unnecessary format conversions

2. **Batch processing:**
   - For throughput-critical applications, use larger batch sizes
   - Profile to find the optimal batch size for your latency requirements

3. **Pipelining:**
   - While one batch is being processed, prepare the next batch
   - Use multiple devices in parallel if available

4. **Asynchronous I/O:**
   - Don't block on input/output while inference is running
   - Use separate threads for data acquisition and inference

### Further Reading

For deeper understanding of the Akida C++ API, consult the full documentation:

- **Getting Started Guide** - Overview of the inference workflow
- **Hardware Device API** - Detailed API reference for `HardwareDevice`
- **Tensor Operations** - Working with Dense and Sparse tensors
- **Input Conversion** - Format conversion utilities
- **Program Info** - Understanding model metadata
- **Device Drivers** - Implementing custom drivers for embedded systems
- **Build System** - Advanced CMake configuration

These documents are included in the engine deployment at `engine/docs/`.

## Support

This package is provided as a reference implementation to help you get started. The code is heavily commented to explain the API usage and design decisions.

For issues specific to:
- **This package:** Review the code comments and troubleshooting section
- **Akida Engine API:** Consult the full documentation in `engine/docs/`
- **Model accuracy:** Verify against Python inference using `generate_test_input.py`
- **Performance:** See the optimization section above

## License

This package is provided as-is for use with BrainChip Akida hardware. The Akida Engine library is proprietary software licensed by BrainChip.

---

**Document Version:** 1.0
**Last Updated:** 2024
**Compatible with:** Akida Engine 2.x, AKD1000, AKD1500
