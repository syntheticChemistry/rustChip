# AKD1000 Hardware Deep-Dive

**Device**: BrainChip AKD1000 Neural Network Coprocessor
**PCIe ID**: `1e7c:bca1` at slot `08:00.0`
**Driver**: `akida_pcie` kernel module → `/dev/akida0`
**Board**: PCIe 2.0 x1 reference board, Arm Cortex-M4 @ 300MHz, 256Mbit LPDDR4

### Device Permissions

The udev rule at `/etc/udev/rules.d/99-akida-pcie.rules` sets `MODE="0666"`
but does not always trigger on boot. Current workaround: `pkexec chmod 666 /dev/akida0`.

**Evolution target (Rust)**: ToadStool's device manager should handle this properly —
open `/dev/akida0` via a Rust `DeviceManager` that either:
1. Uses a udev helper crate to trigger the rule, or
2. Opens the device fd with appropriate capabilities (CAP_DAC_OVERRIDE via
   a small setuid helper), or
3. Ships a systemd `.rules` snippet that adds a `plugdev` group match
   (user is already in `plugdev` group).

This is the same class of problem as GPU device access — solved in Rust,
not with manual chmod.

---

## 1. Architecture Overview

The AKD1000 is a **fully digital neuromorphic processor** — no analog circuits, no
memristors, no exotic physics. It's a deterministic, clocked digital chip that
implements **event-based sparse integer arithmetic** in silicon.

```
                    ┌─────────────────────────────────┐
                    │         AKD1000 SoC              │
                    │                                   │
 PCIe 2.0 x1 ─────►│  ┌──────┐  ┌──────┐  ┌──────┐  │
                    │  │ NP 0 │──│ NP 1 │──│ NP 2 │  │
    LPDDR4 ────────►│  └──┬───┘  └──┬───┘  └──┬───┘  │
   256 Mbit         │     │         │         │       │
                    │  ┌──┴───┐  ┌──┴───┐     ⋮       │
                    │  │ NP 3 │──│ NP 4 │  (80 NPs)   │
                    │  └──────┘  └──────┘             │
                    │                                   │
                    │  Cortex-M4 @ 300MHz   8MB SRAM   │
                    └─────────────────────────────────┘
```

### Neural Processor (NP) Node

Each NP is an independent compute element:

| Property | Value |
|----------|-------|
| NPUs per node | 4 |
| MACs per NPU | 128 |
| Local SRAM | 50-130 KB per NPU (configurable) |
| Total nodes | 80 (AKD1000 reference SoC) |
| Total MACs | 80 × 4 × 128 = **40,960 MACs** |
| Weight precision | 1, 2, 4-bit (Akida 1.0) |
| Activation precision | 1, 2, 4-bit |
| Input precision | 8-bit (first layer), 4-bit (internal) |

### Aggregate Compute

| Metric | Value | Notes |
|--------|-------|-------|
| Peak throughput | ~40 TOPS (at 1 GHz, theoretical) | Sparse event-based; actual depends on sparsity |
| Typical power | **~30 mW** active | Orders of magnitude below GPU |
| SRAM total | **8 MB** on-chip | All weights + activations resident |
| External memory | 256 Mbit LPDDR4 | For overflow / large models |
| PCIe bandwidth | ~500 MB/s (x1 Gen2) | Bottleneck for streaming workloads |

---

## 2. Compute Model: Event-Based Sparse Integer

This is the critical difference from GPUs. Understanding it determines what
we can and can't do.

### How a GPU computes (for comparison)

```
for each workgroup:
    for each thread:
        load operands from global memory
        multiply-accumulate (f32 or f64)
        store result
```

Every thread executes every instruction. Dense. Predictable. Power-hungry.

### How the AKD1000 computes

```
for each input event (non-zero activation):
    lookup connections (synapses) from this input
    for each connected neuron:
        accumulate: membrane_potential += weight × input_value
    if membrane_potential > threshold:
        emit output event (spike)
        reset membrane_potential
```

**Key properties:**

1. **Sparsity-proportional compute**: Zero inputs → zero work. A ReLU activation
   that's 90% zero means 90% fewer MACs. This is where the power savings come from.

2. **Integer-only arithmetic**: All operations are `int8 × int4` or `int4 × int4`
   multiply-accumulate. No floating point. No f32. No f64.

3. **Event-driven communication**: NPs communicate via sparse event packets (neuron
   index + activation value), not dense tensors. Only non-zero activations propagate.

4. **On-chip weight storage**: The 8MB SRAM holds all weights. No DRAM bandwidth
   bottleneck for models that fit. This is the AKD1000's killer advantage for
   small models.

5. **Multi-layer execution without CPU**: Once programmed, the entire network
   runs on-chip. Input → output with zero CPU intervention. The host only
   feeds input and reads output.

---

## 3. What the Hardware Actually Supports

### Supported Layer Types (Akida 1.0 — AKD1000)

| Layer | Operation | Input Bits | Weight Bits | Notes |
|-------|-----------|------------|-------------|-------|
| `InputConvolutional` | Conv2D | 8 | 4 | First layer; accepts raw uint8 |
| `Convolutional` | Conv2D | 4 | 4 | Internal layers |
| `SeparableConvolutional` | Depthwise + Pointwise | 4 | 4 | Efficient for spatial |
| `FullyConnected` | Dense (matmul) | 4 | 4 | Classification / regression |
| `InputData` | Passthrough | 8 | — | Raw data input |

### Supported Layer Types (Akida 2.0 — future IP, not on AKD1000)

Akida 2.0 adds `Dense1D`, `BufferTempConv`, `DepthwiseBufferTempConv`,
`Conv2DTranspose`, `Add`, `Concatenate`, skip connections, and a **lookup-table
activation function** (GeLU, SiLU, LeakyReLU, PReLU). Also supports 8-bit
weights and activations, and **Temporal Event-Based Neural Networks (TENNs)**.

**Critical note**: Our AKD1000 is **Akida 1.0**. It does NOT have TENNs,
8-bit internal weights, or LUT activations. Plan accordingly.

### On-Chip Learning (AKD1000 — Akida 1.0)

The AKD1000 supports **STDP-based edge learning** on the last `FullyConnected`
layer only:

- 1-bit weights and 1-bit inputs only
- Unsupervised / semi-supervised
- Few-shot class augmentation (add new classes without retraining)
- Biologically-inspired homeostatic plasticity

This is fascinating for adaptive classification but **not useful for our
regression/prediction workloads** which need multi-bit precision.

---

## 4. Access Layers

### Layer 1: Python SDK (MetaTF)

```
pip install akida
```

Workflow: TF-Keras model → QuantizeML (quantize to int4/int8) → CNN2SNN (convert) → Akida Model

- Highest abstraction level
- Limited to supported layer types
- Cannot express arbitrary computation
- Good for: prototyping, model validation, benchmarking existing architectures

### Layer 2: Akida Runtime API (Python)

```python
from akida import Model, devices
device = devices()[0]
model = Model("my_model.fbz")
model.map(device)
output = model.forward(input_data)
print(model.statistics)  # FPS, power, clock cycles
```

- Direct model mapping and inference
- Access to hardware statistics (inference_clk, program_clk, power)
- Still constrained to supported layer types
- Good for: hardware benchmarking, deployment validation

### Layer 3: Akida Engine (C++)

The Engine library provides register-level access through `HardwareDriver`:

```cpp
class HardwareDriver {
    virtual void read(uint32_t addr, void* data, size_t size) = 0;
    virtual void write(uint32_t addr, const void* data, size_t size) = 0;
    virtual void* scratch_memory() = 0;
    virtual size_t scratch_memory_size() = 0;
    virtual void* akida_visible_memory() = 0;
    virtual size_t akida_visible_memory_size() = 0;
};
```

- Can read/write arbitrary registers and memory
- Program the chip with raw binary programs
- DMA data to/from `akida_visible_memory`
- **This is our path to direct-wire access** (like we did with GPU f64)

### Layer 4: PCIe BAR Direct Access

The `akida_pcie` driver exposes `/dev/akida0`. With the right ioctls or
mmap, we could theoretically bypass everything and talk to BAR registers
directly. Uncharted territory — the driver source is on GitHub.

### Layer 5: Direct SRAM Access (rustChip)

rustChip provides direct BAR0 register and BAR1 SRAM read/write via
`SramAccessor` (akida-driver). The C++ engine exposes multiple SRAM types
per NP:

| SRAM Type | Width | Purpose |
|-----------|-------|---------|
| Filter (FSRAM) | 64b | Filter/synapse weights |
| Threshold (TSRAM) | 51b | Neuron threshold storage |
| Event (EVSRAM) | 32b | Event buffers |
| Status (STSRAM) | 32b | Neuron state/status |

**BAR1 access methodology**: `SramAccessor` combines BAR0 register probing
with BAR1 memory-mapped access. `VfioBackend::map_bar1()` provides the
VFIO path for BAR1 SRAM. `Capabilities::from_bar0()` extracts runtime NP
count, SRAM size, and mesh topology from BAR0 registers.

**probe_sram binary**: Three modes — `probe` (register discovery), `scan`
(BAR1 region enumeration), `test` (read/write verification). Use for
SRAM layout exploration and load verification.

---

## 5. Performance Envelope

### Measured — BrainChip Reference Benchmarks

| Workload | FPS | Power | Notes |
|----------|-----|-------|-------|
| Visual Wake Word (96×96) | ~100 | ~9 mW | Trivial model |
| DS-CNN Keyword Spotting | ~63 | ~30 mW | 6-layer, 65 NPs |
| AkidaNet ImageNet (224×224) | ~43 | ~100 mW | 15-layer, 68 NPs, 1.4MB program |
| Inference clocks (DS-CNN) | — | — | 93,965 clocks per frame |
| Programming clocks (DS-CNN) | — | — | 152,396 clocks |

### Measured — Our Hardware (Feb 20, 2026)

**ESN Readout Model**: InputConv(1,1,8→50) → FullyConnected(50→1)
- InputConv runs in software (8 channels not hardware-compatible)
- FC readout maps to **1 FNP3 node**, program = **752 bytes**

| Metric | Measured Value | Notes |
|--------|---------------|-------|
| **Inference clocks** | **668 cycles** | FC readout on hardware |
| **Program clocks** | **457 cycles** | Model loading |
| **Hardware FPS** | **1,525** | PCIe round-trip dominated |
| **Hardware latency** | **656 μs** | ~650 μs is PCIe overhead |
| **Software FPS** | **143,710** | CPU simulation, no PCIe |
| **Software latency** | **7 μs** | Pure CPU, in-process |
| **Board floor power** | **918 mW** | PCIe board overhead |
| **Chip inference power** | **< board noise** | Too small to measure above floor |

**4-layer model**: InputConv(3→64) → SepConv(128) → SepConv(256) → FC(10)
- FC readout maps to 1 FNP3, program = 2,840 bytes

| Metric | Value |
|--------|-------|
| Inference clocks | 1,877 |
| Program clocks | 1,729 |
| Board floor | 906 mW |

### Key Insight: PCIe Latency Dominates (But Batch Amortizes)

The AKD1000 computes in **668 clocks** (~0.7 μs at 1 GHz) but the PCIe x1
Gen2 round-trip adds **~650 μs** of latency. For our tiny ESN readout, the
CPU is 100× faster because it avoids the bus transfer entirely.

**Batch inference changes the calculus**: At batch=8, PCIe overhead amortizes
to **390 μs/sample** (2.4× over single inference). The PCIe→compute crossover
is at ~width=512 — above this, compute dominates and the NPU's parallelism
matters more than the bus latency.

**Where NPU wins**: Large FC width (512+), batched inference (8+ samples),
continuous low-power monitoring, or when power budget matters more than latency.
Economy clock mode saves 18% power at only 19% latency cost.

### Model Weight Budget

| Component | Weights | At int4 | Fits in SRAM? |
|-----------|---------|---------|--------------|
| W_in (input projection) | 8 × 50 = 400 | 200 B | Yes |
| W_res (reservoir) | 50 × 50 = 2,500 | 1,250 B | Yes |
| W_out (readout) | 50 × 1 = 50 | 25 B | Yes |
| **Total ESN** | **2,950** | **1,475 B** | **Trivially** |
| 256×256 deep FC | 131,072 | 65 KB | Yes |
| 1024×1024 massive FC | 2,097,152 | 1 MB | Yes |
| 4096×4096 extreme FC | 33,554,432 | ~4 MB | Yes (half SRAM) |
| SRAM capacity | — | **8 MB** | **Tested to 8192 width** |

---

## 6. Constraints & Limitations

### Hard Constraints

| Constraint | Impact on ecoPrimals |
|------------|---------------------|
| **Integer-only arithmetic** | Cannot do f64 or f32 natively. All physics must be quantized to int4/int8 |
| **No custom compute kernels** | Cannot write arbitrary shaders. Must express work as supported layer types |
| **4-bit activations internally** | Dynamic range is 0-15. Physics values must be scaled and quantized |
| **Akida 1.0 layer set** | No temporal convolutions (TENNs), no skip connections, no LUT activations |
| **PCIe x1 Gen2 bandwidth** | ~500 MB/s max. Streaming high-rate MD data is bottlenecked here |
| **No feedback loops** | Feed-forward only. Recurrence must be unrolled or handled on host |

### Soft Constraints

| Constraint | Workaround |
|------------|------------|
| No f32 tanh | Quantize after tanh on host, or approximate with ReLU (loses reservoir dynamics) |
| 8-bit input only | Scale physics inputs to uint8 range [0, 255] |
| Fixed network topology | Pre-compile multiple models for different parameter regimes |
| No dense matmul primitive | Express as `FullyConnected` layer (which IS a matmul, just quantized) |

---

## 7. The GPU Parallel: What "Direct Wire" Means for NPU

### What we did with GPU (BarraCuda)

The conventional wisdom: consumer GPUs have f64 at 1:64 throughput (limited
dedicated FP64 hardware on GeForce). Too slow for compute-heavy science.

**What we actually found** (corrected Feb 24, 2026): Consumer Ampere/Ada
fp64 IS hardware ~1:64 — confirmed by `bench_fp64_ratio` on RTX 3090 (0.33
TFLOPS fp64 via both CUDA and Vulkan). BUT: double-float (f32-pair)
arithmetic on the massive FP32 core array delivers **3.24 TFLOPS at 14-digit
precision** — 9.9× faster than native f64. The Titan V (Volta, 1:2 native)
provides genuine compute-class f64 through the open-source NVK driver.

### What "direct wire" means for NPU

The MetaTF SDK assumes you're doing image classification or keyword spotting.
It presents the AKD1000 as a black-box NN accelerator.

**What we want to explore**:

1. **The `FullyConnected` layer IS a matrix-vector multiply.** Can we load
   arbitrary quantized weight matrices and use the NPU as a fast, low-power
   matmul engine? The ESN readout (`W_out · state`) is exactly this.

2. **The event-based compute model rewards sparsity.** Our reservoir states
   after tanh have significant sparsity (~50-80% zeros after quantization).
   The NPU would skip all zero multiplications — free speedup.

3. **The 8MB SRAM is a scratchpad.** For small models like our ESN (1.5 KB
   weights), we could potentially load dozens of different ESN configurations
   and switch between them without reprogramming.

4. **The C++ Engine exposes register-level read/write.** We can probe the
   actual memory layout, program format, and potentially discover operations
   the SDK doesn't expose.

5. **Multi-model pipelining.** The host feeds velocity features; the NPU
   runs the ESN forward pass; the host reads D*. The NPU draws microwatts
   while the GPU runs the full MD simulation. Heterogeneous compute.
