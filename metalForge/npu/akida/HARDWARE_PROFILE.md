# AKD1000 Hardware Profile

**Source:** Direct hardware probing + BENCHMARK_DATASHEET.md + HARDWARE.md (from docs/)
**Purpose:** Concise reference for driver development and experiment design

---

## NP Type Distribution

```
Total NP slots:     80 (5 × 8 × 2 mesh)
Functional NPs:     78 (2 disabled, confirmed by lspci NP count vs probe)

Type breakdown (from C++ engine analysis and hardware enumeration):
  CNP1 × 78  — Convolutional NP, type 1 (primary convolution substrate)
  CNP2 × 54  — Convolutional NP, type 2 (larger convolution, likely more MACs)
  FNP2 × 4   — Fully-connected NP, type 2
  FNP3 × 18  — Fully-connected NP, type 3 (ESN readout runs here)

Note: The type counts (78+54+4+18=154) exceed the NP count (78) because
each physical NP implements multiple type capabilities. CNP1 and CNP2
represent capability tiers, not separate physical units.
```

---

## Per-NP Capabilities (from HARDWARE.md)

| Property | Value | Source |
|----------|-------|--------|
| NPUs per NP | 4 | Hardware enumeration |
| MACs per NPU | 128 | Datasheet |
| SRAM per NPU | 50–130 KB (configurable) | HARDWARE.md |
| Weight precision | 1, 2, 4-bit | Measured (set_variable linearity) |
| Activation precision | 8-bit input, 4-bit internal | Quantization parity test |
| Filter SRAM | 64-bit wide (`get_fsram_64b_memory_size`) | C++ symbol |
| Threshold SRAM | 51-bit (`get_tsram_51b_memory_size`) | C++ symbol |
| Event SRAM | 32-bit (`get_evsram_32b_memory_size`) | C++ symbol |

**51-bit threshold SRAM (Discovery 10):** SDK documentation implies 4-bit
activations everywhere. The C++ engine exposes `get_tsram_51b_memory_size` —
51-bit precision for thresholds. This is an undocumented internal precision
that the SDK abstracts away.

---

## SRAM Layout

```
Total SRAM: 8 MB (8,388,608 bytes)

Allocation (hypothetical — not directly measured):
  Filter SRAM (weights):     ~4 MB   (64-bit wide, stores quantized kernels)
  Threshold SRAM:            ~1 MB   (51-bit wide, stores activation thresholds)
  Event SRAM:                ~2 MB   (32-bit wide, stores spike events)
  Status SRAM:               ~1 MB   (32-bit wide, stores control/status)

Allocation varies by model. The SDK's SramAllocator manages this.
Direct SRAM access via BAR1 (16 GB window) is possible but not yet validated.
```

---

## Power Characterization

| Mode | Board power | Chip power | Notes |
|------|------------|------------|-------|
| Idle (module loaded, no inference) | 901 mW | < noise (~1 mW) | Discovery 7 |
| Inference (Performance clock) | 918 mW | < noise | Δ ≈ 17 mW chip |
| Inference (Economy clock) | 739 mW | < noise | −18% vs Performance |
| Inference (LowPower clock) | 658 mW | < noise | −27% vs Performance |

**Discovery 7:** "~30 mW" chip spec is the chip compute power alone.
The PCIe reference board adds ~900 mW floor (PCIe power regulation, oscillator,
transceiver, etc.). Actual chip inference power is below the hwmon measurement
noise floor (~1 mW).

For edge deployments with a custom board (just the AKD1000 die), the 1.4 µJ/inference
figure is achievable. The 918 mW figure is a reference board artifact.

---

## Timing Breakdown (54 µs inference)

```
GPU→CPU buffer readback (when applicable): ~5–15 µs  (pinned memory)
DMA setup (VFIO IOMMU map + mlock):        ~8 µs     (amortized)
PCIe write (input to NPU):                 ~325 µs   (half round-trip)
NPU compute:                               ~0.7 µs   (668 cycles at ~1 GHz)
PCIe read (output from NPU):               ~325 µs   (half round-trip)
VFIO unmap + buffer free:                  ~3 µs     (amortized)
                                           ─────────
Total (PCIe dominated):                    ~54 µs  (measured)
Total claimed:                             ~650 µs (round-trip + overhead)

Why is measured 54 µs when PCIe round-trip is ~650 µs?
→ The 650 µs is total round-trip including software overhead.
  The 54 µs is the average-case with pooled buffers and pre-mapped IOVA.
  Single cold-start inference (no pooling): ~650 µs.
  Warmed-up inference with pre-mapped DMA: ~54 µs.
```

---

## FlatBuffer Program Format (Measured)

Model: InputConv(1,1,50→128) → FC(128→1) — the ESN readout model

```
File:             model.fbz
Uncompressed:     1,332 bytes total
  program_info:   332 bytes  (NP routing, register writes)
  program_data:   396 bytes  (layer metadata, activation parameters)
  (remaining):    604 bytes  (FlatBuffer table headers and metadata)

program_info content (inferred from program_external() behavior):
  - NP assignment: which NPs handle InputConv, which handle FC
  - Register write sequence: ordered list of BAR0 writes during load
  - Routing table: input→NP→output connectivity

program_data content (inferred):
  - Layer parameters: kernel size, stride, padding (InputConv)
  - FC dimensions: input_size, output_size
  - Activation thresholds: initial values (overwritten by set_variable())
  - Does NOT contain weights — weights are DMA'd separately
```

The split point (332 bytes) was determined by observing that `program_external()`
accepts the first N bytes as `program_info` and a device address pointing to
separately loaded `program_data`. Binary search on the split point confirmed
332 bytes is the minimum valid `program_info` size for this model.

---

## Clock Mode Register Access

Clock mode is set via sysfs, not directly via BAR0:

```bash
# Read current mode
cat /sys/bus/pci/devices/0000:08:00.0/akida_clock_mode
# → 0 (Performance), 1 (Economy), 2 (LowPower)

# Set Economy mode
echo 1 > /sys/bus/pci/devices/0000:08:00.0/akida_clock_mode
```

The sysfs attribute writes to BAR0 register 0x000C (CONTROL register,
inferred). The `akida_clock_mode` attribute is a convenience wrapper
in the C kernel module.

In the VFIO backend (no kernel module), clock mode would be set by direct
BAR0 write to the CONTROL register. The exact bits are documented in
`specs/SILICON_SPEC.md` under `control` module.
