# Silicon Specification — AKD1000 / AKD1500

**Source**: Direct hardware probing + C++ engine symbol analysis (1,048 exports)
**Methods**: BAR0 MMIO reads, sysfs queries, `program_external()` injection,
             `set_variable()` weight mutation, batch inference sweeps
**Validation**: All claims in `../docs/BEYOND_SDK.md` and `../docs/HARDWARE.md`

> **Note on confidence**: Confirmed values have been directly measured.
> Inferred values match observed behavior. Hypothetical values are labeled
> and should be validated with BrainChip datasheet / silicon team.

---

## 1. PCI Identity

| Property | Value | Source |
|----------|-------|--------|
| Vendor ID | `0x1E7C` (BrainChip) | PCI-SIG assigned |
| AKD1000 device ID | `0xBCA1` | `lspci -d 1e7c:` |
| AKD1500 device ID | `0xA500` | AKD1500 datasheet v1.2 |
| AKD1500 alt ID | `0xBCA2` | Some revisions |
| Class | `0x040000` (Multimedia other) | `lspci -v` |
| Subsystem | `0x1E7C:0x0001` | Confirmed |

---

## 2. PCIe Link

| Property | AKD1000 | AKD1500 |
|----------|---------|---------|
| Link width | x1 | x2 |
| Link speed | Gen2 (5 GT/s) | Gen2 |
| Theoretical bandwidth | ~500 MB/s | ~1 GB/s |
| Measured DMA throughput | **37 MB/s** sustained | (extrapolated ~74 MB/s) |
| PCIe round-trip latency | **~650 µs** | similar |
| DW eDMA controller | Yes (DesignWare Enhanced DMA) | Yes |

The gap between theoretical (500 MB/s) and measured (37 MB/s) reflects:
- Transaction layer overhead (TLP headers, credits, ACKs)
- DW eDMA descriptor overhead
- Kernel/IOMMU path costs
- The AKD1000's PCIe x1 link being primarily latency-bound for small payloads

---

## 3. BAR Layout

```
BAR  Address         Size     Type                    Purpose
──── ──────────────  ──────   ─────────────────────   ──────────────────────────────
 0   0x84000000      16 MB    32-bit non-prefetch      Register space (MMIO) ← primary
 1   0x4000000000    16 GB    64-bit prefetchable      NP mesh / SRAM window
 3   0x4400000000    32 MB    64-bit prefetchable      Secondary memory
 5   0x7000          128 B    I/O ports                Control ports
 6   0x85000000      512 KB   Expansion ROM            Firmware
```

**Discovery 8 (BEYOND_SDK.md):** BAR1 exposes 16 GB decode range — far larger
than the 8 MB physical SRAM spec. With 78 NPs, each could address ~200 MB.
BAR1 first 64 KB reads as all-zeros (sparse NP-mapped layout).

The VFIO driver maps BAR0 for all control; BAR1 exploration is ongoing.

---

## 4. BAR0 Register Map

All offsets are byte addresses within the 16 MB BAR0 window.
`confirmed` = directly probed value matches. `inferred` = behavior-consistent,
not read directly. `hypothetical` = DesignWare eDMA standard layout assumed.

### Device Identity & Status

| Offset | Name | Confirmed Value | Notes |
|--------|------|-----------------|-------|
| `0x0000` | `DEVICE_ID` | `0x194000a1` | Version + device in one word — confirmed |
| `0x0008` | `STATUS` | varies | Ready/Busy/Error/ModelLoaded bits |
| `0x0010` | `CONTROL` | `0x0000a028` | Enable/Reset/PowerSave — confirmed @ 0x001094 |

### NP Mesh Configuration

| Offset | Name | Confirmed Value | Notes |
|--------|------|-----------------|-------|
| `0x0010C0` | `NP_COUNT` | `0x5b` (91) | Reads 91 — 80 NPs + overhead — confirmed |
| `0x1E0C–0x1E20` | `NP_ENABLE[0..5]` | `0x00000001` × 6 | NP enable bits — confirmed |
| `0x4010` | `DMA_MESH_CONFIG` | `0x04aa0001` | DMA/mesh config word — confirmed |
| `0x1410` | `SRAM_REGION_0` | `0x2000` | SRAM region config — confirmed |
| `0x1418` | `SRAM_REGION_1` | `0x8000` | SRAM region config — confirmed |

### Model Load

| Offset | Name | Notes |
|--------|------|-------|
| `0x0100` | `MODEL_ADDR_LO` | Low 32 bits of IOVA (for DMA) or sysfs addr |
| `0x0104` | `MODEL_ADDR_HI` | High 32 bits |
| `0x0108` | `MODEL_SIZE` | Program size in bytes |
| `0x010C` | `MODEL_LOAD` | Write 1 to trigger load; poll STATUS for completion |

### Inference

| Offset | Name | Notes |
|--------|------|-------|
| `0x0200` | `INPUT_ADDR_LO` | Input buffer IOVA low |
| `0x0204` | `INPUT_ADDR_HI` | Input buffer IOVA high |
| `0x0208` | `INPUT_SIZE` | Input size in bytes |
| `0x0300` | `OUTPUT_ADDR_LO` | Output buffer IOVA low |
| `0x0308` | `OUTPUT_SIZE` | Output size (read after completion) |
| `0x0400` | `INFER_START` | Write 1 to start; poll INFER_STATUS |
| `0x0404` | `INFER_STATUS` | Bit 0: done; Bit 1: error |

### Status Register Bits

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `READY` | Device ready to accept commands |
| 1 | `BUSY` | Inference or load in progress |
| 2 | `ERROR` | Error in last operation |
| 3 | `MODEL_LOADED` | Program successfully loaded |

### DesignWare eDMA (hypothetical standard layout)

| Offset | Name | Notes |
|--------|------|-------|
| `0x0200` | `EDMA_WRITE_CH0_CTL` | eDMA write channel 0 control |
| `0x0300` | `EDMA_READ_CH0_CTL` | eDMA read channel 0 control |
| `0x100010` | `EDMA_INT_STATUS` | Interrupt status |

The DW eDMA layout is from the DesignWare PCIe eDMA databook.
Actual offsets must be confirmed against silicon or BrainChip datasheet.

### Per-NP Registers (pattern confirmed at 0xe000+)

| Offset pattern | Name | Notes |
|----------------|------|-------|
| `0xE000 + n×0x100` | `NP_CONFIG[n]` | Per-NP register block; repeating pattern confirmed |

---

## 5. NP Mesh Topology

```
AKD1000: 5 × 8 × 2 NP mesh, 78 functional (2 disabled)

NP types (from C++ engine analysis):
  CNP1  ×78  — Convolutional NP, type 1
  CNP2  ×54  — Convolutional NP, type 2
  FNP2  × 4  — Fully-connected NP, type 2
  FNP3  ×18  — Fully-connected NP, type 3 (ESN readout runs here)
```

**Key capability per NP** (from HARDWARE.md):

| Property | Value |
|----------|-------|
| NPUs per NP | 4 |
| MACs per NPU | 128 |
| Total MACs (78 NPs) | 39,936 (BENCHMARK: 40,960 per spec with 80 NPs) |
| SRAM per NPU | 50–130 KB (configurable) |
| Weight precision | 1, 2, 4-bit |
| Activation precision | 8-bit input, 4-bit internal |

**SkipDMA routing (Discovery 2):** NP-to-NP data transfer bypasses PCIe.
Deep FC chains (up to depth=8 tested) execute as a single hardware pass —
the latency overhead per additional layer is ~0 µs.

---

## 6. FlatBuffer Program Format

Reverse-engineered from `.fbz` model files and `program_external()` call analysis.
See `../docs/BEYOND_SDK.md` Discovery 7 and 9 for methodology.

```
.fbz file structure:
  [4 bytes]  magic "AKIDA" (not standard FlatBuffer)
  [N bytes]  Snappy-compressed payload
    → decompressed: FlatBuffer binary
      → root table → program_info + program_data

program_info  (~332 bytes for minimal ESN model):
  NP routing: which NPs execute which layers
  Register write sequence: ordered list of BAR0 writes during programming
  Structure is independent of weights

program_data  (~396 bytes for minimal ESN model):
  Layer metadata: activation parameters, pooling config
  Initial values for configurable parameters
  Does NOT contain weights (weights are DMA'd via set_variable() / program_external())
```

**program_external() signature** (from C++ engine export):
```
program_external(self, bytes: &[u8], device_address: u64) -> Result<()>
  "Program a device using a serialized program_info bytes object,
   and the address, as it is seen from akida on the device, of
   corresponding program_data that must have been written beforehand."
```

This means:
1. DMA `program_data` bytes to a known IOVA address
2. Pass that IOVA as `device_address` to `program_external()`
3. `program_info` bytes describe the register-write sequence
4. Weights follow via `set_variable()` DMA transfers

---

## 7. Clock Modes (Discovery 4)

Three modes confirmed via sysfs `akida_clock_mode` attribute:

| Mode | Setting | Latency | Power | Use case |
|------|---------|---------|-------|----------|
| Performance | `0` | 909 µs | 901 mW (board) | Default — maximum throughput |
| Economy | `1` | 1,080 µs (+19%) | 739 mW (-18%) | **Preferred for physics workloads** |
| LowPower | `2` | 8,472 µs (+9.3×) | 658 mW (-27%) | Idle / standby — avoid for inference |

Economy mode is the sweet spot: 19% slower, 18% less power. For workloads
where the PCIe round-trip (~650 µs) already dominates, switching to Economy
adds only 171 µs (23% penalty on total latency) while saving 162 mW.

---

## 8. AKD1500 Delta

The AKD1500 uses the same Akida 1.0 IP as AKD1000. Changes:

| Property | AKD1000 | AKD1500 |
|----------|---------|---------|
| Device ID | `0xBCA1` | `0xA500` / `0xBCA2` |
| PCIe lanes | x1 | x2 |
| Bandwidth | ~500 MB/s | ~1 GB/s |
| Package | PCIe card (reference board) | 7×7 mm BGA169 |
| GPIO | None | 24 GPIO pins |
| SPI | None | SPI master/slave |
| SLEEP pin | None | Hardware sleep control |
| CMA kernel | Optional | **Required** for large models |
| Linux support | Up to ~6.8 | "No plans past 6.8" (datasheet) |

**The AKD1500 datasheet's kernel support note** is the primary motivation
for Phase E (Rust kernel module): the C module requires per-kernel rebuilds.
A Rust kernel module using stable bindings doesn't.
