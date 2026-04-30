# Experiment 006: BAR0 Register Probe — True Layout Discovery

**Date:** 2026-04-30
**Hardware:** AKD1000 rev 01 @ PCIe 0000:e2:00.0, NUMA node 1
**Driver:** vfio-pci (pure userspace)
**Binary:** `probe_registers`

## Objective

Determine the actual BAR0 register layout on a cold-boot VFIO-bound
AKD1000 by reading every confirmed and inferred register offset, and
scanning the full 4 MB BAR for non-zero content.

## Key Finding: BAR0 is SRAM, Not a Register File

**BAR0 on a VFIO-bound AKD1000 exposes 4 MB of raw NP SRAM content,
not a discrete control register interface.**

Evidence:

1. **1,040,263 out of 1,048,576 dwords are non-zero** — the entire
   4 MB is populated with data. Only 8,313 dwords are zero.
   No `0xFFFFFFFF` values. No `0xBADF5040` ("bad food") patterns.

2. **All "confirmed" registers read different values** than previous
   probing sessions:

| Offset   | Expected (BEYOND_SDK) | Actual (VFIO cold) | Match |
|----------|----------------------|-------------------|-------|
| 0x0000   | 0x194000a1 (DEVICE_ID) | 0x00890081       | NO    |
| 0x1094   | 0x0000a028 (CONTROL)   | 0x10000108       | NO    |
| 0x10C0   | 0x5b (NP_COUNT=91)     | 0x08000611       | NO    |
| 0x1410   | 0x2000 (SRAM_REGION)   | 0x08000000       | NO    |
| 0x1418   | 0x8000 (SRAM_REGION)   | 0x00100800       | NO    |
| 0x4010   | 0x04aa0001 (DMA_MESH)  | 0x89080800       | NO    |
| 0x1E0C   | 0x1 (NP_ENABLE[0])     | 0x81210600       | NO    |

3. **Write-readback works** — offset 0x1094 is writable:
   ```
   Original:    0x10000108
   After |= 1:  0x10000109
   Restored:    0x10000108
   ```
   This is consistent with SRAM (read-write) rather than read-only
   config space.

4. **STATUS at 0x0008 reads 0x30000910 consistently** — unchanged
   by VFIO_DEVICE_RESET, CONTROL writes, or any other operation.
   This is not a status register; it's a static SRAM word.

## Interpretation

The previous "confirmed" probing (documented in `docs/BEYOND_SDK.md`)
was performed with the **kernel driver loaded** (`akida_pcie`). The
kernel driver initializes the device firmware and configures BAR0 to
expose a register interface. Without the kernel driver's init sequence:

- BAR0 shows raw SRAM content from previous sessions
- The "registers" at 0x0000, 0x1094, etc. are SRAM addresses that
  the firmware monitors — they're a mailbox protocol, not hardware
  registers
- The firmware needs to be running (started by the kernel driver)
  for these addresses to have their documented meanings

The AKD1000 likely has an on-chip microcontroller that:
1. Boots from flash or is initialized by the host driver
2. Monitors specific SRAM addresses for commands (mailbox)
3. Exposes "register" semantics by polling these SRAM words
4. Controls the NP mesh based on mailbox commands

## What Works

| Capability | Status |
|-----------|--------|
| VFIO device open | YES |
| BAR0 mmap (4 MB) | YES |
| BAR0 read (all offsets) | YES |
| BAR0 write-readback | YES |
| BAR2 mmap (SRAM window, 4 MB) | YES |
| BAR2 read/write | YES |
| DMA buffer allocation | YES |
| IOMMU mapping | YES |
| VFIO_DEVICE_RESET ioctl | YES (succeeds, no visible effect) |
| Device init (READY state) | NO — firmware not running |
| Model load via DMA | NO — blocked on firmware |
| Inference | NO — blocked on firmware |

## BAR0 Census

```
Total dwords   : 1,048,576
Zero           : 8,313
0xFFFFFFFF     : 0
0xBADF5040     : 0
Other non-zero : 1,040,263
Scan time      : 1.54s
```

## Per-NP Config Region (0xE000+)

The per-NP configuration blocks at stride 0x100 contain dense,
non-repeating data consistent with NP weight/configuration state
from a previous model load:

```
NP00: 010a0102 0421a890 18148900 00040881 21000100 80701c0c 40008900 40008000
NP01: 84000004 2040082c 120000c8 00052c09 00120840 14260100 00804003 00008280
NP02: 10810000 04032648 20003110 04a0a404 80840000 12800200 90000080 00000940
```

Each NP block has unique content — these are likely the quantized
weight matrices from a model loaded in a previous kernel-backed session.

## Path Forward

### Option A: Firmware Init from Kernel Driver (Pragmatic)

Reverse-engineer the `akida_pcie` kernel module's init sequence:
1. Load the kernel module briefly, capture register writes via `ftrace`
2. Identify the firmware upload and mailbox protocol
3. Replicate in Rust from userspace (write to SRAM addresses)
4. Then unbind and switch to VFIO for ongoing operation

### Option B: eDMA Engine Discovery (Protocol)

The AKD1000 uses a DesignWare eDMA engine. Standard DW eDMA registers
live at known offsets in BAR0. If we can identify and configure the
eDMA channels, we may be able to DMA model data directly to NP SRAM
and trigger inference through the eDMA completion mechanism.

### Option C: Dual-Driver Bootstrap (Hybrid)

Keep the kernel driver for init only:
1. `modprobe akida_pcie` → firmware starts, registers become live
2. Read confirmed registers to verify firmware state
3. `unbind` from akida_pcie, `bind` to vfio-pci
4. VFIO path takes over with firmware already running

This may work if the firmware survives driver unbind/rebind.

## Code Changes

- `vfio/mod.rs`: Added `reset_and_enable()` method with VFIO_DEVICE_RESET
  + CONTROL register writes. Currently non-functional (BAR0 is SRAM).
- `vfio/mod.rs`: Added `read_bar0_u32()`, `write_bar0_u32()`, `bar0_size()`
  for raw register probing.
- `vfio/mod.rs`: `infer()` no longer hard-errors on READY=false; logs
  and attempts anyway (empirical register map).
- `vfio/mod.rs`: Fixed `load_model()` slice copy for page-aligned DMA buffers.
- `vfio/ioctls.rs`: Added `ioctl_vfio_device_reset()` wrapper.
- `probe_registers.rs`: New binary for comprehensive BAR0 probing.
- `hw_live_inference.rs`: End-to-end VFIO inference binary (blocked on firmware).
- `hw_vs_sw_npu.rs`: HW vs SW comparison with graceful degradation.
- `ensemble_npu.rs`: Multi-backend ensemble inference (voting, averaging, cascade).
