# Experiment 005: Wildlife Preserve Baseline

**Date:** 2026-04-30
**Hardware:** AKD1000 rev 01 @ PCIe 0000:e2:00.0, NUMA node 1
**Driver:** vfio-pci (pure userspace, no kernel module)
**IOMMU:** AMD Starship/Matisse, group 92 (isolated)

## Objective

Establish a hardware baseline for the Wildlife Preserve by bringing the
AKD1000 online through the pure Rust VFIO path — no kernel module, no
Python SDK, no root for steady-state operation.

## Discovery Path

### 1. VFIO Device Discovery

`DeviceManager::discover()` was extended with a sysfs fallback path.
When no `/dev/akida*` nodes exist (kernel driver not loaded), the
discovery engine scans `/sys/bus/pci/devices/` for BrainChip vendor ID
`0x1e7c` and creates device entries for any matches, regardless of which
driver is bound.

```
Akida devices: 1
[0] AKD1000 @ 0000:e2:00.0
     PCIe  Gen2 x1  (0.5 GB/s theoretical)
     NPUs  80   SRAM  10 MB
     WeightMut Full
```

### 2. IOMMU Verification

```
IOMMU group for 0000:e2:00.0: 92
Device file: /dev/vfio/92
```

Group 92 is isolated (single device). `/dev/vfio/92` is world-readable
after a one-time `chmod` — no ongoing root needed.

### 3. VFIO Backend Initialization

```
VFIO device: 9 regions, 5 IRQs
BAR0 mapped at 0x7b3921e00000, size=0x400000 (4 MB)
Init time: ~150 ms
```

The VfioBackend opens the VFIO container, attaches the group, opens the
device, and maps BAR0 through the VFIO device fd. This is a different
mapping path than sysfs `resource*` — the latter returns `0xFFFFFFFF`
for VFIO-bound devices because reads bypass the IOMMU.

### 4. BAR Layout (sysfs)

| BAR | Address | Size | Type |
|-----|---------|------|------|
| 0 | 0x4009dc00000 | 4 MB | 64-bit prefetchable |
| 2 | 0x4009d800000 | 4 MB | 64-bit prefetchable |
| 4 | 0x4009d400000 | 4 MB | 64-bit prefetchable |

All BARs are 64-bit. The original BAR enum used sequential indices
(0, 1, 2) which mapped to VFIO regions 0, 1, 2. For 64-bit BARs,
VFIO region 1 is the upper half of BAR0's address and isn't
independently mappable. Fixed to use VFIO regions 0, 2, 4.

### 5. BAR2 SRAM Probe (via VFIO device fd)

```
BAR1/SRAM mapped: 4,194,304 bytes (4 MB) in 199 µs
Probed 80 offsets, 80 non-zero/non-FF
```

Sample data at stride 0x1000:
```
0x00000000: 0x00010011
0x00001000: 0x00018012
0x00002000: 0x218b01c6
0x00003000: 0x01700002
...
0x0000f000: 0x04360220
```

Every probed offset contains data — the SRAM is populated with
configuration/state from previous sessions.

### 6. SRAM Write/Readback

```
Write 0xCAFEBABE → Read 0xCAFEBABE  PASS
```

Bidirectional MMIO through the VFIO device fd works correctly.
This confirms full read/write access to on-chip SRAM from userspace.

### 7. DMA Allocation

```
Allocated 4096-byte DMA buffer
IOVA: 0x10000000
Host size: 4096
```

IOMMU-mapped DMA buffer allocation succeeds. The IOVA space starts
at 256 MB and grows upward.

### 8. Device Readiness

The device reports `is_ready() = false` — the chip has not been
through a reset/init sequence in this session. Model load and
inference require the device to be in READY state first. This is
expected for a cold start without the kernel driver's init path.

## Key Findings

1. **Pure userspace VFIO works** — all BAR access, DMA allocation,
   and SRAM read/write function without root or kernel modules.

2. **64-bit BAR fix required** — the original BAR enum used sequential
   VFIO region indices. For AKD1000's 64-bit BARs, the correct mapping
   is region 0 (BAR0), region 2 (BAR2/SRAM), region 4 (BAR4/secondary).

3. **sysfs resource mmap doesn't work for VFIO** — `/sys/bus/pci/devices/*/resource*`
   returns `0xFFFFFFFF` for all reads when the device is bound to
   `vfio-pci`. The correct path is through the VFIO device fd.

4. **Device needs init sequence** — the AKD1000 requires a
   reset-and-enable sequence before inference. This was previously
   handled by the kernel driver. The VFIO path needs to implement
   this (write to CONTROL register, poll STATUS for READY bit).

## Code Changes

- `discovery.rs`: Added `discover_via_sysfs()` fallback for VFIO-bound devices
- `mmio.rs`: Fixed `Bar` enum to use correct VFIO region indices (0, 2, 4)
- `sram.rs`: Fixed BAR1 sysfs resource index from 1 to 2
- `probe_vfio.rs`: New binary for VFIO-native hardware probing

## Next Steps

- Implement device init sequence via VFIO (reset + enable)
- Create preserve demo binaries that run through software backend
- Port NP SRAM layout discovery from sysfs path to VFIO path
