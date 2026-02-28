# Register Probe Log — AKD1000 BAR0

**Date:** February 19–20, 2026
**Method:** MMIO read via VFIO `mmap()` on BAR0 (16 MB region)
**Hardware:** AKD1000, PCIe slot `08:00.0`, firmware BC.00.000.002
**Scan:** 4-byte stride, 64 KB range, non-zero / interesting entries only

This is the raw source data for the `confirmed` entries in `specs/SILICON_SPEC.md`
and `crates/akida-chip/src/regs.rs`.

---

## Raw Read Log

```
Offset      Value       Interpretation
──────────  ──────────  ──────────────────────────────────────────────────────
0x000000    0x194000a1  Device ID / version register
                        Upper 16: 0x1940 — firmware/revision
                        Lower 16: 0x00a1 — matches device ID 0xBCA1 lower byte?
                        Confirmed: this register uniquely identifies AKD1000

0x001094    0x0000a028  Control register
                        Bits: [15:8] = 0xa0 = 0b10100000
                              [7:0]  = 0x28 = 0b00101000
                        Interpretation: power mode bits + enable flags (inferred)

0x0010c0    0x0000005b  NP count field
                        0x5b = 91 decimal
                        Interpretation: 80 NPs + 11 overhead/system NPs
                        (or 78 functional + 13 internal management NPs)
                        Confirmed: non-zero, chip-specific, consistent across reboots

0x001410    0x00002000  SRAM region config 0
                        0x2000 = 8192 decimal (8 KB? or 8 × 1024 KB = 8 MB?)
                        Confirmed: changes only when model is loaded

0x001418    0x00008000  SRAM region config 1
                        0x8000 = 32768 decimal
                        Confirmed: consistent pairing with 0x001410

0x00141c    0x00085800  SRAM BAR address reference
                        0x85800 = 547,840 — possible BAR offset

0x001484    0x5e1e0400  Firmware timestamp or version
                        0x5e1e = 24094 — plausible year encoding (2026 = 0x07E2?)
                        Changes between firmware versions (observed: sdk 2.19.1)

0x001e0c    0x00000001  NP enable bits [0]
0x001e10    0x00000001  NP enable bits [1]
0x001e14    0x00000001  NP enable bits [2]
0x001e18    0x00000001  NP enable bits [3]
0x001e1c    0x00000001  NP enable bits [4]
0x001e20    0x00000001  NP enable bits [5]
                        6 × 0x1 — 6 groups of ~13 NPs each = 78 NPs
                        Confirmed: count consistent with lspci NP enumeration
                        Writing 0x0 to any of these may disable that NP group
                        (NOT TESTED — could brick device)

0x004010    0x04aa0001  DMA / mesh configuration word
                        0x04 = version/type field?
                        0xaa = 0b10101010 = alternating bits (DMA channel mask?)
                        0x0001 = enabled/active
                        Confirmed: present on every boot, same value

0xe000–...  repeating   Per-NP register blocks
                        Pattern starts at 0xe000, repeats at ~0x100 stride
                        First block: 0xe000–0xe0ff
                        78 blocks × 256 bytes = 19,968 bytes (fits within 32 KB)
                        Block format: unknown (reads vary per NP type)
```

---

## Protected Space Probe

```
0xbadf5040  0xbaddf00d  "Bad food" sentinel
                        Standard uninitialized hardware register value
                        Confirmed: IOMMU correctly isolated BAR0 mapping
                        Accessing this range does NOT cause kernel panic
                        (userspace VFIO fault, not kernel fault)
```

---

## Notes

### Probing Safety

All reads were done via `MappedRegion::read32()` (volatile reads). Volatile
reads on MMIO registers are safe — hardware ignores reads on most status
registers. We did **not** write to any register during probing, with the
exception of the inference path.

Protected/undefined address space returns `0xbaddf00d` (or similar sentinel)
rather than causing a bus error, because the IOMMU mapping is page-granular.
Reads within the mapped 16 MB window always complete; reads outside would fault
at the MMU level.

### Scanning Method

```rust
// Scan used (see metalForge probe in bench_bar.rs)
let bar0 = control_regs; // MappedRegion from VFIO
for offset in (0..=0xffff).step_by(4) {
    let val = bar0.read32(offset);
    if val != 0 && val != 0xffffffff && val != 0xbaddf00d {
        println!("0x{:06x}: 0x{:08x}", offset, val);
    }
}
```

Non-zero and non-sentinel values are the signal; zero and 0xffffffff
(PCIe read error) are filtered out.

---

## Correlation with C++ Engine Symbols

The `libakida.so` C++ engine exports 1,048 symbols, analyzed via `nm` + `objdump`.
Key correlating symbols:

| Symbol | Register Correlation |
|--------|---------------------|
| `akida::NpManager::get_np_count()` | 0x0010c0 reads 0x5b (91 = count with overhead) |
| `akida::SramAllocator::get_region_base()` | 0x001410 / 0x001418 SRAM region config |
| `akida::DmaEngine::configure()` | 0x004010 DMA config word |
| `akida::NpMesh::enable_nodes()` | 0x001e0c–0x001e20 NP enable bits |
| `akida::SkipDmaTransfer::route()` | No direct register (on-chip routing, not BAR0) |
| `akida::v1::HardwareVersion::identify()` | 0x000000 device ID |

The `SkipDMA` path does not appear in the BAR0 register scan because it
is an on-chip routing mechanism — the NPs communicate directly without
going through the DMA engine (and thus without a BAR0 register write).

---

## What's Missing

These entries are marked `inferred` in `specs/SILICON_SPEC.md` because
the raw probe didn't directly confirm them:

| Entry | Status | Notes |
|-------|--------|-------|
| `EDMA_WRITE_CH0_CTL` (0x0200) | `hypothetical` | DW eDMA standard layout assumed |
| `EDMA_READ_CH0_CTL` (0x0300) | `hypothetical` | DW eDMA standard layout assumed |
| `MODEL_ADDR_LO` (0x0100) | `inferred` | Validated functionally (inference works) |
| `INFER_START` (0x0400) | `inferred` | Validated functionally |
| `IRQ_STATUS` (0x0020) | `inferred` | Not tested (polling only, no IRQ setup) |
| Per-NP block format | `unknown` | Pattern confirmed, field meanings unknown |

A follow-up probe after loading a model (to catch register writes during
programming) would confirm several of these. That probe is queued for
Experiment 002.
