# akida-chip

Pure silicon model for the BrainChip Akida AKD1000 / AKD1500 neuromorphic processor.

**Zero dependencies. Zero hardware access. Zero unsafe code.**

This crate models what the chip IS — register addresses, BAR layout, NP mesh
topology, PCIe identifiers, SRAM geometry, and the FlatBuffer program format.
Everything here was established by direct hardware probing on real AKD1000 silicon.

## What's in the box

| Module | Contents |
|--------|----------|
| `pcie` | Vendor/device IDs, PCIe Gen2 x1 timing constants, chip variant detection |
| `bar` | BAR layout — BAR0 (16 MB control), BAR1 (16 GB mesh window), BAR3 (32 MB) |
| `regs` | BAR0 register map — all offsets and bit definitions from hardware probing |
| `mesh` | NP mesh topology — 5×8×2 geometry, 78 functional NPs, SkipDMA routing |
| `program` | FlatBuffer `program_info` / `program_data` format |
| `sram` | BAR1 address layout, per-NP SRAM offsets, probe point generation |

## Usage

```rust
use akida_chip::pcie::{BRAINCHIP_VENDOR_ID, lspci_filter};
use akida_chip::regs;
use akida_chip::mesh::Mesh;

// Know the silicon
let mesh = Mesh::akd1000();
assert_eq!(mesh.total_nps(), 80);
assert_eq!(mesh.functional_nps(), 78);

// Know the registers
let status = regs::STATUS;        // BAR0 offset
let busy = regs::status::BUSY;    // bit mask
let ready = regs::status::READY;  // bit mask
```

## Measured results (from real AKD1000 hardware)

| Property | Value |
|----------|-------|
| Neural Processors | 80 slots, 78 functional (2 disabled) |
| NP types | CNP1×78, CNP2×54, FNP2×4, FNP3×18 (overlapping capabilities) |
| On-chip SRAM | 8 MB (64-bit filter, 51-bit threshold, 32-bit event) |
| PCIe | Gen2 x1 (500 MB/s theoretical, 37 MB/s sustained DMA measured) |
| Inference | 54 µs / 18,500 Hz per frame |
| Power | ~1W (chip inference below measurement floor: 1.4 µJ) |

## No-std compatible

This crate uses `#![forbid(unsafe_code)]` and has no dependencies.
It compiles on every target Rust supports — Linux, Windows, macOS, WASM,
RISC-V, ARM, bare metal. The silicon model goes wherever Rust goes.

## Part of rustChip

This crate is one layer of the [rustChip](https://github.com/syntheticChemistry/rustChip)
sovereign neuromorphic stack:

```
    chipMonk        — organism agent riding on rustChip
       ↓
    akida-models    — model loading, ONNX import, inference
       ↓
    akida-driver    — VFIO, kernel, software backends
       ↓
 →  akida-chip      — YOU ARE HERE: pure silicon model
       ↓
    AKD1000 silicon — 78 NPs, 8 MB SRAM, event-driven spikes
```

Turtles all the way down.

## License

AGPL-3.0-or-later — part of the [ecoPrimals](https://primals.eco) organism.
