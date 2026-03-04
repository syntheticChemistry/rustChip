# Rust All the Way Down — From Application to Silicon

**Status:** Analysis. Phases A–D active. Phase E queued.
**Date:** February 27, 2026

---

## The Vision

```
2024 (start):
  Python application
    → MetaTF (Python SDK)
      → libakida.so (C++ FFI)
        → akida_pcie.ko (C kernel module, per-kernel rebuild)
          → PCIe → AKD1000

2026 (current, Phase D):
  Rust application
    → akida_driver::InferenceExecutor (this repo)
      → VfioBackend (pure Rust ioctls + mmap)
        → SramAccessor (BAR0/BAR1 direct read/write — full on-chip memory access)
        → /dev/vfio/{group} (Linux VFIO — no C in data path)
          → IOMMU → PCIe → AKD1000

2026–2027 (Phase E):
  Rust application
    → akida_driver::InferenceExecutor
      → KernelBackend → /dev/akida0
        → akida_pcie_rs.ko (Rust kernel module, stable bindings)
          → PCIe → AKD1000

Future (Phase F+):
  Rust application
    → Rust all the way through
      → AKD1000 / AKD1500 (or die-to-die integrated silicon)
```

The current Phase D VFIO stack has no C in the inference data path after
kernel-level IOMMU setup. VFIO ioctls (`libc::ioctl`) are the one remaining
non-Rust surface — and that's a language boundary, not a safety issue.

Phase D now includes the **SRAM layer**: `VfioBackend::map_bar1()` exposes BAR1
(NP mesh / SRAM window) for direct read/write. The driver has full access to
all on-chip memory — BAR0 registers and BAR1 SRAM — enabling model verification,
direct weight mutation, and zero-DMA online learning.

Phase E converts the kernel module to Rust. After that: the entire stack from
application to device interrupt handler is Rust.

---

## Phase E: Rust Kernel Module

### Why the C module breaks

`akida-pcie-core.c` uses the standard Linux PCIe driver model:

```c
static const struct pci_device_id akida_pcie_ids[] = {
    { PCI_DEVICE(0x1E7C, 0xBCA1) }, // AKD1000
    { 0 }
};
MODULE_DEVICE_TABLE(pci, akida_pcie_ids);
```

This works until a kernel update changes an internal API. The AKD1500 datasheet
states explicitly: "Linux kernel support ends at 6.8, no plans for newer kernels."

Every Linux major release (6.9, 6.10, 6.11, 6.12 LTS, 6.13, 6.14...) requires
either a vendor patch or a user rebuild. On Ubuntu/Fedora with automatic security
updates, a kernel upgrade silently breaks the driver until the user rebuilds.

### Why the Rust module doesn't break

The Linux kernel's Rust API (`rust/kernel/`) provides abstractions for PCIe
drivers that are committed to stability:

```rust
// The Rust kernel API contract:
use kernel::pci;

// pci::Driver is a stable abstraction
// It won't change signatures between kernel versions
// The underlying C PCIe infrastructure changes, but the Rust wrapper absorbs it
```

This is the same reason the kernel team added Rust support: to provide a
stable, high-level API for driver authors that survives internal refactors.

### Sketch of akida_pcie_rs.ko

```rust
// crates/akida-kmod/src/lib.rs (future — Phase E)
// Requires: nightly Rust, CONFIG_RUST=y, CONFIG_RUST_PHYLIB_ABSTRACTIONS or similar

#![no_std]
#![feature(allocator_api)]

use kernel::prelude::*;
use kernel::pci::{self, Device as PciDevice, DeviceId, Driver};
use kernel::io_mem::IoMem;
use kernel::file::{File, Operations};
use kernel::miscdev::Registration;

// Device state
struct AkidaDevice {
    bar0: IoMem<{ akida_chip::bar::bar0::SIZE as usize }>,
    _dma: kernel::dma::Allocation,
}

// PCIe driver registration
struct AkidaDriver;

impl Driver for AkidaDriver {
    type Data = Box<AkidaDevice>;

    fn probe(dev: &PciDevice, _id: &DeviceId) -> Result<Self::Data> {
        dev.enable_device_mem()?;
        dev.set_master();
        let bar0 = dev.iomap_region(0, "akida BAR0")?;

        // Read device ID register — confirm 0x194000a1
        let device_id = bar0.readl(akida_chip::regs::DEVICE_ID);
        pr_info!("AKD1000: device_id={:#010x}\n", device_id);

        Ok(Box::try_new(AkidaDevice { bar0, _dma: todo!() })?)
    }
}

// Character device operations — /dev/akida0
impl Operations for AkidaDriver {
    fn read(/* ... */) -> Result<usize> {
        // DMA read from NPU output buffer
        todo!()
    }
    fn write(/* ... */) -> Result<usize> {
        // DMA write to NPU input/model buffer
        todo!()
    }
}
```

The `akida_chip` crate (no_std compatible because it's pure constants) would
be available to the kernel module directly. The silicon model and the kernel
driver share the same constant definitions.

### Making akida-chip no_std

`akida-chip` currently requires `std` only for the test framework. Converting
to `no_std` with `alloc` is straightforward:

```rust
// crates/akida-chip/src/lib.rs
#![no_std]
// No alloc needed — all types are Copy or 'static slices
```

The `program.rs` module uses `Vec<u8>` — that would need `alloc::vec::Vec`.
All constants and enums in `pcie.rs`, `bar.rs`, `regs.rs`, `mesh.rs` are
already `no_std` compatible.

---

## Phase F: On-Chip Learning Register Path

Discovery 6 shows that `set_variable()` updates weights without full reprogram
(~14 ms overhead, DMA-based). This requires:
1. DMA the weight matrix to an IOVA address
2. Write that IOVA to the weight update registers
3. Trigger the weight update

If BrainChip opens the on-chip learning path (the `akida_learn_on_chip` symbol
in the C++ engine, currently unexported), the reservoir update step could
execute **on-chip** without PCIe round-trip for the weight matrix update.

This would transform the ESN learning loop:

```
Current (Phase D):
  Reservoir state r(t) → PCIe → CPU → matrix mult → PCIe → SRAM update
  Cost: ~14 ms per weight update

Phase F (on-chip):
  Reservoir state r(t) → on-chip learning registers
  Cost: ~0.7 µs (chip compute, no PCIe)
```

For online evolutionary learning (BingoCube / genetic reservoir search),
this would increase the generation rate from ~136 gen/s to potentially
~100,000 gen/s — 735× improvement.

---

## The Silicon Goal

The full Rust vision is not about replacing C for its own sake. It's about
building a software substrate that:

1. **Survives hardware generations** — register maps evolve; Rust's type system
   catches API mismatches at compile time, not at runtime on production hardware

2. **Enables capability-based dispatch** — the same `akida_chip::regs` and
   `akida_chip::mesh` constants describe both the userspace VFIO driver and
   the kernel module; silicon knowledge lives in one place

3. **Makes silicon capabilities visible** — the `confirmed`/`inferred`/`hypothetical`
   labels in `specs/SILICON_SPEC.md` are a living contract between software and
   silicon; as BrainChip confirms or corrects them, the provenance is tracked
   in git history

4. **Removes the vendor lock-in cliff** — the AKD1500 datasheet's "no plans
   past 6.8" only matters if your driver is C. A Rust VFIO driver already
   doesn't have that ceiling. A Rust kernel module won't either.

The ultimate question is not "can we write Rust drivers?" (we can, and have).
The question is: "can the silicon evolve faster than the software stack
that exposes it?" With a pure Rust stack, the answer is yes — because every
silicon update maps to a constant change or a new variant in `akida-chip`,
and the rest of the stack recompiles cleanly.

---

## What Would Accelerate This

| Action | Impact |
|--------|--------|
| BrainChip publishes DW eDMA register offsets | Completes `specs/SILICON_SPEC.md`; enables full eDMA programming in Phase E |
| BrainChip confirms `inferred` register entries | Removes the 8 entries in `regs.rs` labeled as uncertain |
| On-chip learning API documentation | Enables Phase F (735× gen/s improvement) |
| AKD1500 hardware sample | Validates Phase D on AKD1500; confirms BAR layout and register map transfer |
| Akida IP licensing discussion | Enables die-to-die integration analysis; see `GPU_NPU_PCIE.md` |

None of these are required for the current production system. They are
optimizations on an already-functional baseline.
