# Sovereign Driver Roadmap — Phase A through E

**Context**: The BrainChip AKD1000/AKD1500 has a Linux PCIe driver
(`akida-pcie-core.c`) that must be rebuilt per kernel version. The AKD1500
datasheet explicitly states "no plans for updates past Linux 6.8."

This roadmap documents the progression from vendor-SDK-dependent to fully
sovereign Rust access — from Python wrapper to Rust kernel module.

---

## Phase Overview

```
Phase A: Python SDK → Rust FFI wrapper       [external, not in this repo]
Phase B: C++ Engine → Rust FFI (libakida.so) [external, not in this repo]
Phase C: Direct ioctl/mmap on /dev/akida0    ✅ complete (Feb 26, 2026)
Phase D: Pure Rust VFIO driver               ✅ active (SRAM access complete)
Phase E: Rust akida_pcie kernel module        🔲 queued
```

---

## Phase A — Python SDK → Rust FFI Wrapper

**Status**: Complete (external project)
**What it was**: Called into BrainChip's Python `akida` package via PyO3.
**Why it mattered**: First measurement of actual hardware behavior.
**What we learned**: The SDK adds ~15% overhead; hardware is more capable than documented.

---

## Phase B — C++ Engine → Rust FFI (libakida.so)

**Status**: Complete (external project)
**What it was**: Called into BrainChip's C++ engine (`libakida.so`) directly,
bypassing the Python layer.
**Why it mattered**: C++ engine exports 1,048 symbols; analyzed with `nm` and `objdump`.
**What we learned**:
- `program_external()` takes raw `program_info` bytes + IOVA address
- `set_variable()` updates weights without full reprogram (~14 ms overhead)
- `SkipDMA` is an internal routing mechanism for FC chain merging
- Three `akida::v1`, `akida::v2`, `akida::pico` hardware variants in engine
- 51-bit threshold SRAM exists (not 4-bit as documented)

---

## Phase C — Direct ioctl/mmap on /dev/akida0

**Status**: ✅ Complete — February 26, 2026
**What it is**: Opened `/dev/akida*` directly, without SDK or FFI.
Read/write syscalls drive DMA transfers. MMIO via `/dev/mem` for register access.
**Implementation**: `backends/kernel.rs` in this repo.
**Requires**: C `akida_pcie` kernel module loaded, `/dev/akida0` present.
**Results**:
- 37 MB/s DMA throughput (sustained)
- 54 µs / 18,500 Hz inference
- Weight mutation without reprogram confirmed
- `program_external()` injection proven — full bypass of SDK

---

## Phase D — Pure Rust VFIO Driver

**Status**: ✅ Active — this repository, primary backend. SRAM access complete.
**What it is**: Linux VFIO/IOMMU provides userspace PCIe device access.
No kernel module in the data path. Pure Rust from `open("/dev/vfio/N")` onward.

**Implementation in this repo**:
- `crates/akida-driver/src/vfio/mod.rs` — full VFIO implementation
  - Container, group, device management
  - DMA buffer lifecycle (alloc → mlock → MAP_DMA → use → UNMAP_DMA → free)
  - BAR0 MMIO via `mmap()` on VFIO region
  - **BAR1 SRAM mapping** via `map_bar1()` — direct read/write to all on-chip SRAM
  - Register polling loop with yield for inference completion
  - Inference, model load, reservoir load, power measurement
- `crates/akida-driver/src/sram.rs` — `SramAccessor` for userspace BAR0/BAR1 access
- `crates/akida-driver/src/mmio.rs` — BAR memory mapping
- `crates/akida-driver/src/capabilities.rs` — `from_bar0()` runtime discovery
- `crates/akida-driver/src/backend.rs` — `NpuBackend` SRAM methods
- `crates/akida-driver/src/tenancy.rs` — multi-tenant NP slot management
- `crates/akida-driver/src/evolution.rs` — online weight evolution
- `crates/akida-driver/src/puf.rs` — hardware PUF fingerprinting
- `crates/akida-driver/src/sentinel.rs` — domain drift detection
- `crates/akida-cli/src/main.rs` — `bind-vfio` and `unbind-vfio` subcommands

**Phase D.5 — SRAM Infrastructure (complete)**:
- `SramAccessor`: userspace BAR0 register dump + BAR1 SRAM read/write/probe
- `VfioBackend::map_bar1()`: VFIO-backed BAR1 SRAM access
- `Capabilities::from_bar0()`: runtime NP count, SRAM size, mesh topology
- `NpuBackend::verify_load()`: model integrity via SRAM readback
- `NpuBackend::mutate_weights()`: zero-DMA direct weight patches
- `NpuBackend::read_sram()`: raw SRAM reads for diagnostics
- `probe_sram` binary: 3-mode SRAM diagnostic tool
- `bench_exp002_tenancy --hw`: Phase 2 SRAM isolation verification

**Setup**: One-time per machine (IOMMU enable + vfio-pci bind). After that:
no root, no kernel module, no Python.

**Known gaps**:
- IRQ-based completion (currently polling) — `VFIO_DEVICE_SET_IRQS` ready to use
- Scatter-gather DMA for large payloads
- MSI-X interrupt vectors

---

## Phase E — Rust akida_pcie Kernel Module

**Status**: 🔲 Queued
**What it will be**: A pure Rust replacement for `akida-pcie-core.c` using
the Linux kernel's Rust bindings (`rust/` in the kernel tree, stable since 6.1).

**Why Phase D isn't enough**:
- VFIO requires IOMMU hardware (not all systems have it enabled)
- VFIO requires one-time root setup per machine
- Some embedded/edge deployments need a traditional kernel module interface
- A Rust kernel module can register as a proper PCIe driver (`pci_register_driver`)
  and create `/dev/akida*` without the Python or C++ stack

**Why Phase E beats the C module**:
- Rust kernel module uses stable kernel Rust API (`rust/kernel/` bindings)
- These bindings are committed to not breaking between kernel versions
- The C module must be rebuilt per kernel; the Rust one should not
- Memory safety: no use-after-free, no NULL dereference in the driver
- Single binary: `akida_pcie.ko` → `akida_pcie_rs.ko` → drop-in replacement

**Sketch**:
```rust
// In future crates/akida-kmod/
use kernel::prelude::*;
use kernel::pci::{self, Device as PciDevice};

module! {
    type: AkidaDriver,
    name: "akida_pcie_rs",
    author: "ecoPrimal",
    license: "GPL v2",
}

impl pci::Driver for AkidaDriver {
    fn probe(dev: &pci::Device, _id: &pci::DeviceId) -> Result<Self::Data> {
        dev.enable_device()?;
        dev.set_master();
        dev.request_mem_regions()?;
        // BAR0 mmap, DW eDMA setup, /dev/akidaN creation
        todo!()
    }
}
```

**What BrainChip could do to accelerate this**: Publish the DW eDMA register
map (subset of the DesignWare PCIe eDMA databook) under NDA or open license.
Our probed register map is functional but incomplete; the confirmed offsets
in `crates/akida-chip/src/regs.rs` cover the DMA path but not all features.

---

## Beyond Phase E — Future Directions

### 5.1 AKD1500 native support

AKD1500 adds SPI, GPIO, hardware SLEEP, PCIe x2, BGA169 package. None of
these require code changes in the core driver (PCIe x2 is transparent; SPI/GPIO
use separate kernel interfaces). The single change: device ID in `pcie.rs`.

### 5.2 On-chip recurrent execution (Phase F concept)

Discovery 6 confirms weight mutation at ~14 ms. If BrainChip opens the
`akida_learn_on_chip` symbol path, the reservoir update step could execute
on-chip without PCIe round-trip for the weight matrix.

### 5.3 P2P DMA: GPU → NPU without CPU

Both GPU (NVIDIA) and NPU (AKD1000) are PCIe devices. Peer-to-peer DMA would
allow GPU output to flow directly to NPU input without CPU copy:

```
GPU computes (WGSL shader) → GPU result buffer in BAR → P2P DMA → NPU input IOVA
```

Requirements: Both devices in same IOMMU group (or IOMMU bypass), NVIDIA
`nv_peer_mem` or RDMA-capable driver, VFIO BAR peer mapping. This is
documented in detail in `../whitePaper/explorations/GPU_NPU_PCIE.md`.

### 5.4 Rust all the way down

The long-term vision: Rust from application code to silicon. Current stack:

```
Application (Rust) → akida-driver (Rust) → VFIO (Linux kernel C) → IOMMU (hw) → NPU (silicon)
```

After Phase E:
```
Application (Rust) → akida-driver (Rust) → akida_pcie_rs.ko (Rust kernel) → PCIe (hw) → NPU (silicon)
```

The only non-Rust component is the Linux kernel's PCIe infrastructure itself.
As the kernel Rust API matures, that shrinks further.
