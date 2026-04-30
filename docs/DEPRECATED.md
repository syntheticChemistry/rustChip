# Deprecated: C kernel module

The files at the root of this repository (`akida-pcie-core.c`, `Makefile`,
`install.sh`, `build_kernel_w_cma.sh`) are the original BrainChip C PCIe driver.

**They are kept for reference only. All active development is in `crates/`.**

---

## Why deprecated

| Problem | C module | Rust replacement |
|---------|----------|-----------------|
| Kernel version ceiling | Requires rebuilding per kernel; datasheet states "no plans for updates past 6.8" | VFIO-based Rust driver works without a kernel module |
| Python SDK dependency | C module required by Python SDK (MetaTF) | Pure Rust — no Python, no C++ SDK, no MetaTF |
| Permissions | Requires `chmod 666 /dev/akida*` after every boot | VFIO uses IOMMU for secure isolation per-process |
| Build toolchain | Needs kernel headers, gcc, make | `cargo build --release` |
| Hardware discovery | None — assumes `/dev/akida0` exists | Runtime sysfs scan, no hardcoded paths |
| Portability | Kernel module, Linux only | Rust crate, runs anywhere the hardware exists |

---

## Migration path

```
Before (C module + Python SDK):
  1. make && sudo insmod akida-pcie.ko
  2. sudo chmod 666 /dev/akida0
  3. python3 -c "import akida; model.map(); model(input)"

After (pure Rust):
  Option A — VFIO (no kernel module):
    akida bind-vfio 0000:a1:00.0      # once, requires root
    cargo run --bin enumerate          # no root

  Option B — kernel module fallback:
    sudo insmod akida-pcie.ko          # as before
    cargo run --bin enumerate          # Rust driver opens /dev/akida0
```

---

## What the C module does (for reference)

`akida-pcie-core.c` implements a minimal PCIe driver that:
1. Registers with `pci_register_driver` for vendor `1e7c`, device `bca1`
2. Calls `pci_enable_device`, `pci_set_master`, `pci_request_regions`
3. Wraps the DesignWare eDMA controller (`dw_edma`) for DMA transfers
4. Creates `/dev/akida{N}` character devices
5. Exposes `read`/`write` syscalls that trigger DMA

The Rust VFIO backend (`crates/akida-driver/src/vfio/mod.rs`) replaces
all of this with userspace equivalents via the Linux VFIO/IOMMU framework.
The glowplug module (`crates/akida-driver/src/glowplug.rs`) provides sovereign
device lifecycle management — bind, warm boot, teardown — without this module.
