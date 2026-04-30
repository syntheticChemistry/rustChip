# Glowplug — VFIO Device Lifecycle for rustChip

**Date:** 2026-04-30
**Status:** Absorbed into rustChip, fully standalone

## Overview

rustChip's `glowplug` module is a standalone, NPU-focused VFIO device
lifecycle manager. It was absorbed from coralReef's `coral-ember` and
`coral-glowplug` crates to make rustChip fully self-contained.

For the full-scale GPU + NPU orchestrator with SCM_RIGHTS fd passing,
ring persistence, multi-vendor lifecycle, and systemd integration, see:
`primals/coralReef/crates/coral-ember/` and `coral-glowplug/`.

## Architecture

```
akida-driver::glowplug
├── sysfs       — safe sysfs reads/writes with D-state isolation
├── lifecycle   — vendor-specific NPU lifecycle hooks
├── swap        — driver bind/unbind orchestration
└── sovereign   — warm boot: firmware init via kernel driver, then VFIO
```

## The Problem

The AKD1000's BAR0 exposes 4 MB of raw NP SRAM when bound to `vfio-pci`
without prior firmware initialization. The "registers" in `BEYOND_SDK.md`
are SRAM mailbox addresses that only have their expected values when the
on-chip firmware is running. The firmware is started by the `akida_pcie`
kernel driver's probe routine.

## The Solution: Sovereign Boot

```
┌─────────────────┐
│  Device on       │
│  vfio-pci       │  BAR0 = raw SRAM (firmware dead)
│  (cold)          │
└────────┬────────┘
         │  1. probe BAR0 → firmware dead?
         │  2. unbind from vfio-pci
         │  3. bind to akida_pcie
         ▼
┌─────────────────┐
│  Device on       │
│  akida_pcie     │  Firmware boots, registers go live
│  (warm)          │  /dev/akida0 appears
└────────┬────────┘
         │  4. settle (3 seconds)
         │  5. disable reset_method
         │  6. unbind akida_pcie
         │  7. bind vfio-pci
         ▼
┌─────────────────┐
│  Device on       │
│  vfio-pci       │  BAR0 = firmware mailbox
│  (warm)          │  DEVICE_ID = 0x194000a1
└─────────────────┘
```

## Usage

### Binary

```bash
cargo run --bin warm_boot_akida
cargo run --bin warm_boot_akida -- 0000:e2:00.0
RUST_LOG=debug cargo run --bin warm_boot_akida
```

### Library

```rust
use akida_driver::glowplug;

// Auto-detect lifecycle and run sovereign boot
let result = glowplug::sovereign_boot("0000:e2:00.0");
println!("Firmware alive: {}", result.firmware_alive);

// Or with a custom lifecycle
use akida_driver::glowplug::BrainChipLifecycle;
let lc = BrainChipLifecycle { device_id: 0xbca1 };
let result = glowplug::sovereign::sovereign_boot_with_lifecycle("0000:e2:00.0", &lc);

// Driver swap (standalone)
use akida_driver::glowplug::swap;
let outcome = swap::swap_to_driver("0000:e2:00.0", "vfio-pci", &lc)?;
```

## D-state Protection

Sysfs writes to `driver/unbind` and `bind` can enter uninterruptible
kernel sleep (D-state). A thread in D-state cannot be killed — even
SIGKILL is deferred. The `sysfs::sysfs_write()` function spawns a
short-lived child process with a 10-second timeout. If the child enters
D-state, it's killed and the calling thread remains responsive.

Direct sysfs writes (for `power/control`, `reset_method`, etc.) use
`sysfs::sysfs_write_direct()` which is synchronous — these config-space
attributes never enter D-state.

## Hardware / Software Separation

The glowplug subsystem only operates on **hardware** backends. It has
no interaction with `SoftwareBackend`. The `BackendType::is_hardware()`
/ `is_software()` boundary is structural — `Auto` selection never
crosses from hardware to software domain.

## Generic NPU Support

Adding a new NPU vendor:

1. Implement `NpuLifecycle` for the new hardware
2. Add detection in `lifecycle::detect_lifecycle()` by PCI vendor ID
3. Set `native_driver_module()` and `native_driver_sysfs()`
4. The swap and sovereign boot systems work automatically

```
                  NpuLifecycle trait
                       │
       ┌───────────────┼───────────────┐
       │               │               │
  BrainChip       GenericNpu       [YourNpu]
  Lifecycle       Lifecycle        Lifecycle
       │               │               │
  akida_pcie      (configurable)   your_driver
```

## coralReef Provenance

| rustChip module | Absorbed from |
|----------------|---------------|
| `glowplug::sysfs` | `coral-ember/src/sysfs.rs` |
| `glowplug::lifecycle` | `coral-ember/src/vendor_lifecycle/{mod,brainchip,generic,detect}.rs` |
| `glowplug::swap` | `coral-ember/src/swap/{mod,swap_bind}.rs` |
| `glowplug::sovereign` | `coral-glowplug/src/sovereign.rs` |

### What rustChip absorbs
- sysfs read/write with D-state protection
- NPU vendor lifecycle hooks (BrainChip + Generic)
- Driver swap orchestration (unbind → override → bind → settle)
- Sovereign boot pattern (warm cycle → VFIO takeover)

### What stays in coralReef
- SCM_RIGHTS fd passing (ember daemon)
- Ring/mailbox persistence across restarts
- Multi-vendor GPU lifecycle (NVIDIA Volta, AMD Vega20, Intel Xe)
- DRM isolation and Xorg/udev coordination
- HBM2 training recipes and golden state capture
- systemd integration (watchdog, socket activation)
- Circuit breaker and health monitoring
