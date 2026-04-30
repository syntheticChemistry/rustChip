# Sovereign Boot — Glowplug VFIO Lifecycle Management

**Central idea:** The NPU should manage its own hardware lifecycle — bind, warm
boot, tear down — without relying on external orchestrators, kernel modules, or
the vendor SDK. The glowplug module provides this capability, absorbed from
coralReef's ember/glowplug VFIO passthrough architecture.

---

## What Sovereign Boot Means

In the vendor workflow, bringing up an AKD1000 requires:
1. Load the `akida_pcie` kernel module
2. Wait for `/dev/akida0` to appear
3. Initialize via the Python SDK or C++ engine

With sovereign boot (glowplug), the flow is:
1. VFIO bind (one-time, or via udev rule)
2. `glowplug::warm_boot()` — pure Rust, userspace, no kernel module
3. Device is ready for inference

The warm boot cycle can be repeated without unbinding — useful for recovery
from firmware hangs, power state transitions, or multi-tenant resets.

---

## Architecture

```
Application
    │
    ▼
glowplug::DeviceLifecycle
    ├── bind()      — bind PCIe BDF to vfio-pci driver
    ├── warm_boot() — reset + initialize via BAR0 registers
    ├── health()    — check device readiness and NP availability
    └── teardown()  — clean shutdown, unmap BARs, release VFIO container
         │
         ▼
    VfioBackend [HW]  ←→  SoftwareBackend [SW]
```

The glowplug module operates below the `NpuBackend` trait. It provides the
device to `VfioBackend`; the backend provides inference to the application.

---

## Lineage

This module is derivative of coralReef's `coral-ember` and `coral-glowplug`
crates, which manage GPU VFIO passthrough for sovereign GPU compute. The NPU
adaptation is self-contained within rustChip but retains the full scyBorg
triple-copyleft license under the lineage principle.

The original ember/glowplug pattern:
- `ember` — immortal fd holder, passing duplicates via `SCM_RIGHTS`
- `glowplug` — device lifecycle: bind, reset, initialize, health check

rustChip absorbs the lifecycle pattern (glowplug) directly. The fd-sharing
pattern (ember) is noted for future multi-client NPU access but not yet needed.

---

## Usage

```rust
use akida_driver::glowplug::{DeviceLifecycle, GlowplugConfig};

let config = GlowplugConfig {
    bdf: "0000:e2:00.0".parse()?,
    iommu_group: 92,
    ..Default::default()
};

let lifecycle = DeviceLifecycle::new(config)?;
lifecycle.warm_boot()?;

let backend = lifecycle.into_vfio_backend()?;
let result = backend.infer(&model, &input)?;
```

Or via the CLI:

```bash
cargo run --bin warm_boot
```

---

## Relationship to Other Systems

| System | Interaction |
|--------|------------|
| `VfioBackend` | Glowplug provides the initialized device; VfioBackend uses it for inference |
| `multi_tenancy` | Warm boot resets all NP slots; multi-tenancy re-partitions after boot |
| `online_evolution` | Evolution runs on a booted device; glowplug handles recovery if device hangs |
| `hw_sw_comparison` | HW path requires glowplug boot; SW path is always available |
| coralReef | Upstream source; rustChip's copy is self-contained |
