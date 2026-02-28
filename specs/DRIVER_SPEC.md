# Driver Architecture Specification

**Date**: February 27, 2026
**Crate**: `akida-driver` v0.1.0
**Target**: AKD1000 / AKD1500, Linux x86-64, aarch64

---

## 1. Backend Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Application                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │  akida_driver::DeviceManager::discover()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  DeviceManager — runtime sysfs scan                             │
│  Discovers /dev/akida* and PCIe vendor:device entries           │
│  Returns Vec<DeviceInfo> — no hardware assumed                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  open_first() / open(idx)
                            ▼
┌──────────────────────────────────┐
│  select_backend() auto selection │
│  (BackendSelection::Auto)        │
└────────┬──────────┬──────────────┘
         │          │          │
         ▼          ▼          ▼
┌────────────┐ ┌──────────┐ ┌────────────────┐
│  Kernel    │ │  VFIO    │ │  Userspace     │
│  Backend   │ │  Backend │ │  Backend       │
│            │ │          │ │                │
│ /dev/akida*│ │ /dev/vfio│ │ BAR mmap only  │
│ read/write │ │  +IOMMU  │ │ no DMA, probing│
│ syscalls   │ │  DMA     │ │ development    │
└────────────┘ └──────────┘ └────────────────┘
  fallback       primary      last resort
  (C module      (no C        (dev/debug)
   loaded)        module)
```

**Default auto selection order:**
1. `KernelBackend` — fastest if C module already loaded
2. `VfioBackend` — VFIO IOMMU binding (requires one-time `akida bind-vfio` setup)
3. `UserspaceBackend` — BAR mmap (development only; no inference DMA)

---

## 2. VFIO Backend — Primary Path

VFIO (Virtual Function I/O) provides device access without a kernel driver.
Linux IOMMU maps device address space into userspace. No C code runs in the
data path after setup.

### Requirements

```
1. IOMMU enabled in BIOS:
     Intel: intel_iommu=on iommu=pt (add to GRUB_CMDLINE_LINUX)
     AMD:   amd_iommu=on iommu=pt

2. vfio-pci module loaded:
     sudo modprobe vfio-pci

3. Device bound to vfio-pci (one-time, survives reboots with udev):
     akida bind-vfio 0000:a1:00.0

4. User access to /dev/vfio/<IOMMU_GROUP>:
     sudo chown $USER /dev/vfio/<group>
   or add to vfio group:
     sudo usermod -aG vfio $USER
```

### Setup via akida-cli

```bash
# Find the device
akida enumerate

# Get IOMMU group
akida iommu-group 0000:a1:00.0

# Bind to vfio-pci (requires root)
sudo akida bind-vfio 0000:a1:00.0

# Grant user access
sudo chown $USER /dev/vfio/$(akida iommu-group 0000:a1:00.0)

# Verify — no root required from here
akida enumerate        # should show VfioBackend
akida info 0           # detailed capabilities
```

### What VFIO provides

| Feature | Mechanism |
|---------|-----------|
| BAR0 MMIO | `mmap()` on VFIO region with offset |
| BAR1 NP mesh | Same mechanism, larger region |
| DMA transfer | `VFIO_IOMMU_MAP_DMA` ioctl + `mlock()` |
| IOMMU isolation | IOMMU maps user virtual → device-visible IOVA |
| Interrupts | `VFIO_DEVICE_SET_IRQS` (queued for Phase D completion) |
| Device reset | `VFIO_DEVICE_RESET` (available, not yet used) |

### DMA flow

```
1. Allocate page-aligned buffer:  alloc_zeroed() with Layout::align_to(4096)
2. Lock in RAM:                   mlock(buf, size)   [prevents swap]
3. Map for DMA:                   VFIO_IOMMU_MAP_DMA ioctl → returns IOVA
4. Write IOVA to BAR0 registers:  MODEL_ADDR_LO/HI, INPUT_ADDR_LO/HI
5. Trigger operation:             write 1 to MODEL_LOAD or INFER_START
6. Poll completion:               read STATUS or INFER_STATUS
7. Read result:                   slice the pinned output buffer
8. Unmap on drop:                 VFIO_IOMMU_UNMAP_DMA ioctl
```

---

## 3. Kernel Backend — Fallback Path

When the C `akida_pcie` kernel module is loaded, `/dev/akida*` devices appear.
The kernel backend uses standard `read()`/`write()` syscalls.

```rust
// Kernel backend path — uses /dev/akida0
let f = File::open("/dev/akida0")?;
f.write_all(&model_bytes)?;           // programs the NPU
f.write_all(&input_bytes)?;           // sets input
f.read_exact(&mut output_buf)?;       // reads output
```

No IOMMU, no DMA programming — the kernel driver handles all of that.
Maximum throughput is still bounded by the same PCIe x1 Gen2 link.

---

## 4. Capability Discovery

All capabilities are discovered at runtime from sysfs. Nothing is hardcoded.

```
Discovery path: /sys/bus/pci/devices/{pcie_address}/
  vendor                → 0x1e7c                 → ChipVariant::Akd1000
  device                → 0xbca1
  akida_np_count        → 78                     → Capabilities.npu_count
  akida_sram_size       → 8388608 (bytes)        → .memory_mb = 8
  akida_clock_mode      → 0/1/2                  → .clock_mode
  akida_batch_size      → 8                      → .batch.max_batch
  link_speed            → "5.0 GT/s PCIe"        → .pcie.generation
  current_link_width    → "x1"                   → .pcie.lanes
  resource              → BAR layout             → .pcie.bar_layout
  hwmon/hwmon*/power1_average → power in µW      → .power_mw

Secondary: /dev/akida{N}
  Existence: device available for kernel backend
  iommu_group symlink: IOMMU group for VFIO
```

`DeviceManager::discover()` never fails if no hardware is present — it
returns an empty list. Callers must handle zero-device case.

---

## 5. API Contract

### Core types

```rust
// Discovery
let mgr: DeviceManager = DeviceManager::discover()?;
let info: &DeviceInfo = mgr.device(0)?;
let caps: &Capabilities = info.capabilities();

// Opening a device
let mut dev: AkidaDevice = mgr.open_first()?;      // auto backend
let mut dev = mgr.open(0, BackendSelection::Vfio)?  // explicit VFIO

// I/O
dev.write(&model_bytes)?;     // program the NPU + DMA input
dev.read(&mut out_buf)?;      // DMA output result
dev.write(&input_bytes)?;     // update input (model stays loaded)

// High-level inference
let exec: InferenceExecutor = InferenceExecutor::new(dev);
let result: InferenceResult = exec.run(&input_f32, config)?;

// Capabilities
caps.chip_version           // ChipVersion::Akd1000 | Akd1500
caps.npu_count              // u32 — 78 for AKD1000
caps.memory_mb              // u32 — 8 for AKD1000
caps.clock_mode             // Option<ClockMode>
caps.batch                  // Option<BatchCapabilities>
caps.weight_mutation        // WeightMutationSupport
caps.mesh                   // Option<MeshTopology>
caps.pcie                   // PcieConfig { generation, lanes, bandwidth_gbps }
```

### Error handling

All fallible operations return `Result<T, AkidaError>`. Error variants:

| Variant | When |
|---------|------|
| `NoDeviceFound` | `discover()` finds nothing, `open_first()` on empty manager |
| `DeviceNotFound(idx)` | `open(idx)` where idx >= device_count |
| `BackendUnavailable` | Requested backend (e.g. VFIO) not accessible |
| `TransferFailed(msg)` | DMA map/unmap/mlock error |
| `HardwareError(msg)` | Status register error bit set |
| `CapabilityQueryFailed(msg)` | sysfs read failure |
| `InvalidInput(msg)` | Input size mismatch, null ptr, etc. |

---

## 6. Thread Safety

`AkidaDevice` is `Send` but not `Sync` — it owns a file descriptor and
mutable DMA buffers. For concurrent access across threads, wrap in
`Arc<Mutex<AkidaDevice>>`.

`Capabilities` is `Clone + Send + Sync` — safe to cache and share.

`DeviceManager` is `Clone + Send + Sync` — safe to share for enumeration.

---

## 7. Feature Flags

| Feature | Default | Effect |
|---------|---------|--------|
| `default` | on | Kernel + userspace backends; no async |
| `async` | off | Adds `tokio`; enables async inference path |
| `kernel` | off | Explicitly gates kernel backend (future: needs C module at link time) |

To use VFIO backend: no feature flag needed; it's always compiled.
VFIO backends do require unix target (uses `rustix::mm`, `libc` ioctls).

---

## 8. Unsafe Surface

The `vfio/mod.rs` module contains the only unsafe code in the crate. Every
unsafe block is annotated with:
- Why it's necessary (the specific kernel API requires it)
- The invariants that make it safe
- Who is responsible for maintaining those invariants

Public API is 100% safe Rust. No unsafe code in `akida-chip`.
