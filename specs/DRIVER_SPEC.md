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

## 8. SRAM Access Layer

Two independent paths to on-chip SRAM, both pure Rust:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SRAM Access Paths                             │
│                                                                   │
│  Userspace (SramAccessor)          VFIO (VfioBackend)            │
│  ┌──────────────────┐              ┌──────────────────┐          │
│  │ sysfs BAR0 mmap  │              │ VFIO BAR0 region │          │
│  │ sysfs BAR1 mmap  │              │ VFIO BAR1 region │          │
│  │ No DMA needed    │              │ DMA available    │          │
│  │ read/write/probe │              │ read_u32/write_u32│          │
│  └──────────────────┘              └──────────────────┘          │
│           ↑                                  ↑                   │
│      SramAccessor::open()           VfioBackend::map_bar1()      │
└─────────────────────────────────────────────────────────────────┘
```

### SramAccessor (userspace path)

`crates/akida-driver/src/sram.rs` — direct BAR0/BAR1 access via sysfs mmap:

```rust
let mut sram = SramAccessor::open("0000:a1:00.0")?;

// BAR0 registers
let device_id = sram.read_register(0x0)?;

// BAR1 SRAM (auto-discovers layout from BAR0)
let data = sram.read_bar1(offset, length)?;
sram.write_bar1(offset, &new_data)?;
let results = sram.probe_bar1(&offsets)?;
sram.scan_bar1_range(start, end)?;
```

Layout discovery is automatic: `discover_layout()` reads NP_COUNT (0x10C0),
SRAM_REGION_0 (0x1410), and SRAM_REGION_1 (0x1418) from BAR0 to construct
a `Bar1Layout` with per-NP strides and region offsets.

### VfioBackend SRAM (DMA-capable path)

```rust
backend.map_bar1()?;                          // maps BAR1 via VFIO region
let val = backend.read_sram_u32(offset)?;     // 32-bit SRAM read
backend.write_sram_u32(offset, value)?;       // 32-bit SRAM write
assert!(backend.has_sram_mapped());           // check mapping status
println!("BAR1 size: {}", backend.sram_size()); // mapped region size
```

### NpuBackend SRAM Methods

All backends expose three SRAM operations via the `NpuBackend` trait:

```rust
// Model load verification via SRAM readback
let v = backend.verify_load(&model_bytes)?;   // -> LoadVerification

// Direct weight mutation (zero-DMA for small patches)
backend.mutate_weights(offset, &patch)?;

// Raw SRAM read
let data = backend.read_sram(offset, length)?;
```

### Runtime Capability Discovery

`Capabilities::from_bar0()` reads NP count, SRAM size, and mesh topology
directly from BAR0 registers — replacing hardcoded sysfs assumptions:

```rust
let caps = Capabilities::from_bar0("0000:a1:00.0")?;
// caps.np_count, caps.sram_per_np_kb, caps.mesh (from NP enable bits)
```

---

## 9. Unsafe Surface

The `vfio/mod.rs` module contains the only unsafe code in the crate. Every
unsafe block is annotated with:
- Why it's necessary (the specific kernel API requires it)
- The invariants that make it safe
- Who is responsible for maintaining those invariants

`IoHandle` in `src/io.rs` is zero-unsafe — uses `BorrowedFd<'fd>` via `rustix`.

Public API is 100% safe Rust. No unsafe code in `akida-chip`.
