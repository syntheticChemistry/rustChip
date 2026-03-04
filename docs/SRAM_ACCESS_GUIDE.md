# SRAM Access Guide — Read/Write All On-Chip Memory

**For anyone with an AKD1000 or AKD1500 who wants direct SRAM access.**

No Python SDK. No vendor support. No kernel module. Just Rust + PCIe.

---

## Prerequisites

1. A Linux system with an Akida device on PCIe (`lspci -d 1e7c:` should show it)
2. Rust toolchain (`rustup` — any recent stable)
3. Root access (one-time, for initial setup)

---

## Step 1: Build

```bash
git clone https://github.com/ecoPrimal/rustChip.git
cd rustChip
cargo build --release
```

---

## Step 2: Find your device

```bash
# Check that the device is visible on PCIe
lspci -d 1e7c:
# Example output: a1:00.0 Multimedia controller: BrainChip Inc. Device bca1

# Note the PCIe address (e.g., 0000:a1:00.0)
```

---

## Step 3: Probe SRAM (no setup needed)

The fastest path — `probe_sram` reads SRAM via sysfs mmap. No VFIO, no
kernel module, no root (after initial PCIe visibility).

```bash
# Read-only probe: dumps BAR0 registers + samples BAR1 SRAM
cargo run --release --bin probe_sram

# Deep scan: find all non-zero data across BAR1
cargo run --release --bin probe_sram -- scan

# Write/readback test: writes a pattern, reads it back, confirms match
# WARNING: this overwrites SRAM — will corrupt any loaded model
cargo run --release --bin probe_sram -- test
```

### What `probe_sram` shows you

**BAR0 registers** (control/status — 16 MB region):
```
Register          Offset     Value
─────────────────────────────────────
DEVICE_ID         0x000000   0x194000a1    ← confirms AKD1000
NP_COUNT          0x0010c0   0x0000005b    ← 91 (80 NPs + overhead)
SRAM_REGION_0     0x001410   0x00002000    ← SRAM config
SRAM_REGION_1     0x001418   0x00008000    ← SRAM config
NP_ENABLE[0]      0x001e0c   0x00000001    ← NP 0 enabled
...
```

**BAR1 SRAM** (NP mesh — 8 MB physical, 16 GB decode):
- Shows which per-NP offsets contain data
- Shows where weights are stored after model load
- Identifies the sparse mapping structure

---

## Step 4: Programmatic SRAM access (Rust)

### Userspace path (simplest — no VFIO required)

```rust
use akida_driver::sram::SramAccessor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open SRAM accessor — auto-discovers BAR1 layout from BAR0 registers
    let mut sram = SramAccessor::open("0000:a1:00.0")?;  // your PCIe address

    // ── Read BAR0 registers ──────────────────────────────
    let device_id = sram.read_register(0x0)?;
    println!("Device ID: {device_id:#010x}");

    let np_count = sram.read_np_count()?;
    println!("NP count register: {np_count}");

    // Dump all known registers
    let regs = sram.dump_registers()?;
    for r in &regs {
        if r.readable {
            println!("{:20} @ {:#08x} = {:#010x}", r.name, r.offset, r.value);
        }
    }

    // ── Read BAR1 SRAM ───────────────────────────────────
    // Read 4 KB from the start of NP 0's address window
    let data = sram.read_bar1(0x0000, 4096)?;
    println!("First 16 bytes: {:02x?}", &data[..16]);

    // Read from a specific NP's SRAM
    let layout = sram.layout();
    if let Some(np5_base) = layout.np_base_offset(5) {
        let np5_data = sram.read_bar1(np5_base as usize, 256)?;
        println!("NP 5 first 16 bytes: {:02x?}", &np5_data[..16]);
    }

    // ── Write BAR1 SRAM ──────────────────────────────────
    // WARNING: writing corrupts loaded models
    let test_pattern = vec![0xDE, 0xAD, 0xBE, 0xEF];
    sram.write_bar1(0x1000, &test_pattern)?;

    // Read back to verify
    let readback = sram.read_bar1(0x1000, 4)?;
    assert_eq!(readback, test_pattern);
    println!("Write/readback verified!");

    // ── Probe multiple NPs ───────────────────────────────
    let results = sram.probe_bar1(8)?;  // probe first 8 NPs
    for r in &results {
        let status = if r.has_data { "DATA" } else { "empty" };
        println!("{:30} @ {:#010x}  {status}  {:?}",
                 r.description, r.offset, r.value);
    }

    // ── Scan for non-zero data ───────────────────────────
    let hits = sram.scan_bar1_range(0, 1024 * 1024, 4)?;  // scan first 1 MB
    println!("Found {} non-zero locations", hits.len());
    for (offset, value) in hits.iter().take(10) {
        println!("  {offset:#010x}: {value:#010x}");
    }

    Ok(())
}
```

### VFIO path (DMA-capable, for production use)

If you need DMA for model loading + SRAM access in the same session:

```bash
# One-time setup (requires root)
sudo modprobe vfio-pci
sudo cargo run --bin akida -- bind-vfio 0000:a1:00.0
sudo chown $USER /dev/vfio/$(cargo run --bin akida -- iommu-group 0000:a1:00.0)
```

```rust
use akida_driver::VfioBackend;
use akida_driver::NpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = VfioBackend::init("0000:a1:00.0")?;

    // Map BAR1 for SRAM access
    backend.map_bar1()?;

    // Read/write individual 32-bit words
    let value = backend.read_sram_u32(0x0)?;
    println!("SRAM[0]: {value:#010x}");

    backend.write_sram_u32(0x1000, 0xCAFE_BABE)?;
    let readback = backend.read_sram_u32(0x1000)?;
    assert_eq!(readback, 0xCAFE_BABE);

    // Load a model, then verify via SRAM readback
    let model_bytes = std::fs::read("model.fbz")?;
    backend.load_model(&model_bytes)?;

    let verification = backend.verify_load(&model_bytes)?;
    println!("Load verified: {} bytes checked, match: {}",
             verification.bytes_checked, verification.verified);

    Ok(())
}
```

---

## Step 5: Runtime capability discovery

Don't hardcode NP counts or SRAM sizes — discover them from the hardware:

```rust
use akida_driver::capabilities::Capabilities;

let caps = Capabilities::from_bar0("0000:a1:00.0")?;
println!("NP count: {}", caps.npu_count);
println!("SRAM: {} MB", caps.memory_mb);
println!("Chip: {:?}", caps.chip_version);

// Mesh topology from NP enable bits
if let Some(mesh) = &caps.mesh {
    println!("Mesh: {}×{}×{}, {} functional NPs",
             mesh.rows, mesh.cols, mesh.layers, mesh.functional);
}
```

---

## SRAM address space structure

```
BAR0 (16 MB) — Control registers
  0x000000  DEVICE_ID         Device identity + version
  0x000008  STATUS            Ready/Busy/Error/ModelLoaded
  0x0010C0  NP_COUNT          Number of NPs (91 on AKD1000)
  0x001410  SRAM_REGION_0     SRAM config register
  0x001418  SRAM_REGION_1     SRAM config register
  0x001E0C  NP_ENABLE[0..5]   Per-NP enable bits
  0x00E000  NP_CONFIG[0]+     Per-NP register blocks (0x100 stride)

BAR1 (16 GB decode, 8 MB physical) — NP mesh SRAM window
  Per-NP stride = 16 GB / NP_COUNT
  Each NP contains:
    Filter SRAM     (64-bit entries) — convolution/FC weights
    Threshold SRAM  (51-bit entries) — activation thresholds + biases
    Event SRAM      (32-bit entries) — spike events / activations
    Status SRAM     (32-bit entries) — layer status / control

  Layout auto-discovered by SramAccessor from BAR0 registers.
  Physical SRAM is sparse — most of the 16 GB decode returns zeros.
```

---

## Common tasks

### Dump all weights after model load
```rust
let mut sram = SramAccessor::open(pcie_addr)?;
let layout = sram.layout();
for np in 0..layout.np_count {
    if let Some(base) = layout.np_base_offset(np) {
        let data = sram.read_bar1(base as usize, layout.per_np_sram_bytes as usize)?;
        let non_zero = data.iter().filter(|&&b| b != 0).count();
        if non_zero > 0 {
            println!("NP {np}: {non_zero} non-zero bytes at base {base:#x}");
        }
    }
}
```

### Write test patterns to verify SRAM integrity
```bash
cargo run --release --bin probe_sram -- test
```

### Compare two devices (PUF fingerprinting)
```rust
use akida_driver::puf::{measure_puf, puf_hamming_distance, PufConfig};

let sig_a = measure_puf(&mut sram_device_a, &PufConfig::default())?;
let sig_b = measure_puf(&mut sram_device_b, &PufConfig::default())?;
let distance = puf_hamming_distance(&sig_a, &sig_b);
println!("Inter-device distance: {distance:.3}");  // ~0.5 = distinct devices
println!("Device A entropy: {:.2} bits", sig_a.entropy());
```

### Verify multi-tenant isolation
```bash
# Load two programs, verify no SRAM cross-contamination
cargo run --release --bin bench_exp002_tenancy -- --hw
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `lspci -d 1e7c:` shows nothing | Device not seated / not powered. Check PCIe slot. |
| `SramAccessor::open()` fails | Check `/sys/bus/pci/devices/{addr}/resource0` exists and is readable. May need `chmod` or run as root once. |
| BAR1 reads all zeros | Normal for unprogrammed device. Load a model first, then re-read. |
| BAR1 reads `0xBADF5040` | "Bad food" — protected/uninitialized region. Skip these offsets. |
| BAR1 reads `0xFFFFFFFF` | PCIe bus error — offset beyond physical SRAM. |
| VFIO `map_bar1()` fails | IOMMU may not support BAR1 mapping. Use sysfs path (`SramAccessor`) instead. |

---

## What this enables

With full SRAM read/write, you can:

- **Verify model loads** — read back weights after DMA transfer, compare byte-for-byte
- **Inspect on-chip state** — see what the NPU "thinks" at any point
- **Mutate weights directly** — online learning without full reprogram (~86 µs)
- **Fingerprint devices** — PUF via int4 quantization noise (6.34 bits entropy)
- **Test SRAM integrity** — write patterns, read back, detect bit errors
- **Debug inference failures** — inspect intermediate activations in Event SRAM
- **Multi-tenant verification** — confirm program isolation between NP regions

---

## License

AGPL-3.0-or-later. This guide and all rustChip code.
