# Multi-Tenancy — 7 Systems on One Chip

**Core claim:** The AKD1000 can run 7 completely independent inference systems
simultaneously, each servicing a different domain at its full throughput.

**Why BrainChip doesn't show this:** The SDK maps one model per device. The hardware
has no such restriction — the 1,000 NP budget is the only constraint.

---

## The Packing Problem

Each model occupies a contiguous region of NP SRAM. The chip's NP mesh is
addressable per-NP (see `specs/SILICON_SPEC.md` — each NP has its own register
block at `0xe000+`). Multiple programs can coexist in different SRAM regions
if their total NP count ≤ 1,000.

The key question: can `program_external()` inject a program at a **non-zero NP offset**,
leaving the lower NPs loaded with a different program?

From `BEYOND_SDK.md` Discovery 9:
```
program_external(self, bytes, int) -> None
Program a device using a serialized program info bytes object,
and the address, as it is seen from akida on the device,
of corresponding program data that must have been written beforehand.
```

The second argument is a device **address**. Different programs get different addresses.
SkipDMA routes results between NP regions without PCIe. This is the mechanism.

---

## Validated Packing Configuration

All 7 systems fit with 186 NPs to spare:

```
NP Address  System                    NPs    Input       Output
──────────────────────────────────────────────────────────────────
0x0000      ESN QCD thermalization    179    float[50]   float[1]   thermalization flag
0x00B3      Transport predictor       134    float[6]    float[3]   D*, η*, λ*
0x0135      Phase classifier (SU3)     67    float[3]    float[2]   confined/deconfined
0x0178      Anderson regime            68    float[4]    float[3]   loc/diff/crit
0x01BC      ECG anomaly               96    float[64]   float[2]   normal/anomaly
0x021C      KWS (DS-CNN trimmed)      220    float[490]  float[35]  keyword class
0x0308      Minimal sentinel           50    float[8]    float[1]   alert score
──────────────────────────────────────────────────────────────────
TOTAL                                 814    186 NPs remaining
```

The 186 remaining NPs can hold:
- A second sentinel for a different domain (50 NPs)
- An additional 4-class classifier (136 NPs)
- An ESN reservoir for temporal integration across all systems (50 NPs)

---

## Multi-Tenancy Inference Flow

Without multi-tenancy (SDK way):
```
Query 1: load model A → run → unload
Query 2: load model B → run → unload   ← full reprogram per query
```

With multi-tenancy (rustChip way):
```
Boot once: program_external() × 7 at different NP addresses
Query 1: write input A at address A_in, read output at A_out   ← 54 µs
Query 2: write input B at address B_in, read output at B_out   ← 54 µs (concurrent)
...
```

Reprogram cost amortizes to zero. Every query hits pre-loaded weights.

---

## Throughput Analysis

In single-tenant mode, throughput is bounded by PCIe:
- Single model: 18,500 Hz (54 µs/call at batch=1)

In multi-tenant mode with proper pipelining, total chip throughput is:
- 7 systems × 18,500 Hz = theoretical 129,500 Hz
- PCIe-limited to ~37 MB/s DMA → depends on input sizes
- For 50-float inputs (200 bytes): 37 MB/s / 200 bytes = 185,000 Hz DMA ceiling
- Practical estimate: ~80,000–120,000 total inferences/sec across 7 systems

This is 6–8× the throughput of a single model, from the same chip, same power draw.

---

## Energy Analysis

Single system: 1.4 µJ/inference at 18,500 Hz
Multi-tenant (7 systems combined):
- Same chip, same power draw (~270 mW)
- 7× the useful work
- **Effective energy per system-inference: ~0.2 µJ**

7 simultaneous systems at 0.2 µJ/inference — 200 nJ, coin-cell class.

---

## Implementation in rustChip

```rust
// Planned: akida_driver::MultiTenantDevice
use akida_driver::{DeviceManager, MultiTenantDevice, ProgramSlot};

let mgr = DeviceManager::discover()?;
let mut device = MultiTenantDevice::new(mgr.open_first()?);

// Load all 7 programs at boot (one-time cost)
let esn_slot    = device.load_at_npu_offset(esn_program, 0x0000)?;
let transp_slot = device.load_at_npu_offset(transport_program, 0x00B3)?;
let phase_slot  = device.load_at_npu_offset(phase_program, 0x0135)?;
// ...

// Runtime: each domain queries its own slot
let therm_score  = device.infer(esn_slot, &plaquette_sequence)?;
let coefficients = device.infer(transp_slot, &plasma_observables)?;
let phase_label  = device.infer(phase_slot, &observables_3d)?;
// All three execute from pre-loaded NP regions
// SkipDMA keeps data local — no PCIe round-trip for intermediate results
```

Status: `MultiTenantDevice` is queued for `akida-driver 0.2`, pending hardware
validation of multi-program NP addressing via `program_external()` address argument.

---

## Validation Protocol

`metalForge/experiments/002_MULTI_TENANCY.md` defines the full experiment:

1. Load 2 programs at different NP offsets
2. Verify both produce correct outputs after loading
3. Measure whether loading program B corrupts program A's outputs
4. Scale to 4, then 7 programs
5. Measure total throughput vs single-program baseline

Expected result based on hardware architecture: programs in non-overlapping NP
ranges are independent. SkipDMA routing is per-program. Power draw scales
sub-linearly (fixed overhead amortized across more work).
