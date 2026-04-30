# What If an NPU Were Part of a GPU Die?

**Date:** 2026-04-30
**Context:** rustChip + coralReef (ecoPrimals infra)

## The Question

What would change if a neuromorphic processing unit — like the AKD1000's
neural processor mesh — was fabricated on the same die as a GPU, sharing
the same package, power delivery, and memory subsystem?

This isn't hypothetical. Apple's Neural Engine shares die with GPU cores.
Intel's Meteor Lake has an NPU tile. AMD's XDNA is an AI engine on the
same package. But none of these are *neuromorphic* in the spiking,
event-driven sense. What would a real spike-domain NPU — like Akida's
NP mesh — look like as a GPU co-processor?

## Architecture: Three Integration Levels

### Level 1: Package-Level (chiplet / MCM)

```
┌──────────────────────────────────────────────┐
│                  Package                      │
│  ┌──────────────┐  ┌──────────────┐          │
│  │   GPU Die    │  │   NPU Die    │          │
│  │  (RDNA/Ada)  │  │  (NP mesh)   │          │
│  │              │──│              │  ← IF     │
│  │  CUs / SMs   │  │  80 NPs     │          │
│  │  L2 cache    │  │  10MB SRAM  │          │
│  └──────────────┘  └──────────────┘          │
│              ↕ HBM / GDDR                     │
└──────────────────────────────────────────────┘
```

**Interconnect:** UCIe, EMIB, or proprietary die-to-die.
**Latency:** ~5-20 ns die-to-die (vs ~500 ns for PCIe round-trip today).
**Bandwidth:** 100+ GB/s (vs 0.5 GB/s for PCIe Gen2 x1).

This is what AMD XDNA does — an AI accelerator tile on the same
interposer as the GPU. The NP mesh would replace the XDNA block
with event-driven spiking hardware.

### Level 2: Die-Level (shared silicon, separate domains)

```
┌───────────────────────────────────────────────┐
│                   GPU Die                      │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐    │
│  │  CU 0-31 │ │ CU 32-63 │ │  NP Mesh   │    │
│  │ (shader) │ │ (shader) │ │  80 NPs    │    │
│  └──────────┘ └──────────┘ │  10MB SRAM │    │
│       ↕            ↕        └────────────┘    │
│  ┌────────────────────────────────────────┐   │
│  │           Shared L2 / Infinity Cache    │   │
│  └────────────────────────────────────────┘   │
│                   ↕ HBM                        │
└───────────────────────────────────────────────┘
```

**Latency:** ~2-5 ns via on-die crossbar.
**Bandwidth:** L2 bandwidth (~TB/s).
**Power sharing:** Same VRM, dynamic power allocation.

The NP mesh has direct L2 cache access. GPU CUs can read NP
output from L2 without crossing any external bus. The NP mesh
sees GPU memory as its DMA target — no IOMMU translation.

### Level 3: Integrated (NPs as special-function CUs)

```
┌───────────────────────────────────────────────┐
│                   GPU Die                      │
│                                                │
│    CU  CU  CU  NP  CU  CU  NP  CU  CU       │
│    CU  NP  CU  CU  CU  NP  CU  CU  NP       │
│    CU  CU  CU  CU  NP  CU  CU  NP  CU       │
│                                                │
│    NPs share register file, scheduler,         │
│    and memory bus with shader CUs.             │
│                                                │
│  ┌────────────────────────────────────────┐   │
│  │           Shared L2 / Infinity Cache    │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

This is the most radical option. NPs become another execution unit
type in the GPU's wavefront scheduler, like tensor cores are today.
A compute shader dispatches work to NPs the same way it dispatches
to ALUs or tensor units.

## What Changes

### 1. PCIe Bottleneck Disappears

Today, AKD1000 at PCIe Gen2 x1 caps at ~37 MB/s sustained DMA.
A single inference takes 54 µs — but 80% of that is PCIe latency,
not compute. With on-die integration:

| Path              | Round-trip | Bandwidth  | Inference estimate |
|-------------------|-----------|------------|-------------------|
| PCIe Gen2 x1      | ~500 ns   | 0.5 GB/s   | 54 µs             |
| Chiplet (UCIe)     | ~15 ns    | 100 GB/s   | ~5 µs             |
| On-die (L2)        | ~3 ns     | 1 TB/s     | ~1 µs             |
| Integrated (reg)   | ~1 ns     | N/A (local)| ~0.3 µs           |

The NP mesh itself is fast (sub-microsecond for small models).
Almost all measured latency today is data movement.

### 2. Ensemble Becomes Free

The `ensemble_npu` binary today runs 3 software backends in
~1 ms per sample. With on-die NPUs, a GPU+NPU ensemble could:

- GPU tensor cores: run the dense floating-point classification
- NPU mesh: run the spiking/event-driven feature extraction
- Both results merge in L2 cache within the same clock domain

No serialization. No DMA. No context switches. The GPU shader
reads NP output from a shared address, just like reading from
a texture unit.

### 3. Power Budget Sharing

AKD1000 burns ~1.5W continuous. A GPU die has a 200-350W power
budget. An integrated NP mesh consuming <5W would be invisible
in the GPU's thermal envelope while providing:

- Always-on event monitoring (no GPU wake needed)
- Spike-domain preprocessing (compress sensor data before GPU sees it)
- Low-power inference when GPU CUs are idle

The NP mesh could run during GPU idle phases (memory refresh,
thermal throttling) at essentially zero marginal power cost.

### 4. Memory Hierarchy Integration

Today's AKD1000 has 10MB on-chip SRAM. On a GPU die:

```
NP Mesh SRAM (10 MB)  ← direct NP access
       ↕
L2 / Infinity Cache (32-96 MB)  ← shared with CUs
       ↕
HBM (16-80 GB)  ← GPU VRAM, NP sees via L2
```

NP weights live in SRAM. NP input comes from L2 (where GPU wrote
it). NP output goes to L2 (where GPU reads it). The entire data
path stays on-die. HBM bandwidth is reserved for GPU's bulk
matrix operations.

### 5. Programming Model

From rustChip/coralReef's perspective, the dispatch model simplifies:

```rust
// Today (PCIe-separated)
let gpu_result = coral_reef.dispatch(input)?;     // GPU via PCIe
let npu_result = rust_chip.infer(input)?;          // NPU via PCIe
let ensemble = merge(gpu_result, npu_result);      // CPU merges

// On-die integrated
let result = unified_dispatch(input, &[
    Substrate::Gpu { shader: "classify.comp" },
    Substrate::Npu { model: "esn_readout.fbz" },
], MergeStrategy::WeightedAverage)?;
// Single dispatch, hardware handles routing
```

The `SubstrateSelector` from `hybrid/mod.rs` already abstracts
this — the difference is that on-die integration removes the PCIe
latency that currently makes hardware substrate selection a
meaningful tradeoff.

## What Stays the Same

1. **Model format** — `.fbz` FlatBuffers, weight quantization (int4),
   NP configuration. The neural processor mesh is the same hardware
   regardless of where it lives on silicon.

2. **NpuBackend trait** — The backend abstraction works unchanged.
   A `GpuNpuBackend` would implement `NpuBackend` using GPU-local
   MMIO instead of PCIe VFIO, but the API is identical.

3. **Quantization gap** — f32→int4 quantization cost (~1-3%) is a
   property of the NP hardware, not the bus. Integration doesn't
   change this.

4. **SRAM capacity** — NP mesh SRAM is small (10 MB). Models must
   fit in SRAM regardless of integration level. What changes is how
   fast you can swap models (on-die: microseconds vs PCIe: milliseconds).

## Who's Closest?

| Company     | Product         | NPU Type    | Integration |
|------------|-----------------|------------|-------------|
| Apple       | M-series        | Neural Engine | Die-level, custom |
| Intel       | Meteor Lake     | XDNA NPU    | Tile (chiplet) |
| AMD         | Ryzen AI        | XDNA        | Package-level |
| Qualcomm    | Snapdragon X    | Hexagon NPU | Die-level |
| **BrainChip** | **AKD1000**  | **Spiking NP mesh** | **Discrete PCIe** |

BrainChip is the only one with a true spiking neuromorphic architecture,
but they're the furthest from integration — still discrete PCIe.
The path forward is clear: license the NP mesh IP to a GPU vendor
(AMD, Intel) for chiplet or die-level integration.

## For ecoPrimals

The `toadStool → coralReef + rustChip` dispatch model already
assumes heterogeneous substrates. The code is structured for this
future:

- `SubstrateSelector` picks GPU vs NPU based on task characteristics
- `HybridEsn` mixes hardware and software reservoirs
- `NpuBackend` trait abstracts the bus entirely
- `ensemble_npu` demonstrates cooperative multi-backend inference

When on-die NPUs arrive, the only change is a new backend
implementation. The architecture, models, and dispatch logic
carry forward unchanged.

## The Real Question

The real question isn't "what if an NPU was on a GPU die?" — it's
"what if neuromorphic computing was as cheap as a texture lookup?"

At 1 ns round-trip and zero marginal power cost, you'd use spiking
NPs for *everything*: every sensor read, every frame, every packet.
Not as a special accelerator you dispatch to, but as a fundamental
operation the GPU scheduler can invoke per-wavefront, like `sin()`
or `dot()`.

That's what Level 3 integration means. And that's what the
`hybrid/mod.rs` abstraction is already designed to support.
