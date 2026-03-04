# GPU + NPU Co-location Over PCIe

**Status:** Analysis complete. P2P DMA implementation: Phase D extension (queued).
**Date:** February 27, 2026

---

## The Question

Both the GPU and the AKD1000 NPU are PCIe devices on the same host. Can data
flow directly from GPU BAR memory to NPU IOVA without a CPU copy?

If yes, what is the architectural path? What changes in the driver?

---

## Current Architecture (CPU-mediated)

```
GPU compute (WGSL shader)
  ↓ GPU→CPU readback (wgpu map_async or cudaMemcpy)
CPU-resident buffer: Vec<f32>    ← pinned memory, ~15 µs
  ↓ akida-driver DMA
NPU input IOVA                   ← mlock + VFIO_IOMMU_MAP_DMA, ~14 µs for 512 floats
  ↓ NPU inference
NPU output IOVA                  ← ~5 µs readback
  ↓
CPU-resident result
```

**Total CPU-mediated overhead:** ~35–50 µs for a 512-float feature vector.
This is negligible compared to the 54–390 µs NPU inference itself.

**SRAM readback verification:** After the GPU computes and the NPU loads weights
via DMA, `NpuBackend::verify_load()` (SRAM readback) can confirm correct loading
before inference. This GPU→NPU verification path ensures model integrity in
safety-critical or multi-substrate pipelines.

For Experiment 022, the GPU trajectory takes 7.6 seconds. 50 µs is 0.0007%
of wall time. CPU mediation is not the bottleneck.

However — for edge deployments, streaming inference (>10 kHz), and large
feature vectors (thousands of floats), the copy overhead becomes meaningful.

---

## Peer-to-Peer DMA Path

PCIe P2P DMA allows one PCIe device to read/write another's BAR memory
directly, without routing through host DRAM. Both GPU and NPU support this
at the hardware level.

### Requirements

1. **Same root complex** — both devices must be downstream of the same PCIe
   root port (or connected via a PCIe switch with peer-to-peer capability).
   Most desktop motherboards support this for slots in the same PCH region.

2. **ACS (Access Control Services) disabled** — Intel/AMD PCIe ACS
   enforces IOMMU isolation per device; P2P requires ACS override.
   ```bash
   # Check ACS
   sudo lspci -vvv | grep ACS
   # Disable via kernel parameter (security tradeoff)
   echo "pcie_acs_override=downstream,multifunction" >> /etc/default/grub
   ```

3. **VFIO peer mapping** — Linux 5.16+ has `vfio_pci_core` with P2P mapping
   support. GPU side needs to expose its result buffer as a VFIO-accessible
   BAR region, or both devices need to be in the same VFIO container.

4. **GPU driver support** — NVIDIA requires `nv_peer_mem` (Mellanox OFED) or
   the `nvidia-peermem` kernel module for P2P. AMD GPUs via `amdgpu` support
   P2P via ROCm's `hsa_amd_ipc_memory_*` API.

### The P2P Flow

```
GPU result buffer (in GPU BAR2/VRAM)
  ↓ P2P DMA read (GPU BAR → NPU IOVA)
  [PCIe root complex routes packet directly]
  [No DRAM involvement]
NPU input IOVA (mapped from GPU BAR)
  ↓ NPU inference
NPU output IOVA
```

**Latency reduction:** Eliminates ~15 µs GPU→CPU + ~14 µs CPU→NPU = **~29 µs saved**.
For a 54 µs NPU inference, that's a 35% reduction in total dispatch time.

### Implementation Sketch (Phase D extension)

```rust
// In vfio/mod.rs — peer mapping extension
pub struct P2PDmaPath {
    /// Source: GPU BAR address (device-visible via IOMMU)
    gpu_iova: u64,
    /// Size of the feature vector in bytes
    size: usize,
    /// VFIO container (must contain both devices)
    container: Arc<VfioContainer>,
}

impl P2PDmaPath {
    /// Map GPU BAR memory as NPU input IOVA.
    ///
    /// Both devices must be in the same VFIO container.
    pub fn map_gpu_output_to_npu_input(
        container: &VfioContainer,
        gpu_bar_addr: u64,
        size: usize,
        npu_input_iova: u64,
    ) -> Result<Self> {
        // VFIO_IOMMU_MAP_DMA with vaddr = GPU BAR physical addr
        // This maps the GPU's output BAR region as the NPU's input IOVA
        // ...
        todo!("Phase D P2P extension")
    }
}
```

The key insight: VFIO IOMMU can map any physical address (including GPU BAR
physical addresses) as an IOVA for another device. The IOMMU provides the
translation.

---

## True Die-to-Die Integration (Long Horizon)

If BrainChip licenses the Akida IP (AKD1500 datasheet: "also available as IP"),
the NPU die could sit alongside a GPU die on a multi-chip module:

```
┌──────────────────────────────────────────────┐
│          Multi-Chip Module (MCM)              │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ GPU die  │  │ NPU die  │  │ SRAM die │  │
│  │ f32/f64  │  │ int4/int8│  │ shared   │  │
│  │ compute  │  │ inference│  │ scratchpad│  │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  │
│        └────── interposer ─────────┘         │
└──────────────────────────────────────────────┘
```

Die-to-die bandwidth (Intel EMIB, TSMC SoIC, or chiplet interconnect):
typical 1–10 TB/s (vs 0.5 GB/s for PCIe x1 Gen2).

At die-to-die bandwidth, the PCIe round-trip that currently dominates
(~650 µs) would drop to **~0.65 µs** — 1,000× improvement. At that point:
- Single inference: ~1.4 µs (chip compute time) instead of 54 µs
- SkipDMA latency advantage becomes the bottleneck, not bus transfers
- Batch advantage essentially disappears (no round-trip to amortise)

The Akida 1.0 architecture was designed for event-based computing with
sub-microsecond on-chip latency. Current PCIe delivery masks this entirely.
Die-to-die integration would finally reveal the chip's actual throughput ceiling.

AMD demonstrates this with the APU architecture (CPU + GPU on same package).
Intel does it with CPU + FPGA (Agilex). The pattern is established.
The question is compute area cost: does neuromorphic inference earn its
silicon area for the target workload? Our benchmarks provide that analysis.

---

## Practical Recommendation

For current hardware (AKD1000 PCIe x1, RTX 3090 PCIe x16):

**Use CPU mediation.** The 35–50 µs overhead is below measurement noise for
physics workloads where GPU trajectories take 7+ seconds. CPU-mediated is
simpler, more portable, and already validated in 5,978 production calls.

**Investigate P2P when:**
- Feature vectors exceed ~100 KB (37 MB/s DMA becomes latency-relevant)
- Streaming inference rate exceeds ~10 kHz sustained
- The GPU trajectory shortens to <1 ms (e.g. small lattices, real-time control)

---

## Connection to This Repository

The current GPU+NPU pattern in `specs/INTEGRATION_GUIDE.md` is CPU-mediated.
P2P DMA would be implemented as a `P2PDmaPath` extension in
`crates/akida-driver/src/vfio/mod.rs`, gated behind a `peer-dma` feature flag.

The interface change from the application's perspective: zero. The GPU-computed
`Vec<f32>` is replaced by a `P2PDmaPath` handle, but the `InferenceExecutor::run()`
call remains identical.
