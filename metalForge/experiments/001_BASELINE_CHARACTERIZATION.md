# Experiment 001 — Baseline Characterization

**Date:** February 27, 2026
**Status:** ✅ Complete
**Objective:** Establish baseline for all 10 BEYOND_SDK discoveries via the
pure Rust VFIO driver (Phase D). First confirmed run without any Python,
C++, or kernel module in the data path.

---

## Setup

| Component | Value |
|-----------|-------|
| Hardware | AKD1000, PCIe x1 Gen2, slot `08:00.0` |
| Host | AMD Threadripper 3970X, 256 GB DDR4, Linux 6.17.9 |
| Driver | rustChip akida-driver 0.1.0 — VFIO backend |
| Kernel module | NOT loaded (`akida_pcie` absent) |
| IOMMU | AMD-Vi enabled, iommu=pt |
| vfio-pci | bound via `akida bind-vfio 0000:08:00.0` |
| Rust | stable-x86_64-unknown-linux-gnu 1.93.1 |

---

## Objective

Prior work (hotSpring metalForge, wetSpring V60) validated the hardware via
Python SDK (Phase B) and direct `/dev/akida0` access (Phase C). Experiment 001
is the Phase D validation: all measurements reproduced via VFIO, no C module,
no Python, no FFI.

This experiment answers: **does the pure Rust VFIO driver produce identical
results to the Phase C kernel module path?**

---

## Protocol

1. Verify VFIO binding and IOMMU group
2. Run `cargo run --bin enumerate` — device enumerated correctly?
3. Run `bench_dma` — 37 MB/s sustained?
4. Run `bench_latency` — 54 µs / 18,500 Hz?
5. Run all 10 BEYOND_SDK discovery benchmarks
6. Compare to reference values from docs/BENCHMARK_DATASHEET.md
7. Document any deviations

---

## Results

All 10 discoveries confirmed. Full output in
`npu/akida/benchmarks/exp001_baseline.md`.

**Key numbers:**

| Metric | Phase C (kernel) | Phase D (VFIO) | Match? |
|--------|-----------------|----------------|--------|
| DMA throughput | 37 MB/s | 37.0 MB/s | ✅ |
| Single inference | 54 µs | 54 µs | ✅ |
| Batch=8 per-sample | 390 µs | 390 µs | ✅ |
| VFIO DMA overhead | — | +3 µs setup | ✅ negligible |
| BAR0 register access | via module | via mmap | ✅ identical |
| Device enumeration | /dev/akida0 | /dev/vfio/5 | ✅ same capabilities |

**Conclusion:** Phase D VFIO driver is functionally identical to Phase C
kernel module for all measured parameters. IOMMU overhead is absorbed into
normal measurement variance.

---

## Anomalies

**First-call latency spike:** The first inference call after device open
shows p99=142 µs vs steady-state p99=76 µs. Cause: IOMMU TLB miss on first
DMA map. Pre-warming the DMA pool (`InferenceExecutor::warm_up()`) eliminates
this. Subsequent 999/1000 calls are within normal bounds.

**VFIO_IOMMU_MAP_DMA with large buffers:** Mapping >1 MB at once causes
~20 µs setup overhead (IOMMU page table walk). For DMA benchmarks, buffer
reuse (pre-map, reuse across calls) is critical. The VFIO backend implements
this correctly via buffer pooling.

---

## Next Steps

- **Experiment 002:** Register probe during model load — capture BAR0 writes
  to confirm inferred register addresses (MODEL_ADDR_LO, INFER_START, etc.)
- **Experiment 003:** BAR1 exploration after model load — confirm sparse NP
  mapping structure
- **Experiment 004:** IRQ-based completion (replace polling loop with
  VFIO_DEVICE_SET_IRQS) — expected to reduce p99 latency
