# AKD1000 NPU Characterization

**Hardware:** BrainChip AKD1000, PCIe 2.0 x1, `08:00.0`, BC.00.000.002
**Driver:** rustChip akida-driver 0.1.0 (VFIO primary)
**Date:** February 2026

---

## Quick Reference

| Property | Value |
|----------|-------|
| PCIe ID | `1e7c:bca1` |
| NPs | 78 (5×8×2 mesh, 2 disabled) |
| NP types | CNP1×78, CNP2×54, FNP2×4, FNP3×18 |
| NPUs per NP | 4 |
| MACs per NPU | 128 |
| Total MACs | 39,936 |
| On-chip SRAM | 8 MB |
| BAR0 | 16 MB (registers) |
| BAR1 | 16 GB (NP mesh window — Discovery 8) |
| DMA throughput | 37 MB/s (sustained) |
| Single inference | 54 µs / 18,500 Hz |
| Energy/inference | 1.4 µJ |
| Clock modes | Performance / Economy / LowPower |
| Batch optimal | 8 (2.4× speedup) |

---

## Documents

| Document | Contents |
|----------|---------|
| [`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md) | NP types, capabilities, SRAM layout, register map derivation |
| [`REGISTER_PROBE_LOG.md`](REGISTER_PROBE_LOG.md) | Raw BAR0 read log — the source of `specs/SILICON_SPEC.md` confirmed entries |
| [`benchmarks/`](benchmarks/) | Benchmark results by experiment |
| `../../../docs/BEYOND_SDK.md` | 10 SDK assumptions overturned (the primary discovery doc) |
| `../../../docs/HARDWARE.md` | Full deep-dive from hotSpring metalForge |

---

## Key Findings

### Silicon Probing Methodology

We probed BAR0 directly via MMIO (`mmap()` on VFIO region index 0).
A 64 KB scan at 4-byte stride found:

```
0x000000: 0x194000a1  — Device ID / version register    ← confirmed
0x001094: 0x0000a028  — Control register                ← confirmed
0x0010c0: 0x0000005b  — NP count field (0x5b = 91)      ← confirmed
0x001410: 0x00002000  — SRAM region config 0            ← confirmed
0x001418: 0x00008000  — SRAM region config 1            ← confirmed
0x001484: 0x5e1e04xx  — Timestamp / firmware version    ← confirmed
0x001e0c–0x001e20: 0x00000001 × 6  — NP enable bits     ← confirmed
0x004010: 0x04aa0001  — DMA/mesh configuration          ← confirmed
0xe000+:  repeating pattern  — per-NP register blocks   ← confirmed
0xbadf5040: "Bad food" — uninitialized / protected      ← confirmed
```

The "Bad food" (`0xbaddf00d`) pattern at `0xbadf5040` is a common sentinel
for uninitialized hardware registers. It confirms that the probe reached
protected address space without kernel panic — the IOMMU correctly isolated
the BAR mapping.

### SkipDMA Discovery

The C++ engine exports `SkipDmaLayer` and `SkipDmaTransfer`. These symbols
indicate that NP-to-NP data routing bypasses the DMA engine (and thus PCIe)
for in-mesh transfers. Measurement confirms: 8 FC layers cost only 3 µs more
than 2 FC layers (Discovery 2).

### FlatBuffer Reverse Engineering

`.fbz` model files contain:
1. A Snappy-compressed blob
2. Decompressed: a FlatBuffer binary
3. Root table splits into `program_info` (~332 bytes) and `program_data` (~396 bytes)

SDK version string appears at byte offset 236 in all tested models: "2.19.1"
`program_external()` takes `program_info` bytes + an IOVA pointing to
pre-loaded `program_data`. Weights are separate (DMA'd via `set_variable()`).

---

## Probing Tools

All probing is done via `akida-bench` binaries:

```bash
# Hardware identification
cargo run --bin enumerate

# BAR layout (Discovery 8)
cargo run --bin bench_bar

# Each BEYOND_SDK discovery
cargo run --bin bench_channels     # Discovery 1
cargo run --bin bench_fc_depth     # Discovery 2
cargo run --bin bench_batch        # Discovery 3
cargo run --bin bench_clock_modes  # Discovery 4
cargo run --bin bench_fc_width     # Discovery 5
cargo run --bin bench_weight_mut   # Discovery 6
cargo run --bin bench_dma          # Discovery 9 / production
cargo run --bin bench_latency      # Production latency
```

---

## Outstanding Questions

| Question | Status | How to resolve |
|----------|--------|---------------|
| eDMA descriptor layout | ❓ Inferred from DW spec | BrainChip can confirm / DW databook |
| Per-NP register block format | ❓ Pattern confirmed, content unknown | Requires C++ engine deeper analysis |
| BAR1 sparse mapping offsets | ❓ Reads zero in first 64 KB | Requires programmed model to see NP mapping |
| On-chip learning registers | ❓ Symbol exists, path unknown | Requires BrainChip disclosure |
| AKD1500 register deltas | ❓ Extrapolated from AKD1000 | Requires AKD1500 hardware |
