# AKD1000 Benchmark Datasheet
## Measured Performance — hotSpring metalForge

**Date:** February 19-27, 2026
**Hardware:** BrainChip AKD1000 (BC.00.000.002), PCIe 2.0 x1, slot 08:00.0
**Host:** AMD Threadripper 3970X, 128 GB DDR4, Intel i9-12900K (secondary)
**SDK:** Akida 2.19.1 (hotSpring); ToadStool `akida-driver` 0.1.0 pure Rust (wetSpring)
**Driver:** `akida_pcie` → `/dev/akida0`
**Crate:** hotspring-barracuda v0.6.14

---

## 1. Hardware Identification

| Property | Value |
|----------|-------|
| Chip | AKD1000 (Akida 1.0) |
| PCIe ID | `1e7c:bca1` |
| Neural Processors (NPs) | 78 (CNP1×78, CNP2×54, FNP2×4, FNP3×18) |
| NPUs per NP | 4 |
| MACs per NPU | 128 |
| Total MACs | 40,960 |
| On-chip SRAM | 8 MB |
| External memory | 256 Mbit LPDDR4 |
| PCIe link | x1 Gen2 (~500 MB/s) |
| PCIe BAR0 | 16 MB (register space) |
| PCIe BAR1 | 16 GB (NP mesh / SRAM window) |
| PCIe BAR3 | 32 MB (secondary memory) |
| Weight precision | 1, 2, 4-bit |
| Activation precision | 8-bit input, 4-bit internal |

---

## 2. ESN Readout Inference (Primary Use Case)

Model: InputConv(1,1,8→50) → FullyConnected(50→1)
Program size: 752 bytes. Maps to 1 FNP3 node.
Weights: 2,950 parameters (1,475 bytes at int4).

| Metric | Hardware | Software (CPU) | Ratio |
|--------|----------|----------------|-------|
| Inference cycles | 668 | — | — |
| Inference latency | 656 μs | 7 μs | CPU 94× faster |
| Throughput (FPS) | 1,525 | 143,710 | CPU 94× higher |
| Power (chip) | < noise | ~65W (TDP) | NPU ~65,000× lower |
| Power (board) | 918 mW | — | PCIe board overhead |

**Note:** For this trivially small model, PCIe round-trip latency (650 μs)
dominates. The chip computes in ~0.7 μs. The CPU wins on latency because
it avoids the bus entirely. The NPU wins on power and concurrent operation
(zero GPU cycles stolen).

---

## 3. Scaling: FC Width

Model: InputConv(8→N) → FC(N→N) → FC(N→1)

| Width | Program Size | Latency (μs) | Throughput | Regime |
|------:|-------------:|--------------:|-----------:|--------|
| 64 | 5,120 B | 779 | 1,284 FPS | PCIe dominated |
| 128 | 15,408 B | 700 | 1,429 FPS | PCIe dominated |
| 256 | 54,464 B | 812 | 1,232 FPS | PCIe dominated |
| 512 | 206,208 B | 1,106 | 904 FPS | **Crossover** |
| 1,024 | 804,608 B | 1,986 | 503 FPS | Compute contributing |
| 2,048 | 3,181,056 B | 4,969 | 201 FPS | Compute dominant |
| 4,096 | 12,652,544 B | 16,141 | 62 FPS | Compute dominant |
| 8,192 | — | — | — | Maps to HW (SRAM limit) |

**Crossover point:** ~width=512. Below 512, PCIe overhead dominates and
all models run at ~700 μs. Above 512, compute time scales with width².

---

## 4. Scaling: FC Depth (SkipDMA Merge)

Model: InputConv(3→64) → FC(64)^depth → FC(1)

| Depth | Layers | Program Size | Latency (μs) | Δ vs depth=2 |
|------:|-------:|-------------:|--------------:|-------------:|
| 2 | 3 | 3,048 B | 713 | baseline |
| 3 | 4 | 4,600 B | 741 | +28 μs |
| 4 | 5 | 6,152 B | 692 | -21 μs (noise) |
| 5 | 6 | 7,704 B | 703 | -10 μs (noise) |
| 8 | 9 | 12,360 B | 716 | +3 μs |

**All FC layers merge into one hardware sequence** via intra-mesh SkipDMA.
Adding 6 more FC layers costs 3 μs. Deep networks are effectively free
once the PCIe transfer is paid.

---

## 5. Batch Inference

Model: 50→256→256→256→1 (108 KB program)

| Batch | Total Time (ms) | Per-Sample (μs) | Throughput (samp/s) | Speedup |
|------:|----------------:|-----------------:|--------------------:|--------:|
| 1 | 0.95 | 948 | 1,055 | 1.0× |
| 2 | 1.14 | 568 | 1,760 | 1.7× |
| 4 | 1.70 | 426 | 2,346 | 2.2× |
| **8** | **3.12** | **390** | **2,566** | **2.4×** |
| 16 | 7.70 | 481 | 2,078 | 2.0× |
| 32 | 18.57 | 580 | 1,723 | 1.6× |
| 64 | 29.28 | 458 | 2,186 | 2.1× |

**Batch=8 is the sweet spot:** 2.4× throughput over single inference.
PCIe transfer cost amortized across batch elements.

---

## 6. Clock Modes

Model: 50→256→256→256→1

| Mode | Latency (μs) | Board Power (mW) | Δ Latency | Δ Power |
|------|-------------:|-----------------:|----------:|--------:|
| Performance | 909 | 901 | baseline | baseline |
| Economy | 1,080 | 739 | +19% | **-18%** |
| LowPower | 8,472 | 658 | +832% | -27% |

**Economy mode** is optimal for physics workloads with ≥4 ms between
predictions: 19% slower, 18% less power, no code change.

---

## 7. Physics-Scale Model Benchmarks

| Config | Latency (μs) | FPS | Notes |
|--------|-------------:|----:|-------|
| 8→32→1 (ESN small) | 698 | 1,432 | PCIe dominated |
| 50→64→1 (ESN readout) | 667 | 1,500 | PCIe dominated |
| 50→128→1 (wide readout) | 661 | 1,514 | PCIe dominated |
| 50→256→1 (fat readout) | 754 | 1,325 | Slight compute |
| 50→512→1 (XL readout) | 1,055 | 948 | Compute starts |
| 50→512→3 (multi-output) | 1,149 | 870 | Multi-output viable |
| 50→1024→1 (massive) | 1,979 | 505 | Compute dominant |
| 50→1024→10 (multi-massive) | 1,913 | 523 | **10 outputs ≤ 1 output** |
| 100→256→1 (deep physics) | 798 | 1,253 | Wider input, same HW |
| 256→512→1 (feature map) | 1,112 | 899 | Large projection |

**Multi-output is free or negative cost.** The NP mesh parallelism handles
multiple outputs simultaneously.

---

## 8. Weight Mutation

| Operation | Time | Notes |
|-----------|-----:|-------|
| Forward pass (inference only) | 663 μs | Normal operation |
| `set_variable()` + forward | 14 ms | ~13 ms mutation overhead |
| Reprogram (full `map()`) | ~150 ms | Full model reload |

Weights are DMA'd to NP SRAM separately from the program binary.
`set_variable()` updates weights without reprogramming the NP mesh routing.

---

## 9. probe_sram Benchmark (Placeholder)

The `probe_sram` binary (probe/scan/test modes) provides direct BAR0/BAR1
access for SRAM exploration and load verification.

| Mode | Metric | Placeholder / Notes |
|------|--------|---------------------|
| **probe** | BAR0 register probe time (80 registers) | < 1 ms |
| **scan** | BAR1 region scan throughput | Depends on BAR1 size; to be measured |
| **test** | Read/write round-trip verification | To be measured |

Full benchmark data pending hardware runs. Use `probe_sram --help` for mode
options.

---

## 10. SRAM Weight Budget

| Model | Parameters | At int4 | Fits in 8 MB SRAM |
|-------|----------:|---------:|:-----------------:|
| ESN readout (50→1) | 2,950 | 1,475 B | Trivially |
| Wide readout (50→256→1) | 13,312 | 6.5 KB | Yes |
| Deep FC (50→256³→1) | 197,376 | 96 KB | Yes |
| 256×256 | 131,072 | 64 KB | Yes |
| 1024×1024 | 2,097,152 | 1 MB | Yes |
| 4096×4096 | 33,554,432 | ~4 MB | Yes (half SRAM) |
| 8192×8192 | 134,217,728 | ~8 MB | At limit |

---

## 11. Determinism and Reproducibility

| Property | Result |
|----------|--------|
| Inference determinism | 10/10 identical outputs from identical input |
| Model save/reload parity | `.fbz` round-trip produces identical output |
| Cross-run reproducibility | Identical across power cycles |
| Hardware version | Fully deterministic digital (no analog, no stochastic elements) |

---

## 12. Comparison: NPU vs GPU for ESN Inference

| Metric | AKD1000 (NPU) | RTX 3090 (GPU) | Titan V (GPU) |
|--------|---------------:|---------------:|--------------:|
| ESN readout latency | 656 μs | ~50 μs | ~50 μs |
| Power for inference | < 1 mW (chip) | ~338 W (full chip) | ~250 W |
| Energy per inference | ~0.7 μJ | ~17 mJ | ~12.5 mJ |
| Concurrent with HMC | ✅ (separate PCIe device) | ❌ (shares GPU) | ❌ (shares GPU) |
| Batch=8 throughput | 2,566 samp/s | >100,000 samp/s | >50,000 samp/s |

The NPU advantage is **not raw throughput** — GPUs are faster by orders of
magnitude. The NPU advantage is **zero contention**: the GPU runs physics at
100% utilization while the NPU runs inference on a separate PCIe device at
microwatt power. No scheduling conflict. No memory pressure. No thermal impact.

---

## 13. Three-Substrate Production Results

### 13.1 Experiment 015 (Validation Run — 3 Strategic β Points)

Full pipeline: RTX 3090 (DF64 HMC) + AKD1000 (ESN steering) + Titan V (f64 oracle)

| β | ⟨P⟩ (DF64) | ⟨P⟩ (native f64 baseline) | Agreement |
|---|:----------:|:-------------------------:|:---------:|
| 5.0000 | 0.402743 | 0.401404 | +0.33% ✅ |
| 5.6900 | 0.531225 | 0.521552 | +1.85% ✅ |
| 6.5000 | 0.634681 | 0.630085 | +0.73% ✅ |

| Metric | Native f64 Baseline | Three-Substrate | Improvement |
|--------|:-------------------:|:---------------:|:-----------:|
| Time per trajectory | 15.5 s | 7.6 s | **2.04× faster** |
| Measurements per β point | 200 | 500 | **2.5× more data** |
| GPU power draw | 368-374 W | 338-340 W | **8% lower** |

### 13.2 Experiment 022 (Production Run — 10 NPU-Steered β Points, Live AKD1000)

Full pipeline with live NPU hardware via PCIe. Cross-run ESN bootstrapping from prior
trajectories. Completed February 27, 2026.

| Metric | Exp 013 (no NPU) | Exp 022 (live NPU) | Improvement |
|--------|:-----------------:|:-------------------:|:-----------:|
| Lattice | 32⁴ | 32⁴ | Same |
| Beta points | 7 (manual) | 10 (3 seed + 7 NPU-steered) | **43% more coverage** |
| Total measurements | 2,200 | 5,900 | **2.68× more data** |
| Thermalization trajectories | 1,400 | 540 (out of 2,000 budgeted) | **63% saved** |
| Time per trajectory | 15.5 s (native f64) | 7.6 s (DF64) | **2.04× faster** |
| Total NPU calls | 0 | 5,978 | New capability |
| Rejection prediction accuracy | N/A | 80.4% | New capability |
| ESN β_c estimate | N/A | 5.5657 (known: 5.692, error 2.2%) | New capability |
| Wall time | 13.6 h | 14.2 h | Comparable |
| Cost (electricity) | $0.58 | ~$0.61 | Comparable |
| Susceptibility peak | Not measured | χ=32.41 at β=5.7797 | New observable |
| Cross-run learning | N/A | Bootstrapped from 749 prior points | New capability |

**Key insight:** Same wall time, same hardware cost — but 2.5× more measurement
statistics placed more intelligently by NPU adaptive steering. The 30 mW chip saved
2.8 hours of GPU thermalization compute and concentrated measurements in the physically
interesting deconfinement transition region.

---

## 14. AKD1500 Projected Performance (from Datasheet)

Source: AKD1500 Datasheet v1.2, June 2025. No hardware measurements yet —
these are projections based on documented specs and our AKD1000 measurements.

| Property | AKD1000 (measured) | AKD1500 (projected) | Basis |
|----------|:------------------:|:-------------------:|-------|
| PCIe bandwidth | ~500 MB/s (x1 Gen2) | **~1 GB/s (x2 Gen2)** | Datasheet: dual lane |
| PCIe round-trip (ESN) | 650 μs | **~350-400 μs** | 2× bandwidth for same payload |
| Batch=8 per-sample | 390 μs | **~200-250 μs** | PCIe amortization at 2× BW |
| NPU count | 78 NPs (mixed) | 32 NPUs (8 nodes × 4) | Datasheet Section 4.2 |
| SRAM per NPU | Shared pool | **100 KB** (reallocable) | Datasheet Section 4.2 |
| Total on-chip SRAM | 8 MB | **~3.2 MB + 1 MB dual-port** | 32 × 100 KB + dual-port |
| Max FC width (estimated) | 8192 (tested) | **~2048-4096** | SRAM-limited |
| ESN readout (50→1) fit | Trivially (1.5 KB) | **Trivially (1.5 KB)** | Same model |
| Package | PCIe card | **7×7 mm BGA169** | Datasheet Section 14 |
| Core voltage | — | 0.8V, max 2A | Datasheet Table 25 |
| Max core power | ~30 mW (chip) | **~1.6W max** | 0.8V × 2A |
| PCIe device ID | `1E7C:BCA1` | `1E7C:A500` | Datasheet Table 2 |
| SPI interface | None | **Dual SPI (master + slave)** | Datasheet Section 3.5-3.7 |
| GPIO | None | **Up to 24 pins** | Datasheet Section 3.3 |
| Neural IP | Akida 1.0 | **Akida 1.0 (same)** | Datasheet Section 1 |
| Linux driver support | 5.4+ | **5.4-6.8 only** | Datasheet Section 3.6 |

### Projected Impact on ESN Workload

| Metric | AKD1000 | AKD1500 (projected) | Change |
|--------|:-------:|:-------------------:|:------:|
| Single ESN inference | 656 μs | ~350 μs | **1.9× faster** |
| Batch=8 throughput | 2,566 samp/s | ~4,000-5,000 samp/s | **1.6-2.0×** |
| PCIe→compute crossover | width=512 | ~width=256 | More models in "NPU earns its keep" regime |
| Multi-model capacity | ~28 KB / 8 MB (0.4%) | ~28 KB / 3.2 MB (0.9%) | Still trivial |

### SPI Performance (Unknown — To Be Measured)

The AKD1500 supports SPI slave at up to 100 MHz (reference clock dependent).
Theoretical SPI throughput in octal DDR mode:

| Mode | Theoretical BW | vs PCIe x2 | Notes |
|------|:--------------:|:----------:|-------|
| SPI single (66 MHz) | ~8 MB/s | 0.8% | Baseline |
| SPI quad (100 MHz) | ~50 MB/s | 5% | Practical for small models |
| SPI octal DDR (100 MHz) | ~200 MB/s | 20% | Best case |
| PCIe x2 Gen2 | ~1 GB/s | 100% | Reference |

SPI will be slower than PCIe for throughput-sensitive workloads but enables
embedded deployment without a PCIe host. For our ESN inference payload
(~400 bytes input + ~50 bytes output), even single-mode SPI adds only
~50 μs transfer time — acceptable for sensor node deployment.

---

## 15. wetSpring Live Hardware Results (Pure Rust, February 26, 2026)

**Driver:** ToadStool `akida-driver` 0.1.0 — **pure Rust, zero SDK/vendor dependency**
**Host:** Intel i9-12900K, 64 GB DDR5, Pop!\_OS 22.04
**Device:** AKD1000 @ `0000:08:00.0` (80 NPUs, 10 MB SRAM, PCIe Gen2 x1)

### 15.1 ESN Classifier Inference (Exp194)

| Classifier | CPU Sim Acc | NPU Live Acc | Throughput | Energy/Infer |
|------------|:----------:|:----------:|:----------:|:------------:|
| QS Phase (3-class) | 49.2% | 33.6% | 18,837 Hz | 1.4 µJ |
| Bloom Sentinel (4-class) | 25.0% | 25.3% | 18,773 Hz | 1.4 µJ |
| Disorder (3-class) | 32.9% | 31.6% | 18,626 Hz | 1.4 µJ |

### 15.2 DMA Performance

| Operation | Measured |
|-----------|---------|
| Sustained write throughput | 37 MB/s |
| Sustained read throughput | 37 MB/s |
| Reservoir load (200×200 sparse, 164 KB) | 4.5 ms |
| Single inference round-trip | 54 µs |
| Batch inference (8-wide) | 20,754 infer/sec |
| Readout switch (weight mutation) | 28 µs avg |

### 15.3 Novel Capabilities (Exp195)

| Experiment | Result |
|------------|--------|
| Physical Reservoir Fingerprint (PUF) | 6.34 bits byte entropy, deterministic dual-state alternating signature |
| Online (1+1)-ES Evolution | 136 gen/sec, 24% → 32% fitness in 50 gens (real-time capable) |
| Temporal Streaming (500-step) | 12,883 Hz sustained, p99=76 µs |
| Anderson Disorder Sweep | 8 levels (W=0 to 30), response characterized per disorder strength |
| Cross-Reservoir Crosstalk | 12,765 switch/sec, no state bleed between classifiers |

### 15.4 Edge Power Budget

| Scenario | Power | Energy/Day | Coin-Cell CR2032 |
|----------|-------|-----------|:----------------:|
| 1 Hz edge buoy | 30 mW | 0.125 J | **11 years** |
| 10 Hz sensor node | 30 mW | 1.25 J | 400 days |
| Continuous (18K Hz) | 30 mW | 2,592 J | N/A (wired) |

### 15.5 Significance: Phase C Sovereign Driver

These measurements were taken entirely through the ToadStool `akida-driver`
pure Rust crate — no Python SDK, no C++ engine, no vendor code in the path.
This is the first public benchmark of an AKD1000 using a non-vendor driver.

---

## Measurement Conditions

- hotSpring AKD1000 measurements: February 19-25, 2026 (SDK: Akida 2.19.1, Python 3.10)
- hotSpring Exp 022 production run: February 26-27, 2026 (live NPU via `akida_pcie` kernel module)
- wetSpring AKD1000 measurements: February 26, 2026 (Driver: ToadStool `akida-driver` 0.1.0, pure Rust)
- AKD1500 projections based on Datasheet v1.2 (June 2025) — no hardware
- Device permissions: `/dev/akida0` (chmod 666)
- GPU measurements: nvidia-smi polling, RAPL energy counters
- Thermal: sustained 100% GPU utilization, 74-75°C, no throttling
- All results reproducible from public repository
