# AKD1000 Benchmark Results

Benchmark results are organized by experiment. Each experiment corresponds
to a run of the `akida-bench` suite with a specific hardware configuration
and objective.

---

## How to Run

```bash
# From repository root
cd /path/to/rustChip

# Single benchmark
cargo run --bin bench_latency
cargo run --bin bench_dma
cargo run --bin bench_batch
cargo run --bin bench_clock_modes
cargo run --bin bench_fc_width
cargo run --bin bench_fc_depth
cargo run --bin bench_channels
cargo run --bin bench_weight_mut
cargo run --bin bench_bar

# Full suite (order matters for warmup)
for bin in enumerate bench_dma bench_latency bench_channels \
           bench_fc_depth bench_batch bench_clock_modes \
           bench_fc_width bench_weight_mut bench_bar; do
    echo "=== $bin ==="
    cargo run --bin $bin 2>/dev/null
    echo ""
done
```

Set `RUST_LOG=debug` to see VFIO setup, DMA map, and register access logs.
Set `RUST_LOG=info` for timing-relevant events only.

---

## Reference Values

These are the validated reference measurements. New runs should produce
values within ~10% of these figures on AKD1000 hardware.

| Benchmark | Reference | Source |
|-----------|-----------|--------|
| DMA throughput | 37 MB/s | Exp 001, confirmed in Exp 022 |
| Single inference | 54 µs | Exp 001 |
| Batch=8 per-sample | 390 µs | Exp 001 |
| Batch speedup | 2.4× | Exp 001 |
| Clock: Economy latency | +19% vs Performance | Exp 001 |
| Clock: Economy power | −18% vs Performance | Exp 001 |
| FC depth overhead (×8) | +3 µs | Exp 001 |
| Channel count range | 1–64 all work | Exp 001 |
| FC width crossover | ~512 neurons | Exp 001 |
| Weight mutation overhead | ~14 ms | Exp 001 |
| BAR1 size | 16 GB | Exp 001 |

---

## Results Index

| Experiment | Date | Key Results |
|------------|------|-------------|
| [exp001_baseline](exp001_baseline.md) | Feb 27, 2026 | Baseline characterization: all BEYOND_SDK discoveries confirmed |
