# Tolerance Registry

Named numerical tolerances used across rustChip. No magic numbers in tests.

**Date:** April 30, 2026
**Convention:** Every numerical comparison in a test must reference a named
tolerance. Tolerances are justified by physics, quantization theory, or
hardware measurement — not tuned to pass.

---

## Quantization Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `QUANTIZATION_FLOOR` | 1e-30 | — | Prevents division by zero in scale computation | `akida-models::quantize` |
| `INT8_MAX_ABS_ERROR` | 0.004 | fraction of scale | Symmetric int8: max error = scale/2 ≈ 1/254 | quantize tests |
| `INT4_MAX_ABS_ERROR` | 0.0625 | fraction of range | Symmetric int4: max error = scale/2 ≈ 1/14 | quantize tests |
| `INT4_SCALE_HALF_BOUND` | scale/2 + 1e-5 | adaptive | Per-layer max error bounded by half the quantization step | `max_quantization_error_within_half_scale` |
| `INT1_SCALE_BOUND` | scale + 1e-5 | adaptive | 1-bit: error bounded by full scale (only 2 representable values) | `max_quantization_error_within_half_scale` |
| `ROUND_TRIP_EPSILON` | 1e-5 | f32 | Float→int→float round-trip accumulated error | pack/unpack tests |

---

## Parser Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `DECOMPRESS_RATIO_MIN` | 0.5 | ratio | Snappy compression never expands > 2× | zoo regression |
| `DECOMPRESS_RATIO_MAX` | 100.0 | ratio | Pathological compression ratio ceiling | zoo regression |
| `MIN_LAYER_COUNT` | 1 | count | Every valid model has at least one layer | zoo regression |
| `PARSE_THROUGHPUT_MIN` | 1.0 | MB/s | Minimum acceptable parser throughput | guidestone |

---

## Hardware Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `LATENCY_CEILING` | 100 | µs | AKD1000 single-inference upper bound (measured: 54 µs) | bench_latency |
| `THROUGHPUT_FLOOR` | 10_000 | Hz | AKD1000 minimum sustained throughput (measured: 18,500 Hz) | bench_latency |
| `NP_COUNT_AKD1000` | 78–80 | NPs | Discovered via BAR0 probe (spec says 80, measured 78) | silicon spec |
| `SRAM_SIZE_AKD1000` | 8–10 | MB | BAR1 physical SRAM (spec 8 MB, decode window 16 GB) | silicon spec |
| `CHIP_POWER_TYPICAL` | 30 | mW | Measured chip power during inference | bench_latency |
| `ENERGY_PER_INFERENCE` | 1.4 | µJ | Derived: 30 mW × 54 µs ≈ 1.62 µJ, rounded | bench_latency |

---

## GuideStone Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `GUIDESTONE_CHECK_COUNT` | 225 | checks | 25 models × 9 checks each | guidestone binary |
| `GUIDESTONE_FAILURE_MAX` | 0 | count | Zero failures required for guideStone pass | guidestone binary |
| `GUIDESTONE_THROUGHPUT_MIN` | 1.0 | MB/s | Minimum parse throughput for guideStone pass | guidestone binary |

---

## Multi-Tenancy Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `MAX_CONCURRENT_MODELS` | 7 | models | Measured: 814/1000 NPs used with 7 models | multi_tenancy |
| `CHIP_NPS` | 1000 | NPs | AKD1000 total NP budget | zoo.rs |
| `ISOLATION_SRAM_OVERLAP` | 0 | bytes | SRAM isolation: zero cross-slot data leakage | bench_exp002 |

---

## Evolution Tolerances

| Name | Value | Unit | Source | Used in |
|------|-------|------|--------|---------|
| `EVOLUTION_GEN_RATE_MIN` | 100 | gen/sec | Minimum acceptable evolution rate (measured: 136) | evolution |
| `WEIGHT_SWAP_LATENCY_MAX` | 100 | µs | Maximum per-weight swap (measured: 86 µs) | evolution |

---

## Convention

When adding a new tolerance:

1. Give it a descriptive `SCREAMING_SNAKE_CASE` name.
2. Document the source: physics derivation, hardware measurement, or spec reference.
3. Add it to this registry.
4. Reference it by name in test code, never as a bare literal.
