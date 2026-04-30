# test/ — Historical C Reference Tests

This directory contains the original C test harness from the BrainChip
`akida_dw_edma` kernel module era. These files are **historical reference
only** — they are not part of the active test suite.

## Contents

| File | What it tested |
|------|---------------|
| `test.c` | Basic kernel module DMA transfer |
| `test_host_ddr.c` | Host DDR DMA path |
| `mmap_access.c` | BAR mmap from userspace via `/dev/mem` |
| `akd500_akdma_rc.sh` | AKD500 DMA regression check |
| `Makefile` | Build harness for the C tests |
| `spdk/` | SPDK-based DMA test infrastructure |

## Where the Real Tests Live

All active Rust tests are in the crate source directories:

```bash
cargo test --workspace              # run all 367 tests
cargo test -p akida-models          # model parsing, quantization, zoo
cargo test -p akida-driver          # backend, hybrid ESN, SRAM, evolution
cargo test -p akida-chip            # register map, NP mesh, BAR layout
cargo test -p akida-bench           # benchmark infrastructure
cargo test -p akida-cli             # CLI argument parsing
```

The science demo binaries are standalone executables, not test-framework tests:

```bash
cargo run --bin science_lattice_esn
cargo run --bin science_bloom_sentinel
cargo run --bin science_spectral_triage
cargo run --bin science_crop_classifier
cargo run --bin science_precision_ladder
```
