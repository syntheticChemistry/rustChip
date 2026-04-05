// SPDX-License-Identifier: AGPL-3.0-or-later

# rustChip — Context

## What
Pure-Rust neuromorphic AI driver stack for BrainChip Akida (AKD1000/AKD1500). 5-crate workspace: akida-chip (register maps), akida-driver (VFIO/mmap/hybrid ESN), akida-models (model loading/inference), akida-bench (benchmarks), akida-cli (device CLI).

## Status
- **Grade A** per ecosystem compliance matrix
- Edition 2024, AGPL-3.0-or-later
- 237 tests, 60.8% coverage (software-testable; hardware VFIO/mmap excluded)
- Clippy pedantic+nursery clean (0 warnings)
- 31 unsafe blocks documented with // SAFETY: comments
- deny(unsafe_op_in_unsafe_fn) enforced workspace-wide
- forbid(unsafe_code) on all crates except akida-driver (VFIO/DMA)

## Ecosystem Role
Tool (gen2.5) — standalone hardware interface library. Not a long-running daemon. Used by primals and springs for neuromorphic compute offload.

## Key Dependencies
- libc (VFIO ioctls — hardware access, no pure-Rust alternative)
- No other C dependencies
