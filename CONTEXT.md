# rustChip — Context

BrainChip Akida NPU register-level driver, model format parser, and benchmarks.
Pure Rust implementation — no C++ SDK dependency. Symbiotic exception with BrainChip.
Extracted from toadStool metalForge, evolves via hotSpring.

## Workspace Structure

| Crate | Role | Type |
|-------|------|------|
| `akida-chip` | Register map, firmware interface, DMA control | library |
| `akida-driver` | Host-side driver over USB/PCIe, async command execution | library |
| `akida-models` | Model format (.fbz), layer descriptions, metadata | library |
| `akida-bench` | Benchmark harness for inference latency and throughput | library |
| `akida-cli` | CLI for chip info, firmware, model loading, inference | binary |

## Status

v0.1.0 — Early stage. Register maps defined, model parsing functional, driver stubs in place.
Ecosystem tool (gen2.5). Consumed by toadStool and hotSpring.
