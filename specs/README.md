# rustChip Specifications

**Last Updated**: February 27, 2026
**Status**: Phase D active — VFIO driver functional, `cargo check` clean
**Crates**: akida-chip 0.1.0, akida-driver 0.1.0, akida-models 0.1.0,
           akida-bench 0.1.0, akida-cli 0.1.0

---

## Quick Status

| Component | Status | Notes |
|-----------|--------|-------|
| akida-chip | ✅ Clean | Silicon model — zero deps, zero errors |
| akida-driver (VFIO) | ✅ Functional | Full DMA, IOVA mapping, BAR0 MMIO, power measurement |
| akida-driver (kernel) | ✅ Fallback | /dev/akida* when C module loaded |
| akida-models | ✅ Skeleton | FlatBuffer parser, program_external() injection path |
| akida-bench | ✅ 10 bins | All 10 BEYOND_SDK discoveries + production benchmarks |
| akida-cli | ✅ Functional | enumerate, info, bind-vfio, unbind-vfio, iommu-group |
| docs/ | ✅ Complete | BEYOND_SDK, HARDWARE, TECHNICAL_BRIEF, BENCHMARK_DATASHEET |
| DEPRECATED.md | ✅ | C kernel module clearly marked, migration path documented |

---

## Specifications

| Document | Purpose |
|----------|---------|
| [`SILICON_SPEC.md`](SILICON_SPEC.md) | The chip itself: register map, BAR layout, NP mesh, program format |
| [`DRIVER_SPEC.md`](DRIVER_SPEC.md) | Driver architecture: backend hierarchy, VFIO requirements, API contract |
| [`PHASE_ROADMAP.md`](PHASE_ROADMAP.md) | Sovereign driver roadmap Phase A–E, what's done, what's next |
| [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) | Using rustChip in a downstream project; GPU+NPU co-location pattern |
| [`AI_CONTEXT.md`](AI_CONTEXT.md) | For AI developer context: conventions, extension patterns, crate graph |

---

## Scope

### rustChip IS:
- Pure Rust, standalone software stack for AKD1000 / AKD1500 hardware
- Primary backend: VFIO (no kernel module required)
- Fallback backend: C kernel module (when already installed)
- FlatBuffer model format parser and injection
- Benchmark suite reproducing all 10 BEYOND_SDK hardware discoveries
- Command-line tool for hardware enumeration, info, and VFIO management

### rustChip IS NOT:
- A Python SDK (that is BrainChip's MetaTF)
- A kernel module (the deprecated C code at root is the old one; Phase E queues a Rust one)
- A machine learning training framework (no autograd, no graph compilation)
- Dependent on any ecoPrimals component (toadstool, hotSpring, wetSpring) — completely standalone

### rustChip ENABLES DOWNSTREAM:
- GPU+NPU co-location: `akida-driver` + any Rust GPU crate (wgpu, vulkano, ash)
- The GPU half of the heterogeneous pipeline lives in your project; the NPU half is here
- Online learning: `set_variable()` weight mutation without reprogramming
- Custom model injection: `program_external()` with hand-crafted FlatBuffers

---

## Philosophy

This project is a **fruiting body** — a self-contained expression of the
ecoPrimals methodology, designed to be handed to another team and function
independently. Like a spore, it carries everything it needs to establish
a new colony:

- Code that compiles and runs
- Hardware measurements that justify every design decision
- Documentation that teaches the silicon, not just the API
- A roadmap that continues past the handoff

Three principles govern everything:

**1. Capability-based discovery** — nothing is hardcoded. No `/dev/akida0` assumed,
no PCIe address embedded. Every path is discovered at runtime via sysfs.

**2. Measure, then model** — the register map in `akida-chip/src/regs.rs` came from
direct hardware probing, not from reading a closed datasheet. `confirmed` and
`inferred` labels are explicit.

**3. The C code is deprecated, not deleted** — `akida-pcie-core.c` stays at the root
for upstream reference. We don't erase history; we evolve past it.

---

## Reading Order

**AI developer (orient first, then build)**:
1. `AI_CONTEXT.md` — conventions, crate graph, extension patterns
2. `SILICON_SPEC.md` — what the hardware actually is
3. `DRIVER_SPEC.md` — how the driver is structured

**Engineer evaluating this for integration**:
1. This README (5 min)
2. `INTEGRATION_GUIDE.md` — how to use this in your project
3. `../BEYOND_SDK.md` — the 10 hardware discoveries that justify the approach

**Researcher studying the silicon**:
1. `SILICON_SPEC.md` — register map and NP mesh
2. `../docs/HARDWARE.md` — deep-dive architecture
3. `../docs/BEYOND_SDK.md` — discovery methodology

---

## License

AGPL-3.0-or-later.
