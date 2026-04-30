# rustChip Evolution

**Last Updated**: April 30, 2026
**Status**: Active — VFIO backend live on AKD1000, user-level access via udev

---

## Identity

rustChip is **infrastructure**, not a primal or a spring. It lives at
`infra/rustChip` in the ecoPrimals workspace — alongside `sporePrint`,
`whitePaper`, `fossilRecord`, and other infra artifacts.

It is a **self-sufficient outreach point**: a standalone Rust driver and
benchmark suite for BrainChip Akida neuromorphic processors that anyone
can clone, build, and use without the wider ecoPrimals ecosystem. Its
purpose is to demonstrate what the hardware can do and draw interested
engineers and researchers into the larger project.

In the gen3 licensing framing (`whitePaper/gen3/about/SCYBORG_EXCEPTION_PROTOCOL.md`),
rustChip is a **symbiotic exception candidate** — 100% author-controlled code
offered to BrainChip under reciprocal terms. In the gen4 outreach framing,
it is a practical "what can I do with this?" artifact for builders.

---

## Ecosystem Map

| Organization | Role | Link |
|---|---|---|
| [ecoPrimals](https://github.com/ecoPrimals) | Infrastructure primals (compute, crypto, networking, storage) | [primals.eco](https://primals.eco) |
| [syntheticChemistry](https://github.com/syntheticChemistry) | Science validation (8 springs across physics, biology, agriculture, health) | [primals.eco/springs](https://primals.eco/springs/) |
| [sporeGarden](https://github.com/sporeGarden) | Products (esotericWebb, helixVision, blueFish) | [primals.eco](https://primals.eco) |

**Canonical repo**: [syntheticChemistry/rustChip](https://github.com/syntheticChemistry/rustChip)
(hosted under syntheticChemistry because it is science-facing outreach;
lives locally at `infra/rustChip` because it is infrastructure)

---

## Relationship to toadStool

[toadStool](https://github.com/ecoPrimals/toadStool) is the sovereign
compute hardware primal — GPU/NPU/CPU discovery, tolerance-based routing,
heterogeneous dispatch, 1,910 commits, 20K+ tests. Its neuromorphic crates
(`crates/neuromorphic/akida-driver`, `akida-models`, `akida-setup`, etc.)
are the production-hardened upstream.

rustChip mirrors the NPU subset of toadStool as a standalone extraction.
The relationship is a **one-way port**: toadStool -> rustChip. We port
hardening patterns, error types, and test infrastructure downstream.
We do not create dependencies in either direction.

### What toadStool has that rustChip lacks

- `NpuBackendDispatch` enum dispatch (rustChip uses `Box<dyn NpuBackend>`)
- `toadstool-hw-safe` containment crate
- `akida-reservoir-research` (ensemble, readout, reservoir, state extraction)
- `cross-substrate-validation` (comprehensive substrate benchmark)
- `neurobench-runner` (NeuroBench suite integration)
- JSON-RPC/IPC server infrastructure

### What rustChip has that toadStool lacks

- `akida-chip` — zero-dependency silicon model (register map, NP mesh, BAR layout, SRAM model)
- `HybridEsn` / SRAM / PUF / multi-tenancy / online evolution / sentinel modules
- `SoftwareBackend` — full f32 ESN CPU simulation
- `ProgramBuilder` — layer-by-layer FlatBuffer construction
- Extended `NpuBackend` trait (`verify_load`, `mutate_weights`, `read_sram`)
- `akida-bench` — 12 benchmark binaries (10 BEYOND_SDK + SRAM probe + experiments)
- `akida-cli` — command-line tool with setup/verify/VFIO management

---

## What Has Been Ported (April 2026)

### From toadStool -> rustChip

| Item | Source | Result |
|------|--------|--------|
| `SetupFailed` error variant | `akida-driver/src/error.rs` | Added to rustChip's `AkidaError` enum |
| Unsafe deny policy | Crate-level `deny(unsafe_code)` | Applied with targeted allows on 5 modules (ioctls, dma, container, mmio, mmap) |
| `SyntheticNpuBackend` | `synthetic.rs` | Ported behind `test-mocks` feature flag |
| CLI setup/verify | `akida-setup` binary | Integrated into `akida-cli` as `setup` and `verify` subcommands |

### Standalone fixes (not from toadStool)

| Item | Detail |
|------|--------|
| FBZ parser rewrite | Replaced hardcoded magic bytes with Snappy decompression (`snap` crate). Real `.fbz` files are Snappy-compressed FlatBuffers with zero-padding; parser probes for exact Snappy stream boundary. |
| VFIO ioctl fix | `VFIO_DEVICE_GET_REGION_INFO` constant was `0xc018_3b68` (wrong — used `_IOWR` form with nr=104). Fixed to `_IO(';', 108)` matching kernel VFIO API. BAR mapping now works. |
| VFIO backend validated | VFIO backend init succeeds on live AKD1000 at `0000:e2:00.0`: 80 NPUs, 10 MB SRAM discovered via pure Rust VFIO path. |
| User-level udev rule | Installed `/etc/udev/rules.d/99-akida-vfio.rules` — auto-binds AKD1000 to `vfio-pci` and grants user access to `/dev/vfio/*`. No pkexec at runtime. |
| `vfio_bind` example fix | Vendor/device IDs corrected from `0x1e7f:0x1000` to `0x1e7c:0xbca1`. |
| Format doc reconciliation | Updated 3 conflicting descriptions (SILICON_SPEC, minimal_fc.md, lib.rs) to agree on the actual format |
| Ecosystem docs | README updated with primals.eco links, ecosystem table, toadStool relationship |

### Absorbed from coralReef (April 30, 2026)

| Item | Source | Result |
|------|--------|--------|
| glowplug VFIO lifecycle | `coral-glowplug` | Self-contained device bind/warm-boot/teardown in `akida-driver::glowplug`. No external orchestrator needed. |
| HW/SW backend separation | `BackendSelection` | Explicit `VfioBackend` [HW] / `SoftwareBackend` [SW] labeling. `select_backend()` composition entry point. |
| Lineage principle | scyBorg triple | Absorbed code retains full AGPL-3.0-or-later + ORC + CC-BY-SA 4.0. Akida-specific code exempt; ecosystem-derived systems are not. |

### Science showcase (April 30, 2026)

| Item | Detail |
|------|--------|
| 4 narrative explorations | `WHY_NPU.md`, `SPRINGS_ON_SILICON.md`, `NPU_FRONTIERS.md`, `NPU_ON_GPU_DIE.md` in `whitePaper/explorations/` |
| 5 science demo binaries | `science_lattice_esn`, `science_bloom_sentinel`, `science_spectral_triage`, `science_crop_classifier`, `science_precision_ladder` — each standalone, no external data |
| `warm_boot` binary | Sovereign device lifecycle demo using glowplug |
| Experiment 006 | BAR0 Register Probe: 80 NPs, 10 MB SRAM confirmed via pure userspace VFIO |

---

## Remaining Evolution

### Near-term (Phase D completion)

- **FBZ schema reverse-engineering**: The parser uses heuristic scanning.
  Full FlatBuffer schema-aware parsing needs the `.fbs` schema from
  BrainChip or reverse-engineering from model-zoo `.fbz` files.
  Tracked in [syntheticChemistry/rustChip#1](https://github.com/syntheticChemistry/rustChip/issues/1).
- **Device initialization sequence**: VFIO backend opens and maps BARs
  but the device reports `READY=0`. Need to determine the correct
  reset/enable register sequence from the AKD1000 to bring it to
  command-accepting state.
- **IRQ-based completion**: Currently polling. `VFIO_DEVICE_SET_IRQS`
  infrastructure exists but is not wired.
- **Scatter-gather DMA**: Large payloads currently use single-buffer DMA.
- **MSI-X interrupt vectors**: Not yet configured.

### Medium-term (toadStool port + user-level evolution)

- **User-level operation** (no pkexec at runtime):
  A udev rule (`/etc/udev/rules.d/99-akida-vfio.rules`) is installed
  to auto-bind AKD1000 to `vfio-pci` and grant `0666` permissions on
  `/dev/vfio/*`. This means runtime code never needs privilege escalation.
  The one-time udev install is the only privileged step. Pattern borrowed
  from toadStool's `gpu-mmio` group / `99-ecoprimals-gpu-bar0.rules`.
- **VFIO-aware discovery**: `DeviceManager::discover()` currently only
  scans `/dev/akida*` (kernel module path). Add a VFIO discovery path
  that scans `/sys/bus/pci/devices/` for Akida vendor:device bound to
  `vfio-pci`, matching toadStool's `select_backend(Auto, ...)` pattern
  (Kernel → VFIO → Userspace fallback).
- **Ember fd sharing** (from coralReef): glowplug lifecycle is absorbed
  (see "Absorbed from coralReef" above). The ember fd-sharing pattern
  (immortal fd holder passing duplicates via `SCM_RIGHTS`) is noted for
  future multi-client NPU access but not yet absorbed.
- **Reservoir research patterns**: toadStool's `akida-reservoir-research`
  has ensemble, readout, reservoir, state extraction. rustChip already has
  `HybridEsn` — port ensemble and state extraction as extensions to the
  hybrid module.
- **Cross-substrate validation**: toadStool has comprehensive substrate
  benchmarking. rustChip's `akida-bench` already has `bench_esn_substrate`
  and `bench_hw_sw_parity` — consolidate and extend.
- **Test extraction pattern**: Extract inline tests from large files into
  companion `_tests.rs` files (following toadStool's pattern).

### Long-term (Phase E and beyond)

- **Phase E**: Rust `akida_pcie` kernel module using Linux kernel Rust
  bindings. See `PHASE_ROADMAP.md` for the full sketch.
- **AKD1500 native support**: Device ID already in `pcie.rs`; needs BAR
  size verification and mesh topology for the new chip.
- **P2P DMA**: GPU -> NPU without CPU copy. Requires same IOMMU group.
  See `whitePaper/explorations/GPU_NPU_PCIE.md`.
- **coralReef convergence**: coralReef's `coral-driver` VFIO layer
  (`VfioDevice::open` with iommufd/cdev auto-detect) is PCI-class-agnostic.
  If rustChip moves toward a shared VFIO foundation, the transport layer
  could be extracted into a common crate (analogous to toadStool's
  `toadstool-hw-safe`). The NVIDIA-specific parts (`NvVfioComputeDevice`,
  GR/FECS/PFIFO) do not transfer; only the generic VFIO open/map/DMA path.

---

## What We Are NOT Doing

- **No toadStool dependency.** rustChip must `cargo build` from a fresh clone.
- **No JSON-RPC/IPC server infrastructure.** That's toadStool's domain.
- **No `NpuBackendDispatch` enum dispatch yet.** rustChip's `Box<dyn NpuBackend>`
  works and the extended trait methods (`verify_load`, `mutate_weights`,
  `read_sram`) make enum dispatch harder without losing functionality.
- **No touching toadStool's code.** The port is one-way.
- **No Python bindings.** Use `akida-cli` as a subprocess if needed.
- **No model training.** rustChip is an inference driver. Training belongs
  in the scientific computing projects.

---

## License

scyBorg triple-copyleft: AGPL-3.0-or-later (code) + ORC (game mechanics) +
CC-BY-SA 4.0 (creative/docs). Akida-specific code is a symbiotic exception
candidate; ecosystem-derived systems (glowplug, science demos, etc.) retain the
full scyBorg license under the lineage principle.
