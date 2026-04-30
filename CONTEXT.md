# rustChip — Context for AI Assistants and Indexers

**Type:** Infrastructure tool (gen2.5)
**License:** AGPL-3.0-or-later
**Language:** Rust (edition 2024)
**Tests:** 367 unit/integration tests, all passing
**Models:** 28-model zoo (21 BrainChip + 4 physics + 2 NeuroBench + 1 hand-built)

## What This Is

Pure Rust neuromorphic inference stack for BrainChip Akida processors
(AKD1000, AKD1500). Standalone tool from the ecoPrimals ecosystem.

## Crate Map

```
akida-chip    — silicon model (register map, NP mesh, BAR layout)
akida-driver  — VFIO passthrough driver, software backend, hybrid ESN, glowplug VFIO lifecycle
akida-models  — FBZ parser, quantization, import, serialization, zoo
akida-bench   — benchmark and experiment binaries, 5 standalone science demos
akida-cli     — `akida` command-line tool
```

## Entry Points

- **For coding:** `specs/AI_CONTEXT.md` — naming conventions, hardware rules
- **For use:** `QUICKSTART.md` — 5-command onboarding
- **For science:** `baseCamp/preserve/README.md` — 7 domain patterns
- **For ecosystem:** `LEVERAGE.md` — standalone and integrated usage
- **For narratives:** `whitePaper/explorations/WHY_NPU.md` — the neuromorphic argument

## Recent Additions (April 2026)

- **glowplug absorption** — VFIO device lifecycle management absorbed from coralReef's ember/glowplug, providing sovereign warm boot without external orchestrator
- **HW/SW backend separation** — `VfioBackend` [HW] and `SoftwareBackend` [SW] are explicit and never conflated
- **Narrative explorations** — `WHY_NPU.md`, `SPRINGS_ON_SILICON.md`, `NPU_FRONTIERS.md`, `NPU_ON_GPU_DIE.md`
- **Science demos** — 5 standalone binaries reproducing NPU science claims: `science_lattice_esn`, `science_bloom_sentinel`, `science_spectral_triage`, `science_crop_classifier`, `science_precision_ladder`
- **scyBorg licensing** — AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative/docs); lineage principle for absorbed code

## Key Conventions

- `unsafe_code = "forbid"` at workspace level (driver uses targeted `deny`)
- `missing_docs = "deny"` — all public items documented
- Clippy pedantic + nursery enabled
- Named tolerances only — no magic numbers (see `TOLERANCE_REGISTRY.md`)
- `.fbz` = Snappy-compressed FlatBuffer model format
- Model parsing: `Model::from_file()` or `Model::from_bytes()`
- Conversion: `akida convert --weights <src> --arch <layers> --bits <n>`

## Repository

- **Canonical:** <https://github.com/syntheticChemistry/rustChip>
- **Ecosystem:** <https://github.com/ecoPrimals> / <https://primals.eco>
