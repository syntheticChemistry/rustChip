# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **glowplug absorption** — VFIO device lifecycle management (bind, warm boot,
  tear down) absorbed from coralReef's ember/glowplug. rustChip now manages its
  own hardware lifecycle without external orchestrator or kernel module. Code is
  derivative of the ecoPrimals ecosystem and retains the full scyBorg license
  under the lineage principle.
- **HW/SW backend separation** — `VfioBackend` [HW] and `SoftwareBackend` [SW]
  are explicitly labeled and never conflated. `BackendSelection` enum and
  `select_backend()` function provide the composition entry point.
- **Narrative explorations** — four new whitePaper documents grounding NPU science
  in hardware evidence:
  - `WHY_NPU.md` — the foundational neuromorphic argument
  - `SPRINGS_ON_SILICON.md` — 5 NPU patterns × 3 science domains
  - `NPU_FRONTIERS.md` — 10 creative frontiers for neuromorphic hardware
  - `NPU_ON_GPU_DIE.md` — NPU as a GPU functional unit (area/power analysis)
- **5 standalone science demos** — self-contained binaries reproducing
  peer-reviewed NPU science claims without external data:
  - `science_lattice_esn` — Hybrid ESN for Lattice QCD steering
  - `science_bloom_sentinel` — Streaming Sentinel for harmful algal bloom detection
  - `science_spectral_triage` — Microsecond Gatekeeper for LC-MS spectral triage
  - `science_crop_classifier` — Online Adaptation via (1+1)-ES for seasonal crop stress
  - `science_precision_ladder` — Precision Discipline: f64 → f32 → int8 → int4
- **warm boot binary** — `warm_boot` demonstrates sovereign device lifecycle
  using the absorbed glowplug module.
- **Experiment 006** — BAR0 Register Probe: true layout discovery on cold-boot
  VFIO-bound AKD1000 (80 NPs, 10 MB SRAM confirmed via pure userspace probing).

### Changed

- **scyBorg licensing update** — AGPL-3.0-or-later (code) + ORC (game mechanics)
  + CC-BY-SA 4.0 (creative/docs). Lineage principle: code absorbed from
  ecoPrimals retains full scyBorg license even within the BrainChip exception
  boundary. Akida-specific code exempt; systems endowed from ecoPrimals are not.
- Test count: 353 → 367 (new science demos, glowplug module).
- Zoo model count corrected: 29 → 28 (actual `ZooModel` enum variants).

## [0.1.0] — 2026-04-30

First versioned release. 353 tests passing, 29-model zoo, pure Rust
conversion pipeline, guideStone validation, CI, full documentation suite.

### Added

- **Pure Rust model pipeline** — complete replacement for Python SDK dependency.
  Models can now be created, quantized, serialized, and parsed entirely in Rust:
  - `akida-models::quantize` — per-layer and per-channel int1/2/4/8 symmetric
    quantization with nibble packing and round-trip verification (14 tests).
    Algorithm patterns absorbed from neuralSpring and hotSpring.
  - `akida-models::import` — weight import from `.npy`, `.safetensors`, and raw
    f32 slices with hand-rolled `.npy` parser and f16/bf16 upcast (16 tests).
    safetensors pattern absorbed from neuralSpring.
  - `akida-models::schema` — reverse-engineered FlatBuffer serializer matching
    the structure observed across all 25 `.fbz` artifacts. Produces valid
    Snappy-compressed FlatBuffers that round-trip through the parser (5 tests).
  - `akida-models::schema_parser` — schema-aware FlatBuffer parser using vtable
    navigation instead of heuristic byte scanning. Extracts version, layer names,
    and properties via proper field offsets (6 tests).
  - `akida convert` CLI command — end-to-end conversion: import weights from
    `.npy`/`.safetensors`/`zeros:N`/`random:N`, quantize, serialize via
    FlatBuffer, compress, write `.fbz`, verify round-trip. No Python.
  - `.fbs` schema at `crates/akida-models/schemas/akida_model.fbs` documents the
    reverse-engineered format.
- **New dependencies**: `flatbuffers = "25"`, `safetensors = "0.5"` (both pure
  Rust, no C/Python/vendor SDK).
- **guideStone computational artifact** — `akida guidestone [dir]` runs a
  self-leveling validation: parses 25 models (21 BrainChip zoo + 4 physics),
  computes SHA-256 digests, validates structure (version, layers, decompression
  ratio, NP budget, weight blocks, file size), benchmarks parse throughput
  (15.8 MB/s), and emits a graded report (225 checks, 0 failures). Implemented
  in `crates/akida-models/src/guidestone.rs`. Any subsequent work on this
  build can reference the guideStone run as the anchored baseline.
- **guideStone documentation** — `specs/GUIDESTONE.md` maps the ecoPrimals
  guideStone verification class (5 properties) to NPU compute.
  `baseCamp/GUIDESTONE_CERTIFICATION.md` tracks certification progress.
- **baseCamp documentation suite** — `ZOO_GUIDE.md` (comprehensive model
  zoo guide with all 25 models, Rust ecosystem integration, and what it
  unlocks), `SCIENTIFIC_DEPLOYMENT.md` (whitepaper-style NPU deployment
  guide for scientific/data workloads with production results and deployment
  patterns), `RUST_NPU_ECOSYSTEM.md` (what the Rust NPU ecosystem enables
  and how rustChip fits).
- **Full BrainChip model zoo export** — `scripts/export_zoo.py` exports all 21
  pretrained models from the Akida SDK (v2.19.1) to `.fbz` with a ground-truth
  `zoo_manifest.json`. Python SDK is a one-time validation oracle, not a runtime
  dependency.
- **Physics model export** — `scripts/export_physics.py` generates `.fbz` files
  for the 4 ecoPrimals physics models (ESN readout, phase classifier, transport
  predictor, Anderson classifier) with random weights matching documented
  architectures.
- **Zoo regression test suite** — `crates/akida-models/tests/zoo_regression.rs`
  validates that all 25 `.fbz` files (21 zoo + 4 physics) parse successfully.
  Run with `cargo test --test zoo_regression -- --ignored`.
- **CLI model commands** — `akida parse <file.fbz>` inspects model structure;
  `akida zoo-status [dir]` shows cache status against the full `ZooModel` enum.
- **Spring profiles catalog** — `baseCamp/spring-profiles/README.md` documents
  every NPU workload across ecoPrimals springs with reproducible instructions.
- **`Model::input_shape()` / `output_shape()`** — shape accessors on the parsed
  model, populated via `set_shapes()` from manifest metadata.
- **`ZooModel` enum expanded to 28 variants** — covers all 21 BrainChip
  MetaTF models, 2 NeuroBench benchmarks, 4 ecoPrimals physics models, and 1
  hand-built test. Filenames match actual export output.
- **`ZooModel::brainchip_zoo()`** — convenience accessor for the 21 MetaTF models.
- **`ModelTask::FaceAnalysis`, `Segmentation`, `EyeTracking`** — new task
  categories for the expanded model catalog.

### Changed

- **`ModelZoo::load_metadata`** now parses files via `Model::from_bytes()` to
  determine validity, version, and layer count. The old magic-byte check
  (`[0x80, 0x44, 0x04, 0x10]`) rejected all real zoo `.fbz` files.
- **`ValidationTier::Functional`** assigned to all 21 MetaTF zoo models
  (exported + parsed successfully).

### Fixed

- `ModelZoo::load_metadata` no longer marks real `.fbz` files as invalid.
- `baseCamp/conversion/README.md` no longer references non-existent `quantize.rs`.
- `baseCamp/zoos/brainchip_metatf.md` updated with actual export results.

### Added (0.1.0 release hardening)

- **QUICKSTART.md** — 5-command onboarding: clone, build, convert, parse, test.
- **LEVERAGE.md** — standalone tool leverage guide following wateringHole pattern.
- **Nature Preserve** — `baseCamp/preserve/` with 7 domain application patterns
  (physics, biology, audio, vision, environmental, genomic, industrial).
- **4 Rust-native zoo models** — ESN multi-head, ESN 3-head transport, streaming
  sensor 12ch, adaptive sentinel. Generated via pure Rust `akida convert`.
- **NPU_LEVERAGE.md** added to hotSpring, wetSpring, neuralSpring.
- **sporePrint science page 27** — Nature Preserve: Applied NPU Science.
- **wateringHole** — rustChip entry in `LEVERAGE_GUIDES.md` (v1.1.0), NPU
  onboarding section in `GARDEN_COMPOSITION_ONRAMP.md`.
- **CONTRIBUTING.md** — quality commands, layout, workflow.
- **CITATION.cff** — machine-readable citation metadata.
- **CONTEXT.md** — AI/indexer context following `PUBLIC_SURFACE_STANDARD`.
- **CI workflows** — `.github/workflows/ci.yml` with fmt, clippy, test, doc, deny.
- **TOLERANCE_REGISTRY.md** — named numerical tolerances across all subsystems.
- **justfile** — common workflow commands (`just check`, `just test`, `just guidestone`).
- **.cursor/rules/rustchip.md** — project-level AI guidance.
- **16 new tests** — 8 in akida-cli (arch parser), 8 in akida-bench (PRNG, timer, probe).
- README header reformatted to match `SPRING_PRIMAL_PRESENTATION_STANDARD`
  (Date, License, MSRV, Status with metrics).
