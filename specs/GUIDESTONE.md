# guideStone for NPU Compute

**Date:** April 29, 2026
**Status:** Pre-certification — 3 of 5 properties substantially met,
           2 require packaging work
**Standard:** `wateringHole/fossilRecord/consolidated-apr2026/GUIDESTONE_STANDARD.md`
**Concept Paper:** `whitePaper/gen4/architecture/GUIDESTONE.md`

---

## What guideStone Means for rustChip

guideStone is the ecoPrimals verification class — a quality certification
for artifacts whose output is its own proof of correctness. It is orthogonal
to binary type (ecoBin) and deployment class (NUCLEUS/Niche/fieldMouse).

For rustChip, guideStone certification means: **any user with the binary,
a `.fbz` file, and optionally an AKD1000 can reproduce every claim this
project makes — parse results, timing measurements, physics classifications —
without trusting the producer.**

This is not hypothetical. The model zoo regression suite already runs 25
models through the parser and validates structure, decompression, and layer
counts. The physics models already trace to published papers. The hardware
measurements already come from direct BAR0/BAR1 probing. The gap is
packaging — assembling these existing pieces into a self-verifying,
tolerance-documented artifact that satisfies all five guideStone properties.

---

## The Five Properties Applied to NPU Compute

### Property 1: Deterministic Output

**Requirement:** Same input, same binary, any hardware → same output within
named tolerances.

**NPU-specific meaning:** The Akida AKD1000 is hardware-deterministic.
Same model, same input, same NP configuration → same output, always. This
is a feature of the neuromorphic architecture (event-based, integer
arithmetic, no floating-point rounding variance). The vendor confirms this
and our measurements verify it across thousands of inferences.

**rustChip status:**

| Component | Deterministic? | Evidence |
|-----------|---------------|----------|
| FBZ parser | Yes | Same `.fbz` → identical `Model` struct every time |
| Snappy decompression | Yes | `snap` crate is pure Rust, deterministic |
| Layer extraction | Yes | Heuristic scan is input-deterministic |
| Hardware inference (AKD1000) | Yes | 5,978 identical-input runs, identical output |
| `SyntheticNpuBackend` | Yes | Deterministic mock, same weights → same output |

**What's already done:**
- Parser produces identical output on every run (tested on 25 models)
- Hardware inference is deterministic (measured in hotSpring Exp 022)
- `SyntheticNpuBackend` provides a hardware-free deterministic path

**What's missing:**
- Cross-architecture validation. Parser is tested on x86_64 only. Need
  aarch64 (at minimum via cross-compilation and qemu-user). The parser
  uses no platform-dependent arithmetic, so this is expected to pass, but
  guideStone requires the measurement, not the expectation.
- Formal determinism test: run parser on 25 models × 2 architectures,
  compare output byte-for-byte.

**Confidence: HIGH** — the underlying compute (integer arithmetic on both
the Rust side and the silicon side) is inherently deterministic. This
property needs packaging, not fundamental work.

---

### Property 2: Reference-Traceable

**Requirement:** Every numeric claim traces to a verifiable source.

**NPU-specific meaning:** Every model architecture, every throughput
measurement, every NP budget, every energy figure must trace to either
(a) a published paper, (b) a hardware measurement with documented
methodology, or (c) the BrainChip SDK as the reference oracle.

**rustChip status:**

| Claim | Source | Traceable? |
|-------|--------|-----------|
| 21 BrainChip models parse | `scripts/export_zoo.py` → SDK v2.19.1 | Yes — SDK is the oracle |
| 4 physics models parse | `scripts/export_physics.py` | Yes — architectures in `baseCamp/models/physics/` |
| Parser throughput 13.7 MB/s | `benchmark_parse_throughput` test | Yes — measured, repeatable |
| 18,500 Hz inference throughput | hotSpring Exp 022 | Yes — paper: Bazavov & Chuna |
| 54 µs chip latency | Phase C + D direct measurement | Yes — `SILICON_SPEC.md` methodology |
| 1.4 µJ/inference energy | BrainChip datasheet + Exp 022 | Yes — datasheet + measurement |
| 1,000 NPs on AKD1000 | BAR0 register read (0x10C0) | Yes — `Capabilities::from_bar0()` |
| β_c = 5.69 for SU(3) | Bazavov et al., PRD 85, 054503 | Yes — published comparison |
| W_c = 16.26 ± 0.95 Anderson | groundSpring Exp 028 | Yes — documented experiment |
| `set_variable()` at 86 µs | Phase C measurement | Yes — `BEYOND_SDK.md` Discovery 6 |
| 37 MB/s DMA throughput | Phase C measurement | Yes — `SILICON_SPEC.md` |

**What's already done:**
- Physics model claims trace to published papers
- Hardware measurements trace to documented methodology in `SILICON_SPEC.md`
  and `docs/BEYOND_SDK.md`
- SDK version (2.19.1) is recorded in `zoo_manifest.json`
- NP budgets derived from model architecture analysis

**What's missing:**
- `zoo_manifest.json` should include SDK commit hash, not just version
- Physics model claims should link to specific experiment IDs with
  DOIs or arXiv identifiers in machine-readable metadata
- NP budget estimates for non-physics models are approximated, not
  measured on hardware

**Confidence: HIGH** — the traceability infrastructure exists. The gap
is in machine-readable metadata format, not in actual provenance.

---

### Property 3: Self-Verifying

**Requirement:** The artifact carries its own integrity mechanism.

**NPU-specific meaning:** The model files, the binary, and the reference
data all carry integrity proofs. A corrupted `.fbz` file is detected
before it reaches the hardware.

**rustChip status:**

| Mechanism | Present? | Details |
|-----------|---------|---------|
| CHECKSUMS file | **No** | Not yet generated |
| `.fbz` integrity | Partial | Snappy decompression fails on corruption; FlatBuffer structure checks detect truncation |
| Manifest integrity | **No** | `zoo_manifest.json` has no hash of each `.fbz` |
| Binary integrity | **No** | No signed release artifacts yet |
| Regression test as integrity | Yes | `zoo_regression.rs` validates all 25 models parse correctly |

**What's already done:**
- Snappy decompression is a de facto integrity check (compressed data
  with invalid checksums fails to decompress)
- FlatBuffer parsing rejects malformed structures
- Regression tests catch parser regressions

**What's missing:**
- CHECKSUMS (SHA-256) file in `baseCamp/zoo-artifacts/`
- Per-model SHA-256 in `zoo_manifest.json` and `physics_manifest.json`
- `akida verify-artifact` CLI command that validates checksums before
  any model operation
- Signed release artifacts (requires bearDog integration — future)

**Confidence: MEDIUM** — the raw mechanisms exist (Snappy checks, parser
validation), but the explicit CHECKSUMS infrastructure that guideStone
requires is not built yet. This is straightforward work.

---

### Property 4: Environment-Agnostic

**Requirement:** No hardcoded paths. No "install X first." No platform
assumptions. Pure Rust, static musl, cross-arch.

**NPU-specific meaning:** The binary discovers its own hardware context
at runtime. No assumptions about PCIe addresses, IOMMU groups, device
paths, or kernel versions.

**rustChip status:**

| Requirement | Met? | Evidence |
|-------------|------|---------|
| Pure Rust | Yes | All crates, zero C dependencies |
| No runtime Python | Yes | Python is validation oracle only |
| No sudo (after setup) | Yes | VFIO + udev rules |
| No hardcoded paths | Yes | `Capabilities::from_bar0()`, sysfs discovery |
| No GPU required | Yes | NPU operates independently |
| Static musl | **Not yet** | Builds with glibc; musl cross-compile not tested |
| Cross-arch (aarch64) | **Not yet** | x86_64 only so far |
| CPU-only path | Yes | `SyntheticNpuBackend` provides full software fallback |
| Runtime substrate detection | Yes | `akida verify` reports discovered hardware |

**What's already done:**
- Pure Rust, no C dependencies, no Python at runtime
- Runtime hardware discovery via sysfs and BAR0 reads
- `SyntheticNpuBackend` for CI and hardware-free environments
- CLI reports discovered substrate (`akida verify`, `akida enumerate`)

**What's missing:**
- Static musl build profile (`RUSTFLAGS='-C target-feature=+crt-static'`
  or `x86_64-unknown-linux-musl` target)
- aarch64-unknown-linux-musl cross-compilation
- Formal cross-substrate test: parse 25 models on x86_64 + aarch64,
  compare `Model` struct outputs

**Confidence: HIGH** — Rust's cross-compilation story makes this mechanical.
The parser has no platform-dependent code (no SIMD, no inline assembly,
no system calls). The driver has VFIO ioctls that are Linux-specific by
design, but the parser/model layer is truly portable.

---

### Property 5: Tolerance-Documented

**Requirement:** Every threshold has a derivation. No magic numbers.

**NPU-specific meaning:** Parsing thresholds, timing tolerances, accuracy
claims, and decompression ratio bounds all have documented origins.

**rustChip status:**

| Threshold | Derived? | Source |
|-----------|---------|--------|
| Int4 weight range [-8, 7] | Yes | Akida hardware specification (4-bit signed) |
| Int4 activation range [0, 15] | Yes | Akida hardware specification (4-bit unsigned) |
| 51-bit threshold SRAM width | Yes | Direct BAR0 read (`SILICON_SPEC.md`) |
| Parse throughput > 1 MB/s | **No** | Empirical observation, no derivation |
| Decompression ratio < 5:1 typical | **No** | Observed across 25 models, not derived |
| NP budget estimates (67–380) | Partial | Derived from architecture for physics models; approximated for zoo models |
| β_c = 5.69 tolerance ± 0.01 | Yes | Finite-size scaling analysis (Bazavov et al.) |
| W_c = 16.26 tolerance ± 0.95 | Yes | Finite-size scaling (groundSpring Exp 028) |
| 100% phase classification accuracy | Yes | Full test set, binary classification |
| 3–4% transport prediction error | Partial | Observed mean relative error; stochastic training noise not formally bounded |

**What's already done:**
- Hardware register values are directly measured with documented methodology
- Physics model tolerances trace to published analyses
- Weight quantization ranges are hardware-specified

**What's missing:**
- Parse throughput threshold needs a derivation (minimum acceptable for
  real-time model loading based on PCIe bandwidth and model sizes)
- Decompression ratio bounds need a derivation (Snappy compression
  characteristics for neural network weight matrices)
- Transport prediction error needs formal error bounds (currently just
  mean observed error)
- Machine-readable tolerance metadata in output JSON (currently tolerances
  live in documentation, not in program output)

**Confidence: MEDIUM** — the physics tolerances are well-derived. The
software/parser tolerances are empirical and need derivations.

---

## Current Certification Status

```
Property 1: Deterministic Output      ██████████░  90%  (needs cross-arch validation)
Property 2: Reference-Traceable       ████████░░░  80%  (needs machine-readable metadata)
Property 3: Self-Verifying            ████░░░░░░░  40%  (needs CHECKSUMS infrastructure)
Property 4: Environment-Agnostic      ███████░░░░  70%  (needs musl + aarch64 builds)
Property 5: Tolerance-Documented      ██████░░░░░  60%  (needs software tolerance derivations)
```

**Overall: Pre-certification.** The foundations are strong — the hard parts
(deterministic hardware, traced measurements, pure Rust) are done. The
remaining work is packaging and formalization.

---

## Path to Certification

### Phase G1: Self-Verifying Infrastructure

1. Generate `CHECKSUMS` (SHA-256) in `baseCamp/zoo-artifacts/`
2. Add per-model SHA-256 to `zoo_manifest.json` and `physics_manifest.json`
3. Add `akida verify-artifact <dir>` CLI command
4. Regression test validates checksums before parsing

### Phase G2: Cross-Architecture Validation

1. Add `x86_64-unknown-linux-musl` build target
2. Add `aarch64-unknown-linux-musl` cross-compilation
3. Parse all 25 models on both architectures
4. Byte-compare `Model` struct outputs (determinism proof)

### Phase G3: Machine-Readable Provenance

1. Extend `zoo_manifest.json` with SDK commit hash
2. Add tolerance derivations to parser output JSON
3. Add DOI/arXiv links for physics model claims
4. `akida parse --guidestone` flag outputs full provenance metadata

### Phase G4: Artifact Packaging

1. Create `rustChip-guideStone-v0.1.0` directory structure:
   ```
   rustChip-guideStone-v0.1.0/
   ├── bin/
   │   ├── akida-x86_64-linux-musl
   │   └── akida-aarch64-linux-musl
   ├── models/
   │   ├── zoo-artifacts/      (25 .fbz files)
   │   └── CHECKSUMS
   ├── expected/
   │   ├── parse_results.json  (reference output for all 25 models)
   │   └── CHECKSUMS
   ├── run                     (entry point: ./run validate)
   ├── liveSpore.json          (tracks machines visited)
   └── README.md
   ```
2. `./run validate` — parse all models, compare to expected, report
3. `./run benchmark` — time parsing, report throughput
4. `./run hardware` — if AKD1000 present, run inference validation

### Phase G5: guideStone Naming

Adopt ecosystem naming convention:
- `rustChip-guideStone-v0.1.0` — first certified release
- `-guideStone-` infix in release naming for plasmidBin distribution

---

## What guideStone Enables for rustChip

### For the project itself

**Credibility transfer.** When rustChip claims "our parser handles the
full Akida model zoo," the guideStone artifact is the proof. Anyone runs
`./run validate` and sees 25/25 models parse with matching reference
output. The claim is not documentation — it is executable.

**Regression detection.** The guideStone artifact is a pinned snapshot.
If a code change breaks parsing of model 14, the next `./run validate`
catches it. This is what the regression suite already does, but packaged
as a portable, cross-architecture artifact.

**Hardware onboarding.** Hand someone a USB drive with the guideStone
artifact. They plug in their AKD1000, run `./run hardware`, and see
inference results matching the reference. The hardware works. The driver
works. The conversation starts from evidence, not from "it should work."

### For the wider ecosystem

**Compute as a math object.** The guideStone standard treats computation
as a mathematical object — deterministic, traceable, self-verifying. For
NPU compute, this means: the inference result is a property of the model
and the input, not of the machine or the software stack. The guideStone
proves this by reproducing the result on any substrate.

**The metrological analogy.** Before 2019, the kilogram was a physical
artifact. After 2019, it was a mathematical constant (Planck). guideStone
does the same for compute — the inference result is defined by the
mathematics (model weights × input), not by the instrument (which NPU,
which driver, which OS). The guideStone is the proof that the instrument
is faithful to the mathematics.

**Sovereign science pipeline.** A guideStone-certified rustChip can be
composed into the full ecoPrimals provenance chain:

```
rustChip guideStone (reproducible NPU inference)
  → bearDog signing (integrity, identity)
    → rhizoCrypt DAG (computation trace)
      → loamSpine certificate (permanent record)
        → BTC/ETH anchor (public chain proof)
          = Novel Ferment Transcript (full stack)
```

The NPU inference is the innermost layer. guideStone certification means
it is solid enough to build the rest on top of.

---

## Relationship to Other rustChip Documentation

| Document | Role relative to guideStone |
|----------|---------------------------|
| `SILICON_SPEC.md` | Source of Property 2 (reference-traceable hardware claims) |
| `PHASE_ROADMAP.md` | Context for Property 4 (VFIO removes platform dependencies) |
| `DRIVER_SPEC.md` | Backend architecture supporting Property 1 (determinism) |
| `baseCamp/ZOO_GUIDE.md` | User-facing guide that guideStone artifact validates |
| `baseCamp/SCIENTIFIC_DEPLOYMENT.md` | Deployment patterns enabled by guideStone certification |
| `baseCamp/GUIDESTONE_CERTIFICATION.md` | Living checklist tracking progress per property |

---

## References

- guideStone Standard: `wateringHole/fossilRecord/consolidated-apr2026/GUIDESTONE_STANDARD.md`
- guideStone Concept Paper: `whitePaper/gen4/architecture/GUIDESTONE.md`
- sporePrint section: `sporePrint/content/guidestone/_index.md`
- First deployment artifact: `hotSpring-guideStone-v0.7.0` (reference implementation)
- Glossary entry: `wateringHole/GLOSSARY.md` → guideStone
