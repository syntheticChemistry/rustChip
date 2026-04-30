# guideStone Certification — rustChip

**Artifact:** `rustChip-guideStone-v0.1.0`
**Date:** April 29, 2026
**Standard:** guideStone Standard v1.0 (WateringHole Consensus, March 31, 2026)
**Specification:** `specs/GUIDESTONE.md`
**Binary:** `akida guidestone [dir]` — runs 225 checks across 25 models in ~3 sec

---

## Certification Checklist

### Property 1: Deterministic Output

Same input, same binary, any hardware → same output within named tolerances.

- [x] FBZ parser produces identical `Model` struct for same `.fbz` input
- [x] Snappy decompression is pure Rust, deterministic (`snap` crate)
- [x] AKD1000 hardware is deterministic (integer arithmetic, no FP variance)
- [x] `SyntheticNpuBackend` provides deterministic software fallback
- [ ] Cross-architecture validation: x86_64 vs aarch64 parse output comparison
- [ ] Formal determinism test covering all 25 models × 2 architectures
- [ ] No environment-dependent behavior (locale, timezone, hostname) — audit needed

**Status: SUBSTANTIALLY MET** — determinism is inherent to the compute
model (integer NPU, pure Rust parser). Cross-architecture evidence is
the remaining gap.

---

### Property 2: Reference-Traceable

Every numeric claim traces to a verifiable source.

- [x] 21 BrainChip models traced to Akida SDK v2.19.1 (`zoo_manifest.json`)
- [x] 4 physics models traced to documented architectures (`baseCamp/models/physics/`)
- [x] Hardware measurements traced to direct probing (`SILICON_SPEC.md`)
- [x] Phase classification accuracy traces to Bazavov et al., PRD 85, 054503
- [x] Anderson localization traces to groundSpring Exp 028
- [x] Throughput/latency claims traced to Exp 022 measurements
- [ ] SDK commit hash in `zoo_manifest.json` (version present, hash missing)
- [ ] DOI/arXiv links as machine-readable fields in physics model metadata
- [ ] NP budget estimates for non-physics zoo models: measured vs approximated

**Source registry:**

| Claim domain | Reference type | Source |
|-------------|---------------|--------|
| Parser correctness | Validation oracle | BrainChip Akida SDK 2.19.1 |
| Register map | Direct measurement | BAR0 MMIO reads (`SILICON_SPEC.md`) |
| NP count (1,000) | Hardware read | `Capabilities::from_bar0()` at 0x10C0 |
| β_c = 5.69 | Published paper | Bazavov et al., PRD 85, 054503 (2012) |
| W_c = 16.26 ± 0.95 | Experiment | groundSpring Exp 028 |
| 18,500 Hz throughput | Measurement | hotSpring Exp 022, Phase C driver |
| 54 µs latency | Measurement | Phase C + D direct timing |
| 1.4 µJ/inference | Datasheet + measurement | BrainChip + Exp 022 |
| 37 MB/s DMA | Measurement | Phase C (`SILICON_SPEC.md`) |

**Status: SUBSTANTIALLY MET** — all significant claims are traced. Machine-readable
metadata format is the remaining gap.

---

### Property 3: Self-Verifying

The artifact carries its own integrity mechanism.

- [ ] CHECKSUMS (SHA-256) file in `baseCamp/zoo-artifacts/`
- [ ] Per-model SHA-256 in `zoo_manifest.json`
- [ ] Per-model SHA-256 in `physics_manifest.json`
- [x] Snappy decompression detects corruption (implicit CRC in compressed stream)
- [x] FlatBuffer parsing rejects malformed/truncated structures
- [x] `zoo_regression.rs` validates all 25 models parse correctly
- [x] Decompression ratio sanity check in regression tests
- [x] `akida guidestone` CLI command — 225 checks, SHA-256, structure, throughput
- [ ] Tampered input detection and reporting (not just parse failure)
- [ ] Signed release artifacts (requires bearDog — future phase)

**Status: SUBSTANTIALLY MET** — `akida guidestone` computes SHA-256 per model,
validates structure, benchmarks throughput, and self-grades. CHECKSUMS file
generation (for offline artifact packaging) is the remaining gap.

---

### Property 4: Environment-Agnostic

No hardcoded paths. No "install X first." No platform assumptions.

- [x] Pure Rust — all crates, zero C dependencies
- [x] No Python at runtime (Python is one-time validation oracle only)
- [x] No sudo after initial udev setup
- [x] No hardcoded PCIe addresses or device paths
- [x] Runtime hardware discovery via sysfs and BAR0 reads
- [x] CPU-only path via `SyntheticNpuBackend`
- [x] CLI reports detected substrate (`akida verify`, `akida enumerate`)
- [ ] Static musl build (`x86_64-unknown-linux-musl`)
- [ ] aarch64 cross-compilation (`aarch64-unknown-linux-musl`)
- [ ] Binary reports detected substrate in machine-readable format

**Status: SUBSTANTIALLY MET** — the code is portable and self-discovering.
Static binary packaging is the remaining work.

---

### Property 5: Tolerance-Documented

Every threshold has a derivation. No magic numbers.

- [x] Int4 weight range [-8, 7]: hardware specification (4-bit signed)
- [x] Int4 activation range [0, 15]: hardware specification (4-bit unsigned)
- [x] 51-bit threshold SRAM: direct BAR0 register read
- [x] β_c = 5.69 ± 0.01: finite-size scaling (Bazavov et al.)
- [x] W_c = 16.26 ± 0.95: finite-size scaling (groundSpring Exp 028)
- [ ] Parse throughput threshold derivation (minimum acceptable for real-time loading)
- [ ] Decompression ratio bound derivation (Snappy characteristics for weight matrices)
- [ ] Transport prediction error formal bounds (stochastic training noise)
- [ ] Tolerance metadata in machine-readable output JSON (not just docs)

**Tolerance registry:**

| Threshold | Value | Derivation | Status |
|-----------|-------|-----------|--------|
| Weight quantization range | [-8, 7] | AKD1000 4-bit signed integer | Derived |
| Activation quantization range | [0, 15] | AKD1000 4-bit unsigned integer | Derived |
| SRAM threshold width | 51 bits | BAR0 register 0x1410 probe | Measured |
| Phase boundary β_c | 5.69 ± 0.01 | Finite-size scaling, Bazavov et al. | Published |
| Anderson W_c | 16.26 ± 0.95 | Finite-size scaling, Exp 028 | Experimental |
| Phase classification | 100% | Full binary test set (no stochastic element) | Complete |
| Parse throughput minimum | 1 MB/s | *Needs derivation* | Empirical |
| Decompression ratio | < 5:1 typical | *Needs derivation* | Empirical |
| Transport D* error | 3.1% | *Needs formal bounds* | Observed |
| Transport η* error | 3.8% | *Needs formal bounds* | Observed |
| Transport λ* error | 4.2% | *Needs formal bounds* | Observed |

**Status: PARTIALLY MET** — hardware and physics tolerances are well-derived.
Software performance thresholds need formal derivations.

---

## Summary

```
Property 1: Deterministic Output      [SUBSTANTIALLY MET]  needs cross-arch proof
Property 2: Reference-Traceable       [SUBSTANTIALLY MET]  needs machine-readable metadata
Property 3: Self-Verifying            [SUBSTANTIALLY MET]  akida guidestone runs; CHECKSUMS file remains
Property 4: Environment-Agnostic      [SUBSTANTIALLY MET]  needs musl + aarch64 builds
Property 5: Tolerance-Documented      [PARTIALLY MET]      needs software threshold derivations
```

**Certification target:** `rustChip-guideStone-v0.1.0`
**Current state:** `akida guidestone` runs 225 checks on 25 models in ~3 sec, all pass
**Blocking items:** CHECKSUMS file generation, cross-arch build, tolerance derivations
**Estimated effort:** 1–2 focused sessions

---

## Certification Path (Phases G1–G5)

| Phase | Work | Properties served |
|-------|------|-------------------|
| G1 | CHECKSUMS + manifest hashes + `verify-artifact` CLI | Property 3 |
| G2 | musl builds + aarch64 cross-compile + determinism proof | Properties 1, 4 |
| G3 | Machine-readable provenance in JSON output | Properties 2, 5 |
| G4 | Artifact directory packaging + `./run validate` | All 5 |
| G5 | Release naming (`rustChip-guideStone-v0.1.0`) | Naming convention |

Detailed phase descriptions: `specs/GUIDESTONE.md`

---

## What the First Artifact Validates

When `rustChip-guideStone-v0.1.0` ships, running `./run validate` will:

1. Verify CHECKSUMS of all 25 `.fbz` models
2. Parse each model and compare output to `expected/parse_results.json`
3. Verify layer counts, decompression ratios, version strings
4. Report: `25/25 models validated. All tolerances within derived bounds.`

On hardware (with AKD1000 present):

5. Load a reference model via VFIO
6. Run inference with known input
7. Compare output to expected (integer-exact for NPU)
8. Report hardware throughput and latency measurements

The artifact is the conversation starter. The physics speaks for itself.
