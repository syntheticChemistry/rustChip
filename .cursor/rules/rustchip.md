# rustChip Project Rules

## Identity

rustChip is an ecoPrimals infrastructure tool (gen2.5), not a primal or spring.
It lives at `infra/rustChip` in the ecoPrimals workspace. Canonical repo:
`syntheticChemistry/rustChip`. Ecosystem: `ecoPrimals` org, `primals.eco` site.

## Code Standards

- Edition 2024, `unsafe_code = "forbid"` at workspace level
- Driver crate uses `deny` with per-function SAFETY comments for VFIO/DMA
- `missing_docs = "deny"` — every public item gets a doc comment
- Clippy pedantic + nursery enabled; zero warnings required
- No TODO/FIXME in production code
- No magic numbers in tests — use named tolerances from `TOLERANCE_REGISTRY.md`

## File Conventions

- No production files > 800 lines
- SPDX license headers on source files: `// SPDX-License-Identifier: AGPL-3.0-or-later`
- Keep a Changelog format for `CHANGELOG.md`
- Spec files use `Date`, `Status`, `Authority` headers

## Architecture

- `akida-chip` = silicon model (no deps, no-std capable)
- `akida-driver` = hardware access (VFIO, software backend, hybrid ESN)
- `akida-models` = model handling (parse, quantize, import, serialize, zoo)
- `akida-bench` = experiments and benchmarks (binary crate)
- `akida-cli` = user-facing CLI (`akida` binary)

## Key Types

- `Model` = parsed .fbz model (from `akida-models`)
- `DeviceManager` = hardware discovery (from `akida-driver`)
- `ProgramBuilder` = programmatic model construction
- `ZooModel` = enum of all known models (28 variants)

## Testing

- Run `cargo test --workspace` before every commit
- Zoo regression: `cargo test --test zoo_regression -- --ignored`
- GuideStone: `cargo run -p akida-cli -- guidestone`
- CI runs: fmt, clippy, test, doc, deny

## Documentation Locations

- `specs/` = technical specifications (read before coding)
- `baseCamp/` = model zoo, domain applications, systems
- `baseCamp/preserve/` = Nature Preserve (7 domain patterns)
- `metalForge/` = hardware experiments
- `whitePaper/` = scientific write-ups
