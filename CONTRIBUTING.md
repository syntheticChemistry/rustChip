# Contributing to rustChip

## Quality Commands

Run these before every push:

```bash
# Format — must be clean
cargo fmt --check

# Lint — zero warnings (pedantic + nursery enabled)
cargo clippy --workspace -- -D warnings

# Test — all must pass
cargo test --workspace

# Docs — must build without warnings
cargo doc --workspace --no-deps

# Dependency audit
cargo deny check
```

## Repository Layout

```
crates/
  akida-chip/      Silicon model (register map, NP mesh, BAR layout, SRAM)
  akida-driver/    VFIO driver, software backend, hybrid ESN
  akida-models/    FBZ parser, model zoo, quantization, conversion pipeline
  akida-bench/     Benchmark and experiment binaries
  akida-cli/       `akida` command-line tool
specs/             Technical specifications — read before coding
baseCamp/          Model zoo, domain applications, systems documentation
metalForge/        Hardware experiments and silicon characterization
whitePaper/        Scientific write-ups and outreach
```

## Code Standards

- **Edition 2024** — all crates use Rust 2024 edition.
- **`unsafe_code = "forbid"`** at workspace level. The driver crate uses
  targeted `deny` with SAFETY comments for VFIO/DMA.
- **`missing_docs = "deny"`** — all public items require doc comments.
- **Clippy pedantic + nursery** — both enabled at workspace level.
- **No TODO/FIXME in production code** — use issue tracker or specs.
- **Named constants** — no magic numbers in test assertions. Tolerances
  go in `TOLERANCE_REGISTRY.md` and are referenced by name.

## Adding a New Feature

1. Check `specs/` for relevant specifications.
2. Write the implementation with doc comments on all public items.
3. Add tests — both unit tests and integration tests where appropriate.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Update relevant baseCamp or specs documentation.
6. Run the full quality command suite above.

## Adding a New Model to the Zoo

1. Generate or export the `.fbz` file to `baseCamp/zoo-artifacts/`.
2. Add a variant to the `ZooModel` enum in `crates/akida-models/src/zoo.rs`.
3. Implement all required methods — the compiler will guide you.
4. Update `baseCamp/ZOO_GUIDE.md`.
5. Run `cargo test --test zoo_regression -- --ignored` to verify.

## Commit Messages

Use imperative mood. Reference the affected crate or area:

```
akida-models: add safetensors weight import with f16/bf16 upcast
baseCamp: document streaming sensor domain pattern
ci: add cargo-deny to workflow
```

## License

All contributions are licensed under the **scyBorg Provenance Trio**:
AGPL-3.0-or-later (code), ORC (mechanics), CC-BY-SA 4.0 (creative).
See [LICENSE](LICENSE) and [LICENSE-ORC](LICENSE-ORC).

### Lineage principle for absorbed code

rustChip absorbs functionality from the wider ecoPrimals ecosystem
(coralReef, toadStool, etc.) to operate as a standalone onboarding
tool. **Code with ecoPrimals lineage carries the full scyBorg triple
by inheritance.** Relocation does not strip provenance — these systems
are inheritors of the ecoPrimals commons and are not eligible for any
symbiotic exception.

When absorbing code from a primal or spring into rustChip:

1. Preserve provenance comments identifying the origin
   (e.g. "Absorbed from coral-ember/src/sysfs.rs")
2. The absorbed code carries full scyBorg — it does not become
   exception-eligible simply by living in rustChip
3. New code built on top of absorbed patterns inherits the lineage
4. Document the absorption in the LICENSE tail's lineage section

The BrainChip symbiotic exception (and any future NPU vendor exception)
covers only rustChip-original code with no ecoPrimals ancestry.
