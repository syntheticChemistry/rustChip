# rustChip development commands

# Run all quality checks (pre-push)
check: fmt clippy test doc deny

# Format check
fmt:
    cargo fmt --check

# Lint (pedantic + nursery, zero warnings)
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run all tests
test:
    cargo test --workspace

# Run tests including ignored (zoo regression, requires artifacts)
test-all:
    cargo test --workspace -- --include-ignored

# Build documentation
doc:
    cargo doc --workspace --no-deps

# Dependency audit
deny:
    cargo deny check

# Build release
build:
    cargo build --workspace --release

# Parse a model
parse file:
    cargo run -p akida-cli -- parse {{file}}

# Convert weights to .fbz
convert weights arch output bits="4":
    cargo run -p akida-cli -- convert --weights "{{weights}}" --arch "{{arch}}" --output "{{output}}" --bits {{bits}}

# Show zoo status
zoo-status:
    cargo run -p akida-cli -- zoo-status

# Run guideStone validation
guidestone:
    cargo run -p akida-cli -- guidestone

# Run benchmarks (hardware required)
bench:
    cargo run --bin validate_all -- --sw

# Count tests
count-tests:
    @echo "Total #[test] functions:"
    @rg '#\[test\]' crates/ --type rust -c | awk -F: '{s+=$2} END {print s}'
