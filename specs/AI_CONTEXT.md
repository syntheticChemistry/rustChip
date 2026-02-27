# AI Developer Context

This file is the entry point for AI coding assistants and new developers.
Read this before touching any code.

---

## What this project is

`rustChip` is a **standalone, pure Rust driver and benchmark suite** for
BrainChip Akida neuromorphic processors (AKD1000, AKD1500). It has
**no runtime dependencies** on any other ecoPrimals project (toadStool,
hotSpring, wetSpring). When you clone this repository and run `cargo build`,
you get a fully functional system.

This is intentional. The project is designed to be handed to BrainChip's
engineering team as a complete, self-contained artifact.

---

## Crate graph

```
akida-chip                  ← no deps (pure silicon model)
    ↑
akida-driver                ← depends on akida-chip + rustix + libc (VFIO ioctls)
    ↑               ↑
akida-models      akida-bench     akida-cli
(FlatBuffer)      (10 benchmarks) (CLI tool)
```

Do not create circular dependencies. `akida-chip` must remain zero-dependency.

---

## Naming conventions

| Convention | Rule |
|------------|------|
| Crate names | `akida-{noun}` — kebab-case |
| Module names | `snake_case` |
| Hardware constants | `SCREAMING_SNAKE_CASE` |
| `confirmed` labels | Any constant measured directly from hardware |
| `inferred` labels | Consistent with behavior, not directly read |
| `hypothetical` labels | Assumed from spec/databook, unverified |

Do not remove or downgrade `confirmed` / `inferred` / `hypothetical` labels.
They are the provenance record for the register map.

---

## Hardware discovery rules

**Never hardcode a device path.** All of these are wrong:

```rust
// WRONG
let dev = File::open("/dev/akida0")?;
let addr = "0000:a1:00.0";
let group = 5u32;
```

Always use `DeviceManager::discover()` which scans sysfs at runtime:

```rust
// CORRECT
let mgr = DeviceManager::discover()?;
let dev = mgr.open_first()?;
```

The only acceptable hardcoded values are PCIe vendor/device IDs in
`akida-chip/src/pcie.rs` — those are silicon constants, not configuration.

---

## Safety rules

`akida-chip` has `#![forbid(unsafe_code)]`. Keep it that way.

`akida-driver` has unsafe code in exactly one place: `src/vfio/mod.rs`.
Every unsafe block must have:
1. A comment explaining **why** unsafe is necessary (what kernel API requires it)
2. **Invariants** the code maintains
3. **Caller guarantees** needed

Do not add unsafe code outside `vfio/mod.rs` without a documented reason.

---

## Error handling

All public functions return `Result<T, AkidaError>`. Never use `.unwrap()`
or `.expect()` in library code. In binaries (bench/cli), `?` with `anyhow`
is acceptable.

When adding a new error case, add a variant to `AkidaError` in
`src/error.rs` — don't use `anyhow::Error` in the library crate.

---

## Testing philosophy

Hardware may not be present. Tests that require hardware must:
1. Call `DeviceManager::discover()` at the start
2. Skip gracefully if zero devices found:

```rust
#[test]
fn test_needs_hardware() {
    let mgr = DeviceManager::discover().expect("discover should not fail");
    if mgr.device_count() == 0 {
        eprintln!("No Akida hardware — skipping");
        return;
    }
    // ... hardware test ...
}
```

Unit tests in `akida-chip` require no hardware (pure constants and math).

---

## Directory structure (beyond crates/)

```
specs/          Technical spec — read before coding
baseCamp/       Models, novel systems, extended capabilities
  systems/      Multi-system architectures and novel NPU applications
  models/       Per-model docs (physics, edge, custom, beyond_sdk)
  conversion/   Getting external models into rustChip format
  zoos/         Third-party zoo landscape survey
metalForge/     Hardware experiment protocols and results
  experiments/  Numbered experiment files (002, 003, 004…)
whitePaper/     Analysis and outreach
  explorations/ Deep-dive technical writeups
  outreach/akida/ Material for BrainChip engineering team
```

These directories are documentation-first: they contain `.md` files describing
the architecture, experiments, and findings. The code that validates them lives
in `akida-bench/src/bin/`.

---

## Benchmark philosophy

**Two types of bench binaries:**

1. **BEYOND_SDK reproduction** (`bench_channels`, `bench_dma`, etc.):
   Each reproduces exactly one discovery. File header states:
   - Which discovery it reproduces (e.g. "Discovery 4 from BEYOND_SDK.md")
   - The reference measurement
   - What SDK claim is being overturned
   Reference measurements are **not** test assertions — hardware varies.

2. **metalForge experiments** (`bench_exp002_tenancy`, `bench_exp004_hybrid_tanh`, `run_experiments`):
   Each implements a metalForge experiment protocol. Two phases:
   - **Phase 1** (software simulation): validates the math and architecture model.
     Must work without hardware (`/dev/akida0` absent). All Phase 1 must pass CI.
   - **Phase 2** (hardware dispatch): replaces the SW emulation with actual device calls.
     Gated behind hardware presence check. Activated by `--hw` flag.

   Pattern:
   ```rust
   fn main() {
       let hw = std::path::Path::new("/dev/akida0").exists();
       // Phase 1: always runs
       let (pass, results) = run_phase1(&mut rng);
       // Phase 2: only when hardware present
       if hw { let (pass2, results2) = run_phase2_hardware(); }
   }
   ```

3. **Unified runner** (`run_experiments`): Runs all pending experiments.
   Use this to quickly assess project state. Should always exit 0 on CI.

---

## Documentation philosophy

Every public item needs a doc comment. For hardware-facing constants:

```rust
/// Optimal batch size for PCIe amortisation (Discovery 3).
///
/// `batch=8` gives 2.4× throughput over `batch=1` by spreading the
/// ~650 µs PCIe round-trip cost across 8 inference samples.
/// Source: BEYOND_SDK.md Discovery 3, Feb 2026.
pub const OPTIMAL_BATCH_SIZE: usize = 8;
```

For implementation notes that explain hardware behavior (not just the API):
use `//!` module-level docs or inline `//` comments. Do not write comments
that just describe the code — only document **why**, not **what**.

---

## HybridEsn / substrate pattern

The `HybridEsn` in `crates/akida-driver/src/hybrid.rs` is the pattern for
substrate-agnostic inference. hotSpring and toadStool program against the
`EsnSubstrate` trait, not against a specific backend:

```rust
// The trait — what hotSpring uses
pub trait EsnSubstrate: Send + Sync {
    fn step(&mut self, input: &[f32]) -> Result<Vec<f32>>;
    fn reset(&mut self);
    fn substrate_mode(&self) -> SubstrateMode;
}

// Construction: from tanh-trained weights (no retraining needed for hardware)
let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3)?;
// PureSoftware mode by default — correct tanh behavior, 800 Hz
let out = esn.step(&input)?;

// Upgrade to hardware when available (same weights, same accuracy, 18,500 Hz):
// Pending metalForge/experiments/004_HYBRID_TANH Phase 2
let esn = esn.with_hardware_linear(device)?;
```

**Key finding**: bounded ReLU clips negative pre-activations to 0, destroying
half the signal. This makes ESNs with random weights degenerate. `HybridEsn`
routes the matrix multiply to hardware and applies tanh on the host, bypassing
the activation constraint entirely. See `whitePaper/explorations/TANH_CONSTRAINT.md`.

**Do not** hardcode tanh or bounded ReLU in physics simulation code. Use `EsnSubstrate`.

---

## Extension patterns

### Adding a new backend

1. Create `src/backends/{name}.rs`
2. Implement `NpuBackend` trait from `src/backend.rs`
3. Add variant to `BackendType` enum
4. Add `BackendSelection::{Name}` variant
5. Add arm to `select_backend()` in `src/backend.rs`
6. Export from `src/backends/mod.rs`

### Adding a new benchmark

1. Create `crates/akida-bench/src/bin/{name}.rs`
2. Add `[[bin]]` entry to `crates/akida-bench/Cargo.toml`
3. File header must cite the BEYOND_SDK.md discovery it reproduces
4. Print reference measurement, measured value, and comparison

### Adding a new hardware constant

1. Determine if it belongs in `akida-chip` (silicon model) or `akida-driver` (driver behavior)
2. Add to appropriate module with `confirmed`/`inferred`/`hypothetical` label
3. Add test in `#[cfg(test)]` block validating the constant value

### Supporting AKD1500

The only required changes:
1. `pcie.rs`: `AKD1500 = 0xA500` is already there
2. `bar.rs`: Verify BAR sizes (PCIe x2 may change layout)
3. `mesh.rs`: AKD1500 has different NP count/topology
4. `capabilities.rs`: Handle additional GPIO/SPI capabilities in sysfs

---

## What NOT to do

- Do not add a dependency on `toadstool`, `barracuda`, `hotspring`, or any
  other ecoPrimals project. This repo must be standalone.
- Do not add Python bindings. If a Python consumer wants to use this,
  they can call `akida-cli` as a subprocess.
- Do not implement model training or weight optimization. This is an
  inference driver. Training belongs in the scientific computing projects.
- Do not assume the C kernel module is present. All code paths must work
  without it (using VFIO backend as fallback).
- Do not add tokio as a required (non-feature) dependency. The `async` feature
  gate exists for a reason.
- Do not hardcode NP addresses. Always use the cumulative address map from
  `baseCamp/systems/README.md`. The correct addresses are:
  `0x0000, 0x00B3, 0x0139, 0x0215, 0x0275, 0x02B8, 0x02FC` (7-system packing).
- Do not use non-ASCII Unicode variable names (e.g. `ε`, `α`) in Rust source.
  Use ASCII equivalents (`eps`, `alpha`). rustc warns on mixed-script confusables.
- Do not let bench Phase 1 (software simulation) binaries exit non-zero.
  They must pass without hardware. Phase 2 failures are expected without hardware.
