# Biology — Quorum Sensing, Phylogenetics, Bloom Surveillance

**Spring origin:** wetSpring
**Maturity:** Validated — ESN→int8 classifiers tested across multiple biological domains
**Zoo models:** ESN classifiers (custom), adaptive sentinel

---

## Problem

Biological systems generate continuous streams of measurements — sequencing
reads, metabolite concentrations, community composition indices. Decisions
must be made in real time: Is this microbial community in a quorum sensing
phase transition? Does this phylogenetic placement indicate a novel lineage?
Is an algae bloom forming?

These classification tasks share a structure: feature vectors from biological
pipelines feed into small, fast neural classifiers. The NPU handles the
inference step at microsecond latency, freeing the CPU for the
compute-intensive upstream pipeline (alignment, assembly, ODE integration).

Three sub-problems:

| Task | Question | Input source |
|------|----------|-------------|
| **QS phase classification** | Is this community in quorum-sensing phase transition? | ODE/Gillespie simulation features |
| **Phylogenetic placement** | Where does this sequence belong in the reference tree? | Alignment/k-mer features |
| **Bloom sentinel** | Is a harmful algae bloom forming? | Temporal sensor stream |

---

## Model

### QS Phase Classifier

```
Architecture: InputConv(8,1,1) → FC(64) → FC(3)
Quantization: int8 symmetric per-layer
Input:        8 features (signal molecule concentrations, growth rate, community size)
Output:       3-class probability [pre-QS, transition, post-QS]
Build:        akida convert --weights qs_features.npy --arch "InputConv(8,1,1) FC(64) FC(3)" --bits 8
```

wetSpring's `validate_npu_qs_classifier` trains the classifier from Gillespie
stochastic simulation output and validates against CPU inference.

### Phylogenetic Placement Classifier

```
Architecture: InputConv(16,1,1) → FC(128) → FC(5)
Quantization: int8 symmetric per-layer
Input:        16 features (k-mer frequencies, alignment scores, tree distances)
Output:       5-class clade assignment
Build:        akida convert --weights phylo_features.npy --arch "InputConv(16,1,1) FC(128) FC(5)" --bits 8
```

### Bloom Sentinel (temporal ESN)

```
Architecture: InputConv(1,1,1) → FC(256) → FC(128) → FC(12)
Quantization: int8 symmetric per-layer
Input:        streaming sensor data (chlorophyll-a, turbidity, temperature, pH, ...)
Output:       12-channel classification (bloom species + severity + trend)
.fbz:         baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz (Rust-native template)
```

wetSpring's `validate_temporal_esn_bloom` and `validate_npu_bloom_sentinel`
validate this pattern with real and simulated bloom data.

---

## Rust Path

### Feature extraction — QS phase

```rust
fn extract_qs_features(
    signal_molecules: &[f64],
    growth_rate: f64,
    community_size: f64,
) -> Vec<f32> {
    let mut features = Vec::with_capacity(8);
    for &mol in signal_molecules.iter().take(6) {
        features.push(mol as f32);
    }
    features.push(growth_rate as f32);
    features.push(community_size as f32);
    features
}
```

### Feature extraction — bloom sentinel

```rust
fn extract_bloom_features(
    sensor_window: &[(f64, f64, f64, f64)], // (chlorophyll, turbidity, temp, pH)
) -> Vec<f32> {
    // Flatten the most recent reading into model input
    let latest = sensor_window.last().unwrap();
    vec![
        latest.0 as f32,  // chlorophyll-a
        latest.1 as f32,  // turbidity
        latest.2 as f32,  // temperature
        latest.3 as f32,  // pH
    ]
}
```

### NPU inference and classification

```rust
use akida_models::prelude::*;

let model = Model::from_file("baseCamp/zoo-artifacts/streaming_sensor_12ch.fbz")?;
let backend = akida_driver::SoftwareBackend::new();

// Continuous monitoring loop
loop {
    let features = extract_bloom_features(&sensor_buffer);
    let result = backend.infer(&model, &features)?;

    let severity = result.iter().copied().enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    if severity > BLOOM_ALERT_THRESHOLD {
        trigger_alert(severity, &result);
    }

    std::thread::sleep(std::time::Duration::from_secs(SAMPLE_INTERVAL));
}
```

---

## Output

| Metric | Value | Source |
|--------|-------|--------|
| QS classifier — int8 round-trip tolerance | < 0.5% relative error | wetSpring tolerances |
| Phylo placement — CPU vs NPU parity | Matching classifications | `validate_npu_phylo_placement` |
| Bloom sentinel — streaming latency | Sub-millisecond per inference | Architecture estimate |
| Genome binning — NPU spectral triage | Validated | `validate_npu_spectral_triage` |

The biological classifiers are small (typically 8–64 input features, 2–12
output classes). This means they fit in a fraction of the AKD1000's NP budget,
leaving room for multi-tenancy — run QS classifier, bloom sentinel, and
spectral triage simultaneously on one chip.

---

## Extension Points

**Nanopore real-time basecalling.** wetSpring has `validate_nanopore_*`
binaries for field genomics. The raw signal → feature → classification
pattern maps directly to NPU inference for real-time base quality scoring.

**Soil microbiome QS geometry.** wetSpring's `validate_soil_qs_*` binaries
extend QS classification from laboratory cultures to soil community dynamics,
where spatial structure matters.

**PFAS spectral screening.** The LC-MS spectral matching pipeline
(`validate_pfas_library`, `validate_npu_spectral_screen`) uses NPU
classification to pre-screen mass spectra before expensive library matching.

**Multi-signal cooperation.** The `validate_cooperation` and
`validate_multi_signal` binaries model multi-species quorum sensing with
game-theoretic interactions. The NPU classifies the cooperation regime
(defection, partial cooperation, full cooperation) from observed signal
ratios.

---

## Validation Binaries (wetSpring)

```bash
# In wetSpring/barracuda/ — software mode
cargo run --bin validate_npu_qs_classifier
cargo run --bin validate_npu_phylo_placement
cargo run --bin validate_npu_bloom_sentinel
cargo run --bin validate_npu_sentinel_stream
cargo run --bin validate_npu_spectral_screen
cargo run --bin validate_npu_spectral_triage
cargo run --bin validate_npu_genome_binning
cargo run --bin validate_npu_disorder_classifier

# Hardware mode (requires AKD1000 + VFIO)
cargo run --bin validate_npu_hardware --features npu
cargo run --bin validate_npu_live --features npu
```
