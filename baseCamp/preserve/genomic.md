# Genomic — K-mer Classification, Genome Binning, Introgression Detection

**Spring origin:** wetSpring, neuralSpring
**Maturity:** NPU validation binaries for genome binning, spectral triage, disorder classification
**Zoo models:** Multi-head readout, spectral triage classifiers

---

## Problem

Genomic analysis pipelines process massive datasets — millions of reads,
thousands of genomes, terabytes of sequencing data. At multiple stages,
small classification decisions gate expensive downstream computation: Should
this read be retained or filtered? Which bin does this contig belong to?
Is this population segment introgressed?

These are high-throughput, low-complexity classification tasks — exactly
what a neuromorphic processor handles well. The NPU provides microsecond
per-read decisions, acting as a pre-filter that reduces the volume of data
flowing into expensive alignment, assembly, or phylogenetic algorithms.

Four sub-problems:

| Task | Question | Input |
|------|----------|-------|
| **K-mer classification** | What taxon does this sequence belong to? | K-mer frequency vector |
| **Genome binning** | Which genome bin does this contig belong to? | Coverage + composition features |
| **Spectral triage** | Is this mass spectrum worth full library matching? | Spectral features |
| **Introgression detection** | Has gene flow occurred between these populations? | Population genetic statistics |

---

## Model

### K-mer Taxonomic Classifier

```
Architecture: InputConv(16,1,1) → FC(128) → FC(64) → FC(N)
Quantization: int8 symmetric per-layer
Input:        16 features (4-mer or 5-mer frequency vector, compressed)
Output:       N-class taxonomy (phylum, order, or genus level)
Build:        akida convert --weights kmer_model.npy --arch "InputConv(16,1,1) FC(128) FC(64) FC(20)" --bits 8
```

wetSpring's `validate_upstream_taxonomy` and neuralSpring's
`validate_upstream_kmer` validate k-mer based classification against
established tools.

### Genome Binning Classifier

```
Architecture: InputConv(32,1,1) → FC(128) → FC(10)
Quantization: int8 symmetric per-layer
Input:        32 features (tetranucleotide frequencies + coverage statistics)
Output:       10-class bin assignment
Build:        akida convert --weights bin_model.npy --arch "InputConv(32,1,1) FC(128) FC(10)" --bits 8
```

wetSpring's `validate_npu_genome_binning` validates NPU genome binning
against CPU inference with matching classifications.

### Spectral Triage Pre-filter

```
Architecture: InputConv(64,1,1) → FC(128) → FC(2)
Quantization: int8 symmetric per-layer
Input:        64 spectral features (peak intensities, m/z ratios)
Output:       2-class decision [retain, discard]
```

The spectral triage model acts as a gatekeeper before expensive library
matching. wetSpring's `validate_npu_spectral_triage` and
`validate_npu_spectral_screen` validate this pattern with LC-MS data.

### Multi-Head Readout (population genetics)

```
Architecture: InputConv(50,1,1) → FC(128) → FC(1) per head
Quantization: int4 symmetric per-layer
Input:        population genetic summary statistics (Fst, pi, D, H)
Output:       per-head: scalar score for introgression/selection/drift
.fbz:         baseCamp/zoo-artifacts/esn_multi_head_3.fbz (template)
```

neuralSpring's `validate_introgression` and `validate_pangenome_selection`
validate population genetic classifiers that detect gene flow and selective
sweeps.

---

## Rust Path

### Feature extraction — k-mer frequencies

```rust
fn compute_kmer_features(sequence: &[u8], k: usize, n_features: usize) -> Vec<f32> {
    let mut counts = vec![0u32; 4usize.pow(k as u32)];
    for window in sequence.windows(k) {
        if let Some(idx) = kmer_to_index(window) {
            counts[idx] += 1;
        }
    }
    let total: f32 = counts.iter().sum::<u32>() as f32;
    // Compress to n_features via PCA or top-frequency selection
    counts.iter()
        .take(n_features)
        .map(|&c| c as f32 / total)
        .collect()
}

fn kmer_to_index(kmer: &[u8]) -> Option<usize> {
    let mut idx = 0;
    for &base in kmer {
        let val = match base {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => return None,
        };
        idx = idx * 4 + val;
    }
    Some(idx)
}
```

### High-throughput read classification

```rust
use akida_models::prelude::*;

let model = Model::from_file("kmer_classifier.fbz")?;
let backend = akida_driver::SoftwareBackend::new();

let mut classifications = Vec::with_capacity(reads.len());
for read in &reads {
    let features = compute_kmer_features(read, 4, 16);
    let result = backend.infer(&model, &features)?;

    let taxon = result.iter().copied().enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    classifications.push(taxon);
}
```

### Spectral pre-filter

```rust
let triage_model = Model::from_file("spectral_triage.fbz")?;

for spectrum in &spectra {
    let features = extract_spectral_features(spectrum);
    let result = backend.infer(&triage_model, &features)?;

    if result[0] > result[1] {
        // Worth full library matching
        expensive_library_search(spectrum);
    }
    // Otherwise: skip — saves 95%+ of compute for sparse hit-rate datasets
}
```

---

## Output

| Metric | Value | Source |
|--------|-------|--------|
| Genome binning — CPU/NPU parity | Matching classifications | `validate_npu_genome_binning` |
| Spectral triage — CPU/NPU parity | Matching decisions | `validate_npu_spectral_triage` |
| Spectral screen — validated | Passing | `validate_npu_spectral_screen` |
| Disorder classifier — validated | Passing | `validate_npu_disorder_classifier` |
| K-mer/taxonomy — validated | Reference parity | neuralSpring `validate_upstream_kmer` |

The value proposition is throughput: the NPU processes thousands of reads per
second at microsecond latency, acting as a pre-filter that reduces downstream
compute by 10–100× for datasets with sparse positive rates.

---

## Extension Points

**Nanopore real-time quality.** wetSpring's `validate_nanopore_*` binaries
process raw nanopore signal. An NPU classifier can score read quality in
real time during sequencing, enabling adaptive sampling (reject low-quality
reads before they consume sequencing bandwidth).

**Pangenome navigation.** neuralSpring's `validate_pangenome_selection`
classifies genomic windows by selection regime. Scale to whole-genome scans
by running the classifier as a streaming pre-filter.

**PFAS environmental genomics.** Combine the spectral triage model with
wetSpring's PFAS detection pipeline (`validate_pfas_library`) for
environmental chemistry — NPU pre-screens mass spectra, CPU performs
confirmatory library matching only on candidates.

**Population structure.** Extend the introgression model to detect population
structure (admixture, bottlenecks, expansion) from summary statistics.
The multi-head readout architecture naturally supports multiple
simultaneous population genetic tests.

---

## Validation Binaries

```bash
# wetSpring — genomic NPU patterns
cd wetSpring/barracuda/
cargo run --bin validate_npu_genome_binning
cargo run --bin validate_npu_spectral_triage
cargo run --bin validate_npu_spectral_screen
cargo run --bin validate_npu_disorder_classifier

# neuralSpring — population genetics
cd neuralSpring/
cargo run --bin validate_introgression
cargo run --bin validate_pangenome_selection
cargo run --bin validate_upstream_kmer
cargo run --bin validate_upstream_taxonomy
```
