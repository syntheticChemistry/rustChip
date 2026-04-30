# The Nature Preserve

**Where the zoo meets the field.**

The zoo contains curated models — trained, quantized, tested, shelf-stable.
The Nature Preserve is where those models are applied to real scientific domains.
Less complete than the zoo, but with more real subsystems: partial data,
domain-specific tolerances, extension points, and the messy interfaces between
neuromorphic compute and actual measurement.

---

## What this is

Each domain pattern below answers five questions:

1. **Problem:** What scientific question are you accelerating?
2. **Model:** Which zoo model (or architecture) fits, and why?
3. **Rust path:** How to wire rustChip into your domain pipeline.
4. **Output:** What you get — and what the numbers mean.
5. **Extension:** How to adapt this for your specific variant of the problem.

The patterns are grounded in work done across ecoPrimals springs. Where a
`validate_*` binary exists, it is cited. Where only the architecture exists,
the path from "architecture" to "running on silicon" is documented.

---

## The 7 Domains

| # | Domain | Problem class | Zoo model | Spring origin |
|---|--------|--------------|-----------|---------------|
| 1 | [**Physics**](physics.md) | Lattice QCD steering, transport coefficients, phase classification | ESN readout, phase classifier, transport predictor | hotSpring |
| 2 | [**Biology**](biology.md) | Quorum sensing classification, phylogenetic placement, bloom sentinel | ESN → int8 classifiers | wetSpring |
| 3 | [**Audio**](audio.md) | Keyword spotting, streaming speech, acoustic event detection | DS-CNN KWS, TENN Recurrent | Zoo (BrainChip) |
| 4 | [**Vision**](vision.md) | Object detection, segmentation, face analysis, plant disease | YOLO, UNet, AkidaNet PlantVillage, FaceID | Zoo (BrainChip) |
| 5 | [**Environmental**](environmental.md) | Algae bloom surveillance, ET₀ high-cadence, soil moisture | Streaming sensor, adaptive sentinel, bloom ESN | airSpring, wetSpring |
| 6 | [**Genomic**](genomic.md) | K-mer classification, genome binning, introgression detection | Multi-head readout, spectral triage | wetSpring, neuralSpring |
| 7 | [**Industrial**](industrial.md) | Predictive maintenance, sensor fusion, domain-shift detection | Streaming sensor 12ch, adaptive sentinel | Architecture patterns |

---

## How to read this

**If you have a specific domain:** Go directly to that page. Each is self-contained.

**If you want to understand the pattern:** Read [Physics](physics.md) first — it has
the most complete pipeline (5,978 live hardware calls, published results). Then
read any other domain that interests you.

**If you want to add your own domain:** See [Extension Guide](#adding-a-new-domain) below.

---

## The Pipeline Pattern

Every domain follows the same shape:

```
Domain data (measurements, signals, sequences)
    │
    ▼
Feature extraction (domain-specific, in Rust)
    │
    ▼
Quantization (f32/f64 → int4/int8, symmetric per-layer)
    │
    ▼
NPU inference (rustChip: parse model, load via VFIO, run)
    │
    ▼
Domain interpretation (classify, predict, detect, steer)
    │
    ▼
Output (decision, measurement, alert, next-step control)
```

The feature extraction and interpretation steps are domain-specific.
Everything between — quantization, model loading, inference execution — is
generic and handled by rustChip's crates.

---

## Adding a New Domain

To add domain #8:

1. **Identify the inference task.** What decision does the NPU make?
   Classification, regression, anomaly detection, steering?

2. **Choose or build a model.**
   - If a zoo model fits: use it directly (`Model::from_file()`).
   - If you need a new architecture: define layers with `ProgramBuilder`,
     convert via `akida convert`.
   - If you have trained weights: export to `.npy` or `.safetensors`,
     convert via the pure Rust pipeline.

3. **Write the feature extractor.** This is the domain-specific part.
   Transform your raw data into the tensor shape your model expects.

4. **Wire the pipeline.** Use `akida-models` for parsing, `akida-driver`
   for hardware execution, or the software backend for development.

5. **Document the pattern.** Create `preserve/your_domain.md` following the
   5-question structure (problem, model, Rust path, output, extension).

6. **Add a validation binary.** Name it `validate_preserve_<domain>.rs`
   in `crates/akida-bench/src/bin/` so the regression suite covers it.

---

## Connection to the Wider Ecosystem

The Nature Preserve is rustChip-native — all patterns work with just
`cargo build`. But each domain connects naturally to the wider ecoPrimals stack:

| Need | Primal | What it provides |
|------|--------|-----------------|
| GPU-accelerated feature extraction | [barraCuda](https://github.com/ecoPrimals/barraCuda) | 900+ WGSL shaders, DF64 emulation |
| Heterogeneous dispatch (GPU + NPU + CPU) | [toadStool](https://github.com/ecoPrimals/toadStool) | Tolerance-based routing, 21K+ tests |
| Sovereign GPU compilation | [coralReef](https://github.com/ecoPrimals/coralReef) | WGSL→native NVIDIA/AMD, no vendor SDK |
| Biology domain libraries | [wetSpring](https://github.com/syntheticChemistry/wetSpring) | FASTQ, 16S, phylogenetics, QS networks |
| Physics domain libraries | [hotSpring](https://github.com/syntheticChemistry/hotSpring) | Lattice QCD, MD, transport, ESN reservoirs |
| Environmental models | [airSpring](https://github.com/syntheticChemistry/airSpring) | ET₀, water balance, drought indices |
| Neural architecture validation | [neuralSpring](https://github.com/syntheticChemistry/neuralSpring) | LeNet/LSTM/transformer parity, dispatch |

rustChip is the NPU onboarding point. The Nature Preserve shows you what
to do with it.
