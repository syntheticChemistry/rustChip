# Third-Party Frameworks and Emerging Zoos

**Status:** Reviewed February 2026. Landscape survey for awareness and future integration.

---

## Open Neuromorphic (community hub)

**URL:** https://open-neuromorphic.org
**Purpose:** Community aggregation of open neuromorphic frameworks and models

Maintains a curated list of frameworks compatible with various neuromorphic
hardware targets. As of early 2026, lists 30+ frameworks. Most relevant to Akida:

| Framework | Compatibility | Notes |
|-----------|--------------|-------|
| SNNTorch | ✅ via quantization | Most mature, best documented |
| Sinabs | ✅ via PyTorch weights | Synaptic Intelligence, audio focus |
| Rockpool | ⚠️ selective | SpiNNaker/Xylo primary targets |
| Tonic | ✅ datasets only | Event-based data loaders |

---

## Sinabs (SynapseAI)

**URL:** https://synapseinterface.gitlab.io/sinabs/
**Language:** Python / PyTorch
**Neurons:** LIF (simplified for hardware deployment)
**Hardware targets:** SpiNNaker 2, Xylo, Akida (via quantization)
**License:** LGPL 2.1

Sinabs is designed with deployment in mind — its LIF implementation is
intentionally simplified to match the constraints of physical neuromorphic chips.
This makes it a better Akida target than Norse or BindsNET.

The Sinabs → Akida path:
1. Design network in Sinabs (explicit hardware constraints from day 1)
2. Export weights
3. Quantize with QuantizeML or `akida-models::quantize` (when implemented)
4. Build .fbz via CNN2SNN or `program_external()`

---

## Tonic (event-based datasets)

**URL:** https://tonic.readthedocs.io
**Purpose:** Data loading for event-based (DVS camera) datasets
**Key datasets:**
- DVS128 Gesture (11 classes, 1,342 samples)
- N-MNIST (digit recognition, event-based)
- N-CALTECH101 (object categories, event-based)
- NCARS (cars in urban scenes)
- POKERDVH (poker card symbols)
- DVSLip (lip reading)

These datasets are used by NeuroBench's gesture benchmark. rustChip could
consume event streams directly from Tonic's format (CSV or binary) as input
to the DVS gesture model.

**Rust integration:** No Rust bindings for Tonic. Would need a small data
loader written in `akida-models` or `akida-bench`:

```rust
// Planned: akida_models::datasets::DvsGestureLoader
pub struct DvsGestureLoader {
    path: PathBuf,
}

impl DvsGestureLoader {
    pub fn load_sample(&self, idx: usize) -> Result<Vec<DvsEvent>> { ... }
    pub fn to_frame(&self, events: &[DvsEvent], t_window_ms: f32) -> Array3<u8> { ... }
}
```

---

## Hugging Face Model Hub (neuromorphic section)

**URL:** https://huggingface.co/models?search=snn
**Status:** Small but growing (20–30 models as of early 2026)

Some pre-quantized Akida models appear on Hugging Face:
- `brainchip/akidanet-imagenet-160` (AkidaNet 0.5 in .fbz)
- `brainchip/ds-cnn-kws` (keyword spotting .fbz)

These can be downloaded and loaded directly via `akida-models` without
the Python MetaTF SDK. Worth monitoring as the number grows.

---

## Emerging: Rust-Native SNN Frameworks

No mature Rust SNN training framework exists as of early 2026. The gap:

| Need | Current state | rustChip contribution |
|------|--------------|----------------------|
| SNN inference on Akida | ✅ `akida-driver` | **Done** |
| Model loading from .fbz | ✅ `akida-models` | **Done** |
| Weight quantization | ❌ Rust crate | Queued: `akida-models::quantize` |
| Architecture → .fbz compilation | ❌ Rust crate | Queued: `akida-models::builder` |
| STDP training in Rust | ❌ Rust crate | Long-term (Phase F on-chip path) |

The `akida-models::builder` module (planned) would allow writing Akida-ready
architectures entirely in Rust, eliminating the last Python dependency:

```rust
// Planned API
use akida_models::builder::ProgramBuilder;

let program = ProgramBuilder::new()
    .input_conv(channels_in: 1, channels_out: 128, kernel: 3)
    .fully_connected(out: 2)
    .with_weights_int4(&weights_quantized)
    .compile()?;

// program is a CompiledProgram with .fbz-compatible binary layout
device.program_external(&program.program_info, &program.program_data)?;
```

This would be a significant contribution to the neuromorphic Rust ecosystem —
the first fully sovereign Akida model builder without any Python toolchain.

---

## Benchmarking Zoos: MLPerf Tiny

**URL:** https://mlcommons.org/benchmarks/inference-tiny
**Relevance:** Defines TinyML benchmarks for microcontrollers and edge devices

MLPerf Tiny benchmarks overlap with NeuroBench for audio/vision tasks:
- Keyword spotting (same DS-CNN target)
- Visual wakeword detection
- Image classification (person vs not-person)
- Anomaly detection (ToyADMOS dataset)

Akida is not an official MLPerf Tiny target (too powerful for µC class),
but the model architectures map cleanly. The DS-CNN used in MLPerf Tiny
KWS is the same as in NeuroBench.

---

## Priority for rustChip

Ordered by: (relevance to ecoPrimals workloads) × (conversion complexity⁻¹)

1. **Pre-compiled .fbz from BrainChip/Hugging Face** — load today, zero effort
2. **SNNTorch FC networks** — extract weights, quantize, `program_external()` — Phase 0.2
3. **Tonic datasets** — pure data loading, no conversion — Phase 0.2
4. **Sinabs architectures** — hardware-friendly, good conversion path — Phase 0.3
5. **NeuroBench streaming benchmarks** — extend ecoPrimals ESN — Phase 0.3
6. **Rust `akida-models::builder`** — eliminate Python entirely — Phase 0.4
