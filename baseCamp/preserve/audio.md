# Audio — Keyword Spotting, Streaming Speech, Acoustic Events

**Spring origin:** Zoo (BrainChip pretrained models)
**Maturity:** Models parsed and validated; integration pattern documented
**Zoo models:** DS-CNN KWS, TENN Recurrent SC12, TENN Recurrent UORED

---

## Problem

Audio classification at the edge: a microphone produces a continuous stream
of samples, and the system must detect keywords, classify sounds, or
transcribe speech — all with sub-100ms latency, minimal power, and no
cloud connectivity.

Neuromorphic processors excel here because audio is inherently temporal
and sparse. The AKD1000 processes quantized feature frames at microsecond
latency, making real-time keyword detection practical on battery-powered
devices.

Three sub-problems:

| Task | Question | Latency budget |
|------|----------|---------------|
| **Keyword spotting** | Did the user say one of 33 wake words? | < 100 ms |
| **Speaker command (12-class)** | Which of 12 commands was spoken? | < 100 ms |
| **Utterance recognition (4-class)** | Which of 4 utterance types? | < 100 ms |

---

## Model

### DS-CNN Keyword Spotting

```
Architecture: Depthwise-separable CNN, 9 layers
Quantization: int4 (BrainChip MetaTF)
Input:        49×10×1 (MFCC spectrogram, ~1 second of audio)
Output:       [1,1,33] — 33 keyword classes
.fbz:         baseCamp/zoo-artifacts/ds_cnn_kws.fbz (41 KB)
Parse time:   3.8 ms
```

This is the smallest model in the zoo — 41 KB on disk. It demonstrates
that neuromorphic keyword spotting is viable with minimal resources.

### TENN Recurrent SC12 (12-class speaker command)

```
Architecture: Temporal Event Neural Network (recurrent), streaming
Quantization: int4 (BrainChip MetaTF)
Input:        1×256×1 (streaming audio features)
Output:       [1,1,12] — 12 command classes
.fbz:         baseCamp/zoo-artifacts/tenn_recurrent_sc12.fbz (70 KB)
```

TENN (Temporal Event Neural Network) is BrainChip's temporal architecture
that processes events as they arrive, maintaining internal state between
frames. This enables true streaming inference — no windowing, no overlap,
just continuous processing.

### TENN Recurrent UORED (4-class utterance)

```
Architecture: Temporal Event Neural Network (recurrent), streaming
Quantization: int4 (BrainChip MetaTF)
Input:        1×256×1 (streaming audio features)
Output:       [1,1,4] — 4 utterance classes
.fbz:         baseCamp/zoo-artifacts/tenn_recurrent_uored.fbz (37 KB)
```

---

## Rust Path

### Parse and inspect

```rust
use akida_models::prelude::*;

let model = Model::from_file("baseCamp/zoo-artifacts/ds_cnn_kws.fbz")?;
println!("DS-CNN KWS: {} layers, {} bytes", model.layer_count(), model.file_size());
// DS-CNN KWS: 9 layers, 41360 bytes
```

### Feature extraction (MFCC)

Audio → MFCC conversion is the domain-specific step. In Rust, you can use
crates like `mel-spec` or implement directly:

```rust
fn compute_mfcc(
    audio_samples: &[f32],
    sample_rate: u32,
    n_mfcc: usize,
    n_frames: usize,
) -> Vec<f32> {
    // Window: 30ms frames, 10ms hop → 49 frames for ~1 second
    let frame_len = (sample_rate as f32 * 0.030) as usize;
    let hop_len = (sample_rate as f32 * 0.010) as usize;

    let mut features = Vec::with_capacity(n_frames * n_mfcc);
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_len;
        let frame = &audio_samples[start..start + frame_len];
        // FFT → mel filterbank → log → DCT → first n_mfcc coefficients
        let mfcc = mfcc_single_frame(frame, sample_rate, n_mfcc);
        features.extend_from_slice(&mfcc);
    }
    features
}
```

### Inference

```rust
let features = compute_mfcc(&audio_buffer, 16000, 10, 49);
// features shape: [49, 10, 1] — matches DS-CNN KWS input

let backend = akida_driver::SoftwareBackend::new();
let result = backend.infer(&model, &features)?;

let keyword_idx = result.iter().copied().enumerate()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .map(|(idx, _)| idx)
    .unwrap();

println!("Detected keyword: {}", KEYWORD_LABELS[keyword_idx]);
```

### Streaming with TENN

```rust
let model = Model::from_file("baseCamp/zoo-artifacts/tenn_recurrent_sc12.fbz")?;

// TENN models process streaming frames — no windowing needed
loop {
    let frame = capture_audio_frame(256); // 256 features per frame
    let result = backend.infer(&model, &frame)?;

    let command = result.iter().copied().enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    if result[command] > CONFIDENCE_THRESHOLD {
        execute_command(command);
    }
}
```

---

## Output

| Metric | Value | Notes |
|--------|-------|-------|
| DS-CNN KWS model size | 41 KB (.fbz) | Smallest zoo model |
| DS-CNN KWS parse time | 3.8 ms | Cold parse, no cache |
| TENN SC12 model size | 70 KB (.fbz) | Streaming architecture |
| TENN UORED model size | 37 KB (.fbz) | Minimal utterance classifier |
| All audio models parsed | 3/3 | `zoo_regression.rs` |

---

## Extension Points

**Custom wake words.** Train a DS-CNN variant on your vocabulary (medical
commands, industrial controls, accessibility phrases), export weights to
`.npy`, convert via `akida convert --bits 4`.

**Multi-language keyword spotting.** The 33-class DS-CNN architecture
generalizes to other languages by retraining on localized speech datasets.

**Acoustic event detection.** Replace keyword labels with environmental
sounds (glass breaking, machinery fault, animal calls). The MFCC feature
extraction is domain-agnostic.

**Continuous dictation.** Chain TENN models with a CTC decoder for
continuous speech recognition. The streaming architecture avoids the
latency penalty of windowed approaches.

**Sensor fusion.** Combine audio classification with the streaming sensor
pattern (see [Industrial](industrial.md)) — audio + vibration + temperature
for multi-modal anomaly detection.

---

## CLI Quick Test

```bash
# Parse all audio models
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/ds_cnn_kws.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/tenn_recurrent_sc12.fbz
cargo run -p akida-cli -- parse baseCamp/zoo-artifacts/tenn_recurrent_uored.fbz
```
