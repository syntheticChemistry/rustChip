# DVS Gesture Recognition

**Architecture:** Event-based CNN → FC(128→11)
**Status:** 📋 Analysis complete; event data loading queued
**Task:** Classify 11 hand gestures from DVS128 event camera
**Source:** NeuroBench gesture benchmark; IBM DVS128 Gesture Dataset

---

## What makes this model special

The DVS128 camera is an event-based sensor — it outputs (x, y, t, polarity)
events rather than frames. This is the natural modality for spiking neural
networks: events are already spike-like.

On Akida, event streams are converted to frame-based representations
(event count frames or time-surface frames) for the CNN front-end.
This is a concession to Akida's current architecture — the AKD1000 processes
frames, not raw event streams.

Future AKD1500 and beyond may support direct event stream input.

---

## Architecture

```
Input: float[64×64×2]  (64×64 spatial, 2 channels: ON/OFF events per frame)
  │
  ▼
InputConv(2→32, kernel=3×3)
  │
  ▼
Conv(32→64, kernel=3×3, stride=2)    ← spatial downsampling
  │
  ▼
Conv(64→128, kernel=3×3, stride=2)
  │
  ▼
GlobalAveragePooling → FC(128→11)
  │  softmax → one of 11 gesture classes
  ▼
Output: float[11]
```

Classes: hand clap, right-hand clockwise rotation, right-hand counter-CW,
left-hand clockwise, left-hand counter-CW, arm roll CW, arm roll CCW,
air drums, air guitar, other gesture, random noise.

---

## Event-to-Frame Conversion

```rust
// Sketch: event stream → frame for Akida input
// Each DVS128 event: (x: u8, y: u8, t: u32, polarity: bool)
// Window: T_frame = 40ms (25 Hz frame rate)

pub struct DvsFrameAccumulator {
    width: usize,   // 64 for DVS128
    height: usize,  // 64 for DVS128
    on_counts: Array2<u32>,
    off_counts: Array2<u32>,
}

impl DvsFrameAccumulator {
    pub fn add_event(&mut self, x: usize, y: usize, polarity: bool) {
        if polarity { self.on_counts[[y, x]] += 1; }
        else        { self.off_counts[[y, x]] += 1; }
    }

    pub fn to_normalized_frame(&self, max_count: u32) -> Array3<f32> {
        // Shape: [2, 64, 64] — ON channel, OFF channel
        // Normalized to [0, 1]
        // → flatten to [64×64×2 = 8192] for Akida input
    }
}
```

---

## metalForge Application

The DVS gesture model is the primary test case for event-based input
in rustChip's metalForge. Planned experiment:

```
metalForge/experiments/002_DVS_GESTURE_VFIO.md
  Protocol: DVS128 event stream → Rust accumulator → AKD1000 VFIO inference
  Goal: reproduce NeuroBench 97.9% accuracy with Rust driver
  Hardware: DVS128 camera (USB) + AKD1000 (PCIe)
  Expected result: match NeuroBench within ±0.5%
```

---

## ecoPrimals Extension: Environmental Events

Tonic datasets include events beyond gesture:
- **NCARS** (cars in urban scenes): vehicle detection for environmental monitoring
- **DVSLip** (lip reading): acoustic + visual multi-modal
- **Pokerdvh**: high-speed classification benchmark

The accumulation + CNN pattern extends to any event-based sensor:
- Acoustic transducers with spiking front-ends
- Chemical sensor arrays with event-like threshold crossings
- Field sensor anomaly triggers

The common thread: any physical event stream → DVS-style (x, t, polarity)
representation → same CNN → Akida execution.
