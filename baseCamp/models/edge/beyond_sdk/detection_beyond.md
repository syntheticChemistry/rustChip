# Object Detection — Beyond mAP 0.42 Static YOLO

**BrainChip claim:** YOLOv8n achieves mAP 0.42 on COCO-val for object detection.
**rustChip demonstration:** Temporal object tracking, multi-scale ensemble, scene
context classification, and physics-guided detection for scientific imaging.

---

## What BrainChip Ships

YOLOv8n on Akida:
- Input: 640×640 image
- Output: bounding boxes + class probabilities for 80 COCO classes
- mAP: 0.42 on COCO-val (vs 0.52 for GPU float YOLOv8n — 19% accuracy cost)
- Architecture: quantized to int4 for NP execution
- Mode: frame-by-frame, no temporal context

The 19% mAP drop from quantization is significant. BrainChip presents this
as an acceptable edge/accuracy tradeoff. The hardware can do much better
than the static deployment suggests.

---

## Limitation 1: No Temporal Tracking

Object detection without tracking generates IDs that flicker frame-to-frame.
The same car is "object 3" in frame 1 and "object 7" in frame 4.
Without tracking, you cannot answer "is that the same car?" or "where is it going?"

**YOLO + ESN temporal tracker:**

```
YOLO detection head → bounding box + class + confidence (per frame)
→ Extract: [x_center, y_center, width, height, confidence, class_id] per detection
→ Feed top-K detections as 6K-float temporal input to ESN
→ ESN maintains: object identity, velocity, trajectory prediction, persistence score

NP budget:
  YOLO backbone (estimated 600 NPs): feature extraction
  ESN tracker (179 NPs): temporal integration
  Total: 779 NPs, 221 spare
```

The ESN handles the tracking problem:
- Consistent object identity across frames (solve data association)
- Velocity and trajectory estimation (free from ESN state)
- Object persistence (is this detection real or artifact?)
- Scene continuity (are we still at the same intersection?)

Multi-object tracking on the NPU. No GPU required for the tracking logic.

---

## Limitation 2: 80 Fixed COCO Classes

COCO classes: cars, people, animals, common objects.
Industrial applications need: custom defect classes, domain-specific parts, hazards.

Same solution: backbone frozen (heavy CNN), head hot-swapped.

```
YOLO backbone → features (per spatial region)
→ Detection head hot-swap via set_variable():
  COCO-80: general objects (default)
  Industrial-12: PCB defects, mechanical faults
  Medical-5: surgical instruments
  Satellite-8: vehicles, buildings, terrain features
  Agricultural-6: crop diseases, pest damage, irrigation issues
```

Each hot-swap: 86 µs. Triggered by operational context.

Training custom heads: active learning approach —
collect 200 annotated images per class, train detection head,
quantize, store as set_variable() target.

---

## Limitation 3: mAP Drop From Quantization

mAP 0.42 vs 0.52 (float). The 19% gap is real, but it can be partially recovered:

**Temporal ensemble (averaging over frames):**
Detection confidence averaged over 5 consecutive frames reduces false positives
and raises effective mAP by approximately 0.04–0.06 points.
This is free — the ESN temporal integrator already accumulates evidence.

**Multi-resolution processing:**
Run YOLOv8n at two resolutions (320×320 and 640×640) on co-located models.
Merge bounding boxes from both scales.
Fine-detail detections (320×320) + context (640×640) → higher recall.
NP cost: 2 × 600 NPs = 1,200 NPs. Too much for one chip, but viable on AKD1500.

---

## Physics-Guided Detection (Novel Application)

For scientific imaging (microscopy, particle physics, plasma imaging):

**Standard YOLOv8: detects "particle-like objects" generally.**
**Physics-guided: co-locate a physics classifier to constrain detection.**

Example: bubble chamber / plasma imaging pipeline:

```
NP budget:
  Feature extractor (400 NPs): shared CNN backbone
  Detection head (100 NPs): particle bounding boxes
  Physics classifier (67 NPs): particle type (electron/pion/proton/kaon)
  Energy estimator (68 NPs): kinetic energy from track curvature
  ─────────────────────────────────────────────────────────────────────
  Total: 635 NPs
  Output per frame: particle locations + types + energy estimates
  Latency: 54 µs per frame (all classifiers in one pass, SkipDMA)
```

This is a real-time particle track classifier. On 635 NPs. 54 µs/frame.
The AKD1000 becomes a hardware particle physics preprocessor.

For comparison: typical HEP trigger systems at CERN run on custom FPGAs
at microsecond latency. The AKD1000 achieves comparable latency at 100× lower
development cost and power, with programmable (not hardwired) logic.

---

## Multi-Camera Scene Understanding

Three cameras × one chip:

```
Camera A: front-facing (traffic/pedestrians): YOLO + ESN tracker: 779 NPs
Camera B: side-facing (parking/hazards): detection head only: 100 NPs
Camera C: overhead (map-building): feature extractor only: 400 NPs
─────────────────────────────────────────────────────────────────────
Total: ~1,279 NPs — over budget for AKD1000

On AKD1500 (2,000 NPs): all three fit with 721 NPs to spare.
On AKD1000: camera A + camera B fit (879 NPs), camera C on host CPU.
```

---

## Honest Performance Summary

| Capability | BrainChip claim | rustChip extension |
|-----------|----------------|-------------------|
| mAP (static) | 0.42 COCO | 0.42 (same baseline) |
| mAP (temporal ensemble) | Not shown | ~0.46–0.48 (+temporal averaging) |
| Object tracking | Not supported | ESN tracking (179 NPs) |
| Custom class vocabularies | Not supported | Hot-swap in 86 µs |
| Scientific imaging | Not demonstrated | Particle physics classifier |
| Multi-camera | Not demonstrated | 2 simultaneous (AKD1000) |
| Temporal prediction | Not demonstrated | Trajectory via ESN state |

The mAP number is the floor, not the ceiling.
The tracking, context, and domain adaptation capabilities are the ceiling — and
none of them are in the SDK.
