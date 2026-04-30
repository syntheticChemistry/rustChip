# NPU Frontiers — What Can Neuromorphic Hardware Explore?

**Date:** April 2026
**Status:** Living document
**License:** CC-BY-SA 4.0

We measured the AKD1000. We know what it does: 54 µs inference, 1.4 µJ
per decision, 80 neural processors, 10 MB SRAM. Three springs deployed
it across physics, biology, and agriculture.

Now the question: what else?

This document is not a roadmap. It is a set of open questions — each one
an experiment waiting to happen. Some require only software. Some require
hardware we have. Some require hardware that doesn't exist yet.

---

## 1. From 5,978 to 500,000

hotSpring's Exp 022 made 5,978 NPU calls over 24 hours of lattice QCD
simulation. That is roughly one NPU call every 14 seconds — because the
GPU HMC trajectories take that long.

But the NPU is idle between calls. It can handle 18,500 per second. What
if we used that idle capacity?

**Multi-observable steering:** Instead of one readout per trajectory,
evaluate multiple observables at every leapfrog step within the trajectory.
Plaquette, Polyakov loop, chiral condensate, topological charge — each
gets its own readout head. The NPU evaluates all four in ~220 µs (4 x 54 µs).
The leapfrog step takes milliseconds. The NPU fits inside every step.

**Adaptive step size:** Use the NPU to predict optimal leapfrog step size
for the next integration step, based on the current phase-space position.
Instead of a fixed step size (which wastes acceptance rate), the NPU
suggests a step size that maximizes acceptance probability. The readout
is a regression: phase-space features → optimal dt.

**Early termination with confidence:** Instead of a binary accept/reject
prediction, output a confidence score. High-confidence rejections abort
immediately. Low-confidence predictions let the trajectory continue. The
threshold is tunable — trading NPU compute for GPU savings.

If step-level steering works, a 24-hour simulation could make **500,000+
NPU calls**. The physics question: does intra-trajectory steering change
the statistical quality of the ensemble? Does it introduce autocorrelation?
Does it break detailed balance?

These are physics questions that can only be answered by running the
experiment on hardware.

**Prerequisite:** hotSpring + rustChip integration at leapfrog level.
**Hardware:** AKD1000 (have it). GPU for lattice simulation (have it).

---

## 2. Ensemble NPU — Agreement as Signal

The `ensemble_npu` binary already runs multiple backends on the same
input and compares outputs. But agreement itself is information.

**Disagreement as anomaly signal:** When three ESN readouts (different
reservoir sizes, different spectral radii) agree, the classification is
high-confidence. When they disagree, the input is near a decision
boundary — which is exactly where the interesting science lives.

In lattice QCD: disagreement near a phase transition tells you the
configuration is at the boundary between confinement and deconfinement.
In biology: disagreement on a quorum sensing sample tells you the
population is at a phase transition between induction and competence.
In agriculture: disagreement on a crop stress reading tells you the
plant is at the boundary between healthy and stressed.

The ensemble doesn't just classify. It maps decision boundaries.

**Multi-chip voting:** The metalMatrix has slots for three AKD1000s.
What if each chip runs a different model — different reservoir sizes,
different quantization levels, different readout architectures — and
they vote? Hardware-level ensemble with PCIe-level parallelism.

**Prerequisite:** Multi-device support in `akida-driver`. Code exists
for single-device; multi-device enumeration works but parallel inference
is not yet orchestrated.

---

## 3. The NPU as a Scientific Instrument

We have been running models on the NPU. What about running the NPU
itself as an experiment?

**SRAM dynamics:** The AKD1000 has 10 MB of SRAM distributed across 80
neural processors. What patterns appear in SRAM content after inference?
Are there residual activation patterns? Do they correlate with input
structure? SRAM is readable via BAR1 — we can dump the full 10 MB after
every inference and study the spatial distribution of activations across
NPs.

**Weight sensitivity surfaces:** For a given model, perturb each weight
by a small delta and measure the output change. Map the full sensitivity
surface. This is computationally expensive on CPU (one inference per
perturbation) but trivial on NPU (18,500 perturbations per second). The
sensitivity surface reveals which weights matter and which are redundant
— a form of hardware-accelerated interpretability.

**Neuromorphic dynamics:** The AKD1000 uses integrate-and-fire neurons
with programmable thresholds. What happens at the boundary? Set thresholds
near firing point and study the statistical distribution of outputs.
Are there stochastic effects from quantization? Can we use the hardware
as a source of structured randomness?

**Spike propagation timing:** Load a model that chains NPs end-to-end
(FC → FC → FC → ... → FC, 80 layers, one NP each). Measure end-to-end
latency as a function of chain length. The slope reveals the per-NP
propagation delay. Map the timing across different NP positions on the
mesh. Are some NPs faster than others? Does the mesh topology create
latency asymmetries?

**Prerequisite:** BAR1 read access (have it via `probe_sram`). VFIO
with firmware alive for controlled inference (needs warm boot or
kernel driver cooperation).

---

## 4. GPU-NPU Co-Location: Removing the CPU

Current architecture:

```
GPU ──PCIe──> CPU memory ──PCIe──> NPU
```

The CPU mediates every data transfer. GPU results go to host memory,
then the CPU copies them to the NPU via DMA. Round-trip: ~200 µs
overhead on top of the 54 µs inference.

**PCIe peer-to-peer:** Modern platforms support P2P DMA between PCIe
devices. GPU writes directly to NPU BAR0 (or vice versa). No CPU copy.

```
GPU ──P2P DMA──> NPU
```

This would reduce the GPU→NPU→GPU round-trip from ~300 µs to potentially
~100 µs (dominated by PCIe link latency, not CPU copies).

**Implications for streaming:** If the GPU can push inference requests
directly to the NPU without CPU involvement, the NPU becomes a hardware
accelerator that the GPU treats like a coprocessor. GPU compute kernels
could include NPU calls as a subroutine — launch GPU kernel, write
intermediate result to NPU, NPU classifies, GPU reads result back, GPU
continues.

**Exploration exists:** [GPU_NPU_PCIE.md](GPU_NPU_PCIE.md) documents the
PCIe topology, current CPU mediation overhead, and P2P feasibility analysis.

**Prerequisite:** IOMMU configuration for P2P. Both devices in the same
IOMMU domain or P2P-compatible domains. coralReef's VFIO stack handles
multi-device IOMMU; rustChip's glowplug is the standalone subset.

---

## 5. NPU on GPU Die: The Neuromorphic Tile

What if the NPU was not a separate PCIe card but a tile on the GPU die?

**Exploration exists:** [NPU_ON_GPU_DIE.md](NPU_ON_GPU_DIE.md) analyzes
three levels of integration:

1. **Package-level:** NPU chiplet on GPU interposer. Shared HBM.
   Latency: ~50 ns (vs ~200 µs over PCIe). Energy: ~10 pJ per sample
   transfer (vs ~1 µJ over PCIe).

2. **Die-level:** NPU tile fabricated on the same die as GPU CUs.
   Shared L2 cache. Latency: ~10 ns. GPU CU writes to L2, NPU tile
   reads from L2. No DMA, no PCIe, no CPU.

3. **Integrated CU:** Neuromorphic processing elements inside existing
   GPU compute units. The CU switches between floating-point and
   spiking modes. Shared register file.

**Spring implications at die level:**

hotSpring: every leapfrog step writes lattice observables to L2.
The neuromorphic tile reads them in ~10 ns and classifies. The GPU
reads the classification result from L2 and decides the next step.
The entire steering loop fits inside a single GPU kernel — no kernel
launch, no DMA, no PCIe round-trip.

wetSpring: mass spectrometry preprocessing on GPU (FFT, peak detection)
writes candidate peaks to L2. The neuromorphic tile triages them. Only
interesting peaks get written to host memory for library matching. The
95.8% triage happens at L2 speed, not PCIe speed.

**This hardware does not exist.** But the software architecture in
rustChip is already structured to support it: the `NpuBackend` trait
abstracts the transport layer. A die-level NPU would implement
`NpuBackend` with L2 load/store instead of PCIe DMA. The science
code doesn't change. Only the backend changes.

---

## 6. Pangenome Triage at Hardware Speed

wetSpring and neuralSpring process genomic sequences: k-mer frequencies,
contig binning, introgression detection. These are classification tasks
on fixed-size feature vectors.

The bottleneck in genomic pipelines is not the classifier — it is the
volume. A single metagenomic sample produces millions of contigs. Each
contig needs classification: which genome bin does it belong to?

At 18,500 classifications per second (AKD1000 throughput), a million
contigs take ~54 seconds. On CPU, the same million take ~300 seconds
(~0.3 ms per classification).

But the real question is triage: not all contigs need the expensive
placement algorithm. Most are unambiguous — they clearly belong to one
bin. Only the ambiguous ones (near bin boundaries) need full placement.

**NPU triage for pangenomics:** the same gatekeeper pattern that
wetSpring uses for spectral triage. NPU classifies each contig in 22 µs.
High-confidence bins: accepted directly. Low-confidence: sent to the
full placement algorithm.

If 90% of contigs are high-confidence (typical for well-sampled
environments), triage reduces the expensive placement load by 10x.
The pipeline processes a million contigs in ~60 seconds (NPU triage) +
~30 seconds (CPU placement of 100k ambiguous contigs) = ~90 seconds,
vs ~300 seconds without triage.

**Prerequisite:** k-mer feature extraction pipeline → int8 quantization →
NPU inference. The feature extraction exists in wetSpring/neuralSpring.
The int8 path exists in rustChip. The integration is straightforward.

---

## 7. Neuromorphic Control Loops

Every pattern so far is classification or regression — the NPU observes
and reports. What about the NPU in the **feedback path** of a physical
system?

**Irrigation control (airSpring):** Not just "is this reading anomalous?"
but "how much water should this zone get in the next cycle?" The NPU
outputs a continuous control signal, not a class label. The readout is a
regression: sensor vector → irrigation duration in seconds.

The 48 µs inference latency means the control loop can run at sensor
rate. The control signal updates every reading, not every batch. This
is real-time closed-loop control with a neuromorphic controller.

**Bioreactor regulation (wetSpring):** Quorum sensing dynamics are
nonlinear and oscillatory. A neuromorphic controller trained on reservoir
states could regulate growth conditions (temperature, nutrient feed,
pH) to maintain the population in a target phase. The ESN reservoir
captures the nonlinear dynamics; the NPU readout outputs control
parameters.

**Simulation steering as control (hotSpring):** HMC step size
adaptation is already a control problem — adjust the integrator
step based on observed acceptance rate. The NPU can close this loop
at every trajectory step, not every trajectory.

**The question:** neuromorphic control is underexplored because hardware
was too slow or too expensive to sit in feedback loops. At 54 µs and
1.4 µJ, the AKD1000 changes the economics. What physical systems can
benefit from a microsecond controller that costs microjoules?

---

## 8. The NPU as a Random Number Generator

The AKD1000 uses fixed-point arithmetic with quantized weights. At the
boundary of decision thresholds, the output is sensitive to the least
significant bits of the computation. This sensitivity is deterministic
for a given input but practically unpredictable for inputs near the
boundary.

**Structured randomness:** Train a model where the output is balanced
between classes for random inputs. The NPU's quantization noise becomes
a source of structured randomness — not cryptographic, but useful for
Monte Carlo sampling, stochastic optimization, or evolutionary
algorithms.

**Hardware-in-the-loop evolution:** airSpring already uses (1+1)-ES for
weight evolution. What if the mutation step used NPU-generated random
perturbations instead of software PRNG? The hardware would generate its
own variation, evaluate its own fitness, and evolve its own weights —
a closed evolutionary loop on silicon.

**PUF (Physical Unclonable Function):** rustChip already has a PUF
module that reads chip-specific SRAM patterns. SRAM power-on values are
physically unique to each chip. This is not computation — it is physics.
Each AKD1000 has a unique SRAM fingerprint that can serve as a hardware
identity or entropy source.

---

## 9. Multi-Modal Neuromorphic Pipelines

The zoo has 28 models spanning vision, audio, spatiotemporal, edge,
and physics domains. What happens when you chain them?

**Vision → Physics:** A camera observes a physical experiment (bubble
chamber, plasma discharge, material fracture). AkidaNet classifies the
frame (vision model, ~54 µs). If the classification is "interesting
event," the frame is sent to GPU for detailed analysis. The NPU is the
gatekeeper between the camera and the compute.

**Audio → Biology:** A microphone monitors a bioreactor (wetSpring
territory). DS-CNN KWS detects acoustic signatures of contamination
or phase transitions. The NPU listens continuously at streaming rate.

**Spatiotemporal → Agriculture:** DVS (dynamic vision sensor) cameras
monitor crop canopy. TennSpatiotemporal classifies motion patterns
(wind, pest, growth). The event-driven sensor produces spikes; the
neuromorphic processor classifies spikes. The entire pipeline is
event-driven, not frame-based.

These are not hypothetical. The models exist in the zoo. The hardware
exists on the metalMatrix. The integration is software.

---

## 10. The Open Question

Every frontier in this document shares a structure: take a microsecond
decision and put it somewhere new. Inside a simulation loop. In a
feedback path. Between a sensor and a response. At the boundary of
what we understand.

The AKD1000 is the first chip we have. It is small (80 NPs), limited
(bounded ReLU, int4/int8), and well-characterized (metalForge, three
springs, 28 zoo models). It is also, as far as we can tell, the only
neuromorphic processor with a pure Rust driver, VFIO userspace access,
and published production deployment numbers in scientific computing.

The next chip might have more NPs, better nonlinearities, native tanh,
on-chip learning, or die-level GPU integration. The code is ready for
it — `NpuBackend` is a trait, `BackendType` distinguishes hardware from
software, `NpuLifecycle` handles vendor-specific transitions, and the
science demos work on any backend.

**The frontier is not the hardware. The frontier is the experiment.**

What would you do with a microsecond decision?

---

## References

- [WHY_NPU.md](WHY_NPU.md) — The foundational neuromorphic argument
- [SPRINGS_ON_SILICON.md](SPRINGS_ON_SILICON.md) — Cross-domain patterns
- [GPU_NPU_PCIE.md](GPU_NPU_PCIE.md) — GPU-NPU PCIe data paths
- [NPU_ON_GPU_DIE.md](NPU_ON_GPU_DIE.md) — NPU integrated on GPU die
- [TANH_CONSTRAINT.md](TANH_CONSTRAINT.md) — Bounded ReLU vs tanh
- [hotSpring](https://github.com/syntheticChemistry/hotSpring) — Lattice QCD
- [wetSpring](https://github.com/syntheticChemistry/wetSpring) — Sentinel microbe
- [airSpring](https://github.com/syntheticChemistry/airSpring) — Agricultural IoT
- [sporePrint / primals.eco](https://primals.eco) — Public verification
