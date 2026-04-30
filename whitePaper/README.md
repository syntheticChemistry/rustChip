# rustChip White Paper

Scientific write-ups, outreach materials, and exploratory analyses.

**Ecosystem context**: rustChip is **infrastructure** — a self-sufficient
outreach artifact, not a primal or a spring. It lives at `infra/rustChip`
in the [ecoPrimals](https://github.com/ecoPrimals) workspace. For the
full evolution story, see [`specs/EVOLUTION.md`](../specs/EVOLUTION.md).
For the broader project, see [primals.eco](https://primals.eco).

---

## Contents

| Directory | Contents |
|-----------|----------|
| [`outreach/akida/`](outreach/akida/) | Technical brief and benchmark datasheet for BrainChip team |
| [`explorations/`](explorations/) | Deep technical analyses of specific questions |

---

## Explorations Index

| Document | Question answered |
|----------|------------------|
| [`explorations/WHY_NPU.md`](explorations/WHY_NPU.md) | **The foundational neuromorphic argument** — microsecond decisions, energy economics, hybrid ESNs, streaming, and the NPU as a decision layer. Grounded in hardware evidence from springs. |
| [`explorations/SPRINGS_ON_SILICON.md`](explorations/SPRINGS_ON_SILICON.md) | **5 NPU patterns × 3 science domains** — Hybrid ESN, Microsecond Gatekeeper, Streaming Sentinel, Online Adaptation, Precision Discipline mapped across physics, biology, and agriculture. |
| [`explorations/NPU_FRONTIERS.md`](explorations/NPU_FRONTIERS.md) | **10 creative frontiers** — ensemble NPUs, NPU as scientific instrument, GPU-NPU co-location, pangenome triage, neuromorphic control loops, multi-modal pipelines, and more. |
| [`explorations/NPU_ON_GPU_DIE.md`](explorations/NPU_ON_GPU_DIE.md) | What if the NPU were a functional unit on the GPU die? Area, power, and latency analysis for integrated neuromorphic compute. |
| [`explorations/GPU_NPU_PCIE.md`](explorations/GPU_NPU_PCIE.md) | How does GPU+NPU co-location work over PCIe? Can data go directly from GPU BAR to NPU? |
| [`explorations/RUST_AT_SILICON.md`](explorations/RUST_AT_SILICON.md) | What would it take to go full Rust from application to silicon? What's the timeline? |
| [`explorations/VFIO_VS_KMOD.md`](explorations/VFIO_VS_KMOD.md) | VFIO userspace driver vs C kernel module — tradeoffs and migration strategy |
| [`explorations/TANH_CONSTRAINT.md`](explorations/TANH_CONSTRAINT.md) | The bounded ReLU finding: AKD1000 does not implement tanh — impact on ESN architectures |

---

## Outreach

`outreach/akida/` is the standing technical brief directed at BrainChip's
engineering team. It documents:
- What we measured on their hardware
- What SDK assumptions were overturned by direct probing
- Why a Rust driver exists and what it achieves
- What collaboration would accelerate

This is not a sales pitch. It is a technical report from an independent
team running real physics workloads on real silicon, with all data published
under AGPL-3.0. In the gen3/gen4 licensing framing, rustChip is a
symbiotic exception candidate — see `gen3/about/SCYBORG_EXCEPTION_PROTOCOL.md`
in the wider `infra/whitePaper/` directory.

---

## Relationship to docs/

`docs/` contains the raw measurement documents (BEYOND_SDK.md, HARDWARE.md,
TECHNICAL_BRIEF.md, BENCHMARK_DATASHEET.md) — the ground truth.

`whitePaper/` contains derived analyses and outreach materials that build
on that ground truth.

When `docs/` is updated, relevant `whitePaper/` documents should be revisited.
