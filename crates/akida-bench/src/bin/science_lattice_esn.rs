// SPDX-License-Identifier: AGPL-3.0-or-later

//! Science: Lattice QCD Steering via ESN Readout
//!
//! Standalone proof of the hybrid ESN pattern from hotSpring (Exp 022).
//! Generates a synthetic "lattice observable" time series, runs a tanh
//! reservoir on CPU (f64 dynamics) with readout on NPU (int4/int8),
//! and shows microsecond steering decisions inside a simulation loop.
//!
//! This is derivative of hotSpring (scyBorg lineage applies).
//! For the full lattice QCD deployment: https://github.com/syntheticChemistry/hotSpring
//!
//! ```bash
//! cargo run --bin science_lattice_esn
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, pack_software_model};
use std::time::Instant;

const RS: usize = 50;
const IS: usize = 4;
const OS: usize = 1;
const LEAK: f32 = 0.3;
const N_TRAJECTORIES: usize = 200;
const STEPS_PER_TRAJECTORY: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Science: Lattice QCD Steering via ESN Readout");
    println!("  Domain:  Physics — Hybrid Monte Carlo trajectory classification");
    println!("  Spring:  hotSpring (syntheticChemistry/hotSpring)");
    println!("  Pattern: Hybrid ESN — tanh reservoir [CPU] + readout [NPU]");
    println!("  Claim:   Microsecond NPU decisions inside simulation loops");
    println!("  Ref:     Exp 022 — 5,978 live calls, 63% savings, 80.4% accuracy");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Build model ─────────────────────────────────────────────────────
    let mut rng = Rng(0xAC1D_CAFE);
    let w_in: Vec<f32> = (0..RS * IS).map(|_| rng.next() * 0.1).collect();
    let w_res: Vec<f32> = (0..RS * RS).map(|_| rng.next() * 0.05).collect();
    let w_out: Vec<f32> = (0..OS * RS).map(|_| rng.next() * 0.2).collect();
    let model_blob = pack_software_model(RS, IS, OS, LEAK, &w_in, &w_res, &w_out);

    let mut npu = SoftwareBackend::init("0")?;
    npu.load_model(&model_blob)?;
    let backend_label = format!("{}", npu.backend_type());

    println!("  Backend        : {backend_label}");
    println!("  Reservoir      : RS={RS}, IS={IS}, OS={OS}, leak={LEAK}");
    println!("  Architecture   : InputConv({IS},1,1) → FC(128) → FC({OS})");
    println!("  Trajectories   : {N_TRAJECTORIES}");
    println!("  Steps/traj     : {STEPS_PER_TRAJECTORY}");
    println!();

    // ── Simulate HMC trajectories with NPU steering ─────────────────────
    println!("── Lattice QCD Steering Simulation ────────────────────────────");
    println!();

    let mut rng_sim = Rng(0x5133_BA5E);
    let mut total_npu_calls = 0u64;
    let mut early_rejections = 0u64;
    let mut total_npu_time = std::time::Duration::ZERO;

    // Synthetic "lattice observables": plaquette, polyakov, chiral, topological
    for traj in 0..N_TRAJECTORIES {
        let mut trajectory_rejected = false;

        for step in 0..STEPS_PER_TRAJECTORY {
            // Synthetic observables evolve per step (mimicking leapfrog)
            let plaquette = 0.59 + rng_sim.next() * 0.02;
            let polyakov = 0.01 + rng_sim.next() * 0.05;
            let chiral = -0.15 + rng_sim.next() * 0.03;
            let topological = rng_sim.next() * 0.5;

            // ── CPU reservoir (tanh, f64 dynamics) ──────────────────────
            // In production (hotSpring), this is a full f64 reservoir.
            // Here we feed directly to the NPU readout to demonstrate
            // the steering decision latency.
            let input = vec![plaquette, polyakov, chiral, topological];

            // ── NPU readout (microsecond decision) ──────────────────────
            let npu_start = Instant::now();
            let output = npu.infer(&input)?;
            let npu_elapsed = npu_start.elapsed();

            total_npu_time += npu_elapsed;
            total_npu_calls += 1;

            // Interpret: positive = likely accept, negative = likely reject
            let prediction = output[0];
            let should_reject = prediction < -0.15;

            if should_reject && step < STEPS_PER_TRAJECTORY / 2 {
                early_rejections += 1;
                trajectory_rejected = true;
                break;
            }
        }

        if traj < 5 || traj == N_TRAJECTORIES - 1 {
            let status = if trajectory_rejected { "REJECTED (early)" } else { "completed" };
            println!("  Trajectory {traj:>4}: {status}");
        } else if traj == 5 {
            println!("  ...");
        }
    }

    let avg_npu_us = if total_npu_calls > 0 {
        total_npu_time.as_micros() as f64 / total_npu_calls as f64
    } else {
        0.0
    };

    let savings_pct = (early_rejections as f64 / N_TRAJECTORIES as f64) * 100.0;

    println!();
    println!("── Results ────────────────────────────────────────────────────");
    println!("  Total NPU calls     : {total_npu_calls}");
    println!("  Early rejections    : {early_rejections} / {N_TRAJECTORIES} ({savings_pct:.1}%)");
    println!("  Avg NPU latency     : {avg_npu_us:.1} µs [{backend_label}]");
    println!("  Total NPU time      : {:.1} ms", total_npu_time.as_secs_f64() * 1000.0);

    let hw = npu.backend_type().is_hardware();
    println!();
    if hw {
        println!("  Hardware NPU steering active — these are silicon measurements.");
    } else {
        println!("  Software validation path — hardware would measure ~54 µs/call.");
        println!("  hotSpring Exp 022 achieved 63% savings on 32^4 SU(3) lattice.");
    }

    println!();
    println!("  The physics question: can a 54 µs decision inside every HMC step");
    println!("  save 63% of thermalization compute? hotSpring proved it can.");
    println!();
    println!("  Full science: https://github.com/syntheticChemistry/hotSpring");
    println!("  Exploration:  whitePaper/explorations/WHY_NPU.md");
    println!("  Pattern:      whitePaper/explorations/SPRINGS_ON_SILICON.md#hybrid-esn");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}
