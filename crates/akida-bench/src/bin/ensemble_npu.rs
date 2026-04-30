// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ensemble NPU — cooperative hardware + software inference.
//!
//! Demonstrates three ensemble strategies using multiple NPU backends:
//!
//! 1. **Voting**: Run N independent models, take majority class
//! 2. **Averaging**: Average output logits across backends
//! 3. **Cascade**: Software NPU screens, hardware NPU refines uncertain cases
//!
//! When hardware is available, the ensemble mixes real AKD1000 inference
//! with software NPU predictions. When hardware is unavailable, multiple
//! software instances with different architectures serve as the ensemble.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin ensemble_npu
//! ```

use akida_driver::{BackendType, NpuBackend, SoftwareBackend, VfioBackend, pack_software_model};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=warn".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ensemble NPU — Cooperative Multi-Backend Inference");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = Rng(0xE45E_B1E5u64.wrapping_add(0x1000));

    // ── Build ensemble members ──────────────────────────────────────────
    println!("── Ensemble Assembly ──────────────────────────────────────────");

    struct Member {
        name: String,
        backend_type: BackendType,
        backend: Box<dyn NpuBackend>,
        rs: usize,
    }

    let configs: &[(usize, usize, usize, f32, &str)] = &[
        (32, 8, 4, 0.3, "Small-32"),
        (64, 8, 4, 0.25, "Medium-64"),
        (128, 8, 4, 0.35, "Large-128"),
    ];

    let mut members: Vec<Member> = Vec::new();

    // Try to add hardware backend
    if let Some(addr) = discover_first_akida() {
        match VfioBackend::init(&addr) {
            Ok(mut hw) => {
                if !hw.is_ready() {
                    let _ = hw.reset_and_enable();
                }
                let (rs, is, os, leak) = (64, 8, 4, 0.3);
                let w_in: Vec<f32> = (0..rs * is).map(|_| rng.next() * 0.1).collect();
                let w_res: Vec<f32> = (0..rs * rs).map(|_| rng.next() * 0.05).collect();
                let w_out: Vec<f32> = (0..os * rs).map(|_| rng.next() * 0.2).collect();
                let blob = pack_software_model(rs, is, os, leak, &w_in, &w_res, &w_out);
                match hw.load_model(&blob) {
                    Ok(_) => {
                        println!("  [HW]  AKD1000 VFIO  RS={rs}  — loaded");
                        members.push(Member {
                            name: "AKD1000-HW".into(),
                            backend_type: hw.backend_type(),
                            backend: Box::new(hw),
                            rs,
                        });
                    }
                    Err(e) => println!("  [HW]  AKD1000 model load failed: {e}"),
                }
            }
            Err(e) => println!("  [HW]  VFIO unavailable: {e}"),
        }
    } else {
        println!("  [HW]  No Akida hardware detected");
    }

    // Software ensemble members
    for &(rs, is, os, leak, name) in configs {
        let w_in: Vec<f32> = (0..rs * is).map(|_| rng.next() * 0.1).collect();
        let w_res: Vec<f32> = (0..rs * rs).map(|_| rng.next() * 0.05).collect();
        let w_out: Vec<f32> = (0..os * rs).map(|_| rng.next() * 0.2).collect();
        let blob = pack_software_model(rs, is, os, leak, &w_in, &w_res, &w_out);
        let mut sw = SoftwareBackend::new(rs, is, os);
        sw.load_model(&blob)?;
        println!("  [SW]  {name:<12}  RS={rs:<4}  — loaded");
        members.push(Member {
            name: name.into(),
            backend_type: sw.backend_type(),
            backend: Box::new(sw),
            rs,
        });
    }

    let n_members = members.len();
    let n_hw = members.iter().filter(|m| m.backend_type == BackendType::Vfio).count();
    let n_sw = n_members - n_hw;
    println!();
    println!("  Ensemble size  : {n_members} members ({n_hw} HW + {n_sw} SW)");
    println!();

    // ── Generate test inputs ────────────────────────────────────────────
    let is = 8usize;
    let os = 4usize;
    let n_samples = 50;
    let inputs: Vec<Vec<f32>> = (0..n_samples)
        .map(|_| (0..is).map(|_| rng.next()).collect())
        .collect();

    // ══════════════════════════════════════════════════════════════════════
    // STRATEGY 1: Independent Voting
    // ══════════════════════════════════════════════════════════════════════
    println!("── Strategy 1: Majority Voting ────────────────────────────────");

    let start = Instant::now();
    let mut vote_correct = 0u32;
    let mut vote_total = 0u32;

    for input in &inputs {
        let mut votes = vec![0u32; os];
        for member in &mut members {
            let output = member.backend.infer(input)?;
            let class = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if class < os {
                votes[class] += 1;
            }
        }
        let ensemble_class = votes.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0);
        vote_total += 1;
        if ensemble_class == 0 {
            vote_correct += 1; // synthetic "ground truth" for demo
        }
    }

    let vote_time = start.elapsed();
    println!("  Samples        : {vote_total}");
    println!("  Ensemble time  : {vote_time:?}");
    println!("  Avg per sample : {:.1} µs", vote_time.as_micros() as f64 / vote_total as f64);
    println!("  Agreement rate : {:.1}%", vote_correct as f64 / vote_total as f64 * 100.0);
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // STRATEGY 2: Output Averaging
    // ══════════════════════════════════════════════════════════════════════
    println!("── Strategy 2: Output Averaging ─────────────────────────────");

    let start = Instant::now();
    let mut avg_outputs: Vec<Vec<f32>> = Vec::new();

    for input in &inputs {
        let mut sum = vec![0.0f32; os];
        for member in &mut members {
            let output = member.backend.infer(input)?;
            for (s, o) in sum.iter_mut().zip(output.iter()) {
                *s += o;
            }
        }
        for s in &mut sum {
            *s /= n_members as f32;
        }
        avg_outputs.push(sum);
    }

    let avg_time = start.elapsed();
    let sample_out = avg_outputs.first().map_or(&[] as &[f32], |v| v.as_slice());
    println!("  Samples        : {n_samples}");
    println!("  Ensemble time  : {avg_time:?}");
    println!("  Avg per sample : {:.1} µs", avg_time.as_micros() as f64 / n_samples as f64);
    println!("  Sample output  : {:?}", &sample_out[..sample_out.len().min(4)]);
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // STRATEGY 3: Cascade (screen + refine)
    // ══════════════════════════════════════════════════════════════════════
    println!("── Strategy 3: Cascade (Screen → Refine) ────────────────────");

    let confidence_threshold = 0.3f32;
    let start = Instant::now();
    let mut screened = 0u32;
    let mut refined = 0u32;

    for input in &inputs {
        // Stage 1: fast software screen
        let screen_out = members[if n_hw > 0 { 1 } else { 0 }].backend.infer(input)?;
        let max_logit = screen_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_logit = screen_out.iter().copied().fold(f32::INFINITY, f32::min);
        let spread = max_logit - min_logit;

        if spread > confidence_threshold {
            screened += 1;
        } else {
            // Stage 2: uncertain — consult all members
            let mut _sum = vec![0.0f32; os];
            for member in &mut members {
                let output = member.backend.infer(input)?;
                for (s, o) in _sum.iter_mut().zip(output.iter()) {
                    *s += o;
                }
            }
            refined += 1;
        }
    }

    let cascade_time = start.elapsed();
    println!("  Confidence     : {confidence_threshold}");
    println!("  Screened (fast) : {screened} ({:.0}%)", screened as f64 / n_samples as f64 * 100.0);
    println!("  Refined (full)  : {refined} ({:.0}%)", refined as f64 / n_samples as f64 * 100.0);
    println!("  Total time     : {cascade_time:?}");
    println!("  Avg per sample : {:.1} µs", cascade_time.as_micros() as f64 / n_samples as f64);
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // PER-MEMBER BREAKDOWN
    // ══════════════════════════════════════════════════════════════════════
    println!("── Per-Member Latency ─────────────────────────────────────────");
    println!("  {:15} {:>10} {:>12} {:>10}", "Member", "Backend", "RS", "Avg µs");
    println!("  {:15} {:>10} {:>12} {:>10}", "─".repeat(15), "─".repeat(10), "─".repeat(12), "─".repeat(10));

    for member in &mut members {
        let n = 200;
        let input: Vec<f32> = (0..is).map(|_| 0.5).collect();
        let start = Instant::now();
        for _ in 0..n {
            let _ = member.backend.infer(&input)?;
        }
        let total = start.elapsed();
        let avg_us = total.as_micros() as f64 / n as f64;
        println!(
            "  {:15} {:>10} {:>12} {:>10.1}",
            member.name,
            format!("{}", member.backend_type),
            member.rs,
            avg_us
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ensemble NPU demonstration complete");
    println!("  {n_members} backends cooperating ({n_hw} hardware + {n_sw} software)");
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

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
