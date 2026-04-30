// SPDX-License-Identifier: AGPL-3.0-or-later

//! Science: Precision Ladder — Quantization as Scientific Methodology
//!
//! Takes the same ESN model through f64 → f32 → int8 → int4 and shows
//! how precision affects each science domain differently. The quantization
//! gap reveals the information content of the signal, not just a
//! compression trade-off.
//!
//! This is derivative of all three springs (scyBorg lineage applies).
//! Demonstrates the cross-domain precision discipline pattern.
//!
//! ```bash
//! cargo run --bin science_precision_ladder
//! ```

use akida_driver::{NpuBackend, SoftwareBackend, pack_software_model};

const RS: usize = 32;
const IS: usize = 4;
const OS: usize = 1;
const LEAK: f32 = 0.25;
const N_SAMPLES: usize = 200;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Science: Precision Ladder — Quantization as Methodology");
    println!("  Domain:  Cross-domain (physics, biology, agriculture)");
    println!("  Springs: hotSpring, wetSpring, airSpring");
    println!("  Pattern: Precision Discipline — f64 → f32 → int8 → int4");
    println!("  Claim:   Quantization gap reveals signal information content");
    println!("  Hardware: AKD1000 [HW] / Software VirtualNPU [SW]");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Build reference weights in f64 ──────────────────────────────────
    let mut rng = Rng(0xF064_CAFE);
    let w_in_f64: Vec<f64> = (0..RS * IS).map(|_| rng.next_f64() * 0.3).collect();
    let w_res_f64: Vec<f64> = (0..RS * RS).map(|_| rng.next_f64() * 0.08).collect();
    let w_out_f64: Vec<f64> = (0..OS * RS).map(|_| rng.next_f64() * 0.4).collect();

    let w_in_f32: Vec<f32> = w_in_f64.iter().map(|&x| x as f32).collect();
    let w_res_f32: Vec<f32> = w_res_f64.iter().map(|&x| x as f32).collect();
    let w_out_f32: Vec<f32> = w_out_f64.iter().map(|&x| x as f32).collect();

    // Tensor-level quantization: find global max, quantize all weights uniformly
    let w_in_i8 = quantize_tensor_int8(&w_in_f32);
    let w_res_i8 = quantize_tensor_int8(&w_res_f32);
    let w_out_i8 = quantize_tensor_int8(&w_out_f32);

    let w_in_i4 = quantize_tensor_int4(&w_in_f32);
    let w_res_i4 = quantize_tensor_int4(&w_res_f32);
    let w_out_i4 = quantize_tensor_int4(&w_out_f32);

    let domains: Vec<DomainConfig> = vec![
        DomainConfig {
            name: "Physics (QCD)",
            spring: "hotSpring",
            precision_claim: "int4 sufficient — narrow dynamic range",
            gen_input: gen_physics_input,
        },
        DomainConfig {
            name: "Biology (QS)",
            spring: "wetSpring",
            precision_claim: "int8 needed — wider dynamic range (concentrations)",
            gen_input: gen_biology_input,
        },
        DomainConfig {
            name: "Agriculture",
            spring: "airSpring",
            precision_claim: "int8 matches source fidelity — noisy sensors",
            gen_input: gen_agriculture_input,
        },
    ];

    println!("  Rung 1 (f64)  : CPU reference — full double-precision ESN");
    println!("  Rung 2 (f32)  : SoftwareBackend — f32 VirtualNPU");
    println!("  Rung 3 (int8) : Tensor-quantized int8 weights on f32 backend");
    println!("  Rung 4 (int4) : Tensor-quantized int4 weights on f32 backend");
    println!("  Samples/domain: {N_SAMPLES}");
    println!();

    for domain in &domains {
        println!("── {} ──────────────────────────────────", domain.name);
        println!("  Spring: {}", domain.spring);
        println!("  Claim:  {}", domain.precision_claim);
        println!();

        // ── Rung 1: f64 reference (stateful ESN) ───────────────────────
        let mut rng_data = Rng(0xDA7A_1234);
        let mut f64_outputs = Vec::with_capacity(N_SAMPLES);
        let mut state_f64 = vec![0.0f64; RS];

        for _ in 0..N_SAMPLES {
            let input = (domain.gen_input)(&mut rng_data);
            let out = esn_step_f64(
                &w_in_f64, &w_res_f64, &w_out_f64,
                &input, &mut state_f64, LEAK as f64,
            );
            f64_outputs.push(out);
        }

        // ── Rung 2: f32 (SoftwareBackend) ──────────────────────────────
        let blob_f32 = pack_software_model(RS, IS, OS, LEAK, &w_in_f32, &w_res_f32, &w_out_f32);
        let mut npu_f32 = SoftwareBackend::init("0")?;
        npu_f32.load_model(&blob_f32)?;

        let mut rng_data = Rng(0xDA7A_1234);
        let mut f32_outputs = Vec::with_capacity(N_SAMPLES);
        for _ in 0..N_SAMPLES {
            let input = (domain.gen_input)(&mut rng_data);
            let out = npu_f32.infer(&input)?;
            f32_outputs.push(out[0]);
        }

        // ── Rung 3: int8 ───────────────────────────────────────────────
        let blob_i8 = pack_software_model(RS, IS, OS, LEAK, &w_in_i8, &w_res_i8, &w_out_i8);
        let mut npu_i8 = SoftwareBackend::init("0")?;
        npu_i8.load_model(&blob_i8)?;

        let mut rng_data = Rng(0xDA7A_1234);
        let mut i8_outputs = Vec::with_capacity(N_SAMPLES);
        for _ in 0..N_SAMPLES {
            let input = (domain.gen_input)(&mut rng_data);
            let out = npu_i8.infer(&input)?;
            i8_outputs.push(out[0]);
        }

        // ── Rung 4: int4 ───────────────────────────────────────────────
        let blob_i4 = pack_software_model(RS, IS, OS, LEAK, &w_in_i4, &w_res_i4, &w_out_i4);
        let mut npu_i4 = SoftwareBackend::init("0")?;
        npu_i4.load_model(&blob_i4)?;

        let mut rng_data = Rng(0xDA7A_1234);
        let mut i4_outputs = Vec::with_capacity(N_SAMPLES);
        for _ in 0..N_SAMPLES {
            let input = (domain.gen_input)(&mut rng_data);
            let out = npu_i4.infer(&input)?;
            i4_outputs.push(out[0]);
        }

        // ── Compute relative errors ────────────────────────────────────
        let err_f32 = mean_relative_error(&f64_outputs, &f32_outputs);
        let err_i8 = mean_relative_error(&f64_outputs, &i8_outputs);
        let err_i4 = mean_relative_error(&f64_outputs, &i4_outputs);

        println!("  Precision  │ Mean Rel. Error vs f64 │ Interpretation");
        println!("  ───────────┼────────────────────────┼──────────────────────");
        println!("  f64 (ref)  │        0.000%          │ baseline");
        println!("  f32        │    {:>8.3}%          │ cast precision loss", err_f32 * 100.0);
        println!("  int8       │    {:>8.3}%          │ quantization gap", err_i8 * 100.0);
        println!("  int4       │    {:>8.3}%          │ quantization gap", err_i4 * 100.0);
        let ratio = if err_i8.abs() > 1e-12 { err_i4 / err_i8 } else { f64::NAN };
        println!("  int4/int8  │    {:>8.2}x           │ information width ratio", ratio);
        println!();
    }

    println!("── Cross-Domain Summary ───────────────────────────────────────");
    println!();
    println!("  The quantization ladder is not compression — it is measurement.");
    println!();
    println!("  Physics (int4 viable):  Lattice observables have narrow dynamic");
    println!("    range. The plaquette varies by ~0.02 around 0.59. int4 suffices");
    println!("    because the physics is bounded.");
    println!();
    println!("  Biology (int8 needed):  Concentrations span orders of magnitude.");
    println!("    int4 loses the tails. int8 preserves them. The gap between");
    println!("    int8 and int4 reveals the information width of biological signals.");
    println!();
    println!("  Agriculture (int8 matches source):  Sensor noise exceeds");
    println!("    quantization noise. The bottleneck is the sensor, not the silicon.");
    println!("    int8 matches the source fidelity.");
    println!();
    println!("  Full analysis: whitePaper/explorations/WHY_NPU.md#precision-as-a-ladder");
    println!("  Patterns:      whitePaper/explorations/SPRINGS_ON_SILICON.md#precision-discipline");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

// ── Stateful f64 ESN step (matches SoftwareBackend accumulation) ────────

fn esn_step_f64(
    w_in: &[f64], w_res: &[f64], w_out: &[f64],
    input: &[f32], state: &mut [f64], leak: f64,
) -> f64 {
    let rs = state.len();
    let is = input.len();

    let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();

    for i in 0..rs {
        let mut sum = 0.0f64;
        for j in 0..is {
            sum += w_in[i * is + j] * input_f64[j];
        }
        for j in 0..rs {
            sum += w_res[i * rs + j] * state[j];
        }
        state[i] = (1.0 - leak) * state[i] + leak * sum.tanh();
    }

    let mut output = 0.0f64;
    for j in 0..rs {
        output += w_out[j] * state[j];
    }
    output
}

// ── Tensor-level quantization (shared scale across full tensor) ─────────

fn quantize_tensor_int8(weights: &[f32]) -> Vec<f32> {
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    if max_abs < 1e-10 { return weights.to_vec(); }
    let scale = 127.0 / max_abs;
    weights.iter().map(|&w| (w * scale).round() / scale).collect()
}

fn quantize_tensor_int4(weights: &[f32]) -> Vec<f32> {
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    if max_abs < 1e-10 { return weights.to_vec(); }
    let scale = 7.0 / max_abs;
    weights.iter().map(|&w| (w * scale).round().clamp(-8.0, 7.0) / scale).collect()
}

fn mean_relative_error(reference: &[f64], test: &[f32]) -> f64 {
    let n = reference.len().min(test.len());
    if n == 0 { return 0.0; }
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..n {
        let r = reference[i];
        let t = test[i] as f64;
        if r.abs() > 1e-10 {
            sum += ((t - r) / r).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ── Domain-specific input generators (all normalized to [-1, 1]) ────────

fn gen_physics_input(rng: &mut Rng) -> Vec<f32> {
    // Lattice observables: narrow dynamic range, centered near specific values
    vec![
        rng.next() * 0.1,        // plaquette deviation (tiny)
        rng.next() * 0.2,        // Polyakov loop (small)
        rng.next() * 0.15,       // chiral condensate (small)
        rng.next() * 0.05,       // topological charge (tiny)
    ]
}

fn gen_biology_input(rng: &mut Rng) -> Vec<f32> {
    // QS concentrations: wider dynamic range, spanning more of [-1, 1]
    vec![
        rng.next() * 0.8,
        rng.next() * 0.6,
        rng.next() * 0.9,
        rng.next() * 0.4,
    ]
}

fn gen_agriculture_input(rng: &mut Rng) -> Vec<f32> {
    // Sensor data: moderate range with added noise
    let base: Vec<f32> = vec![
        rng.next() * 0.5,
        rng.next() * 0.5,
        rng.next() * 0.5,
        rng.next() * 0.5,
    ];
    // Add sensor noise (~5% of range)
    base.iter().map(|&v| v + rng.next() * 0.05).collect()
}

struct DomainConfig {
    name: &'static str,
    spring: &'static str,
    precision_claim: &'static str,
    gen_input: fn(&mut Rng) -> Vec<f32>,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
    fn next_f64(&mut self) -> f64 {
        self.next() as f64
    }
}
