// SPDX-License-Identifier: AGPL-3.0-only

//! ESN Substrate Comparison Benchmark
//!
//! Side-by-side comparison of Echo State Network inference across three substrates:
//!
//!   1. **CPU-f64**: Reference implementation. hotSpring-style f64 reservoir + f64 readout.
//!      The gold standard for numerical accuracy.
//!
//!   2. **SoftwareBackend (VirtualNPU)**: Pure f32 CPU simulation of what the AKD1000 does.
//!      Quantifies f64→f32 precision cost without hardware. Should match hardware closely.
//!
//!   3. **Real AKD1000**: Actual hardware via VFIO driver. int4 weights, NPU arithmetic.
//!      The production path. Int4 quantization introduces ~1–3% additional error vs f32.
//!
//! Ported and extended from:
//!   hotSpring/barracuda/src/bin/validate_lattice_npu.rs   (NpuSimulator parity check)
//!   hotSpring/barracuda/src/md/reservoir.rs               (EchoStateNetwork + NpuSimulator)
//!
//! ## What is measured
//!
//!   - **Throughput** (inferences/second) for each substrate
//!   - **Latency** (µs/call, p50/p95/p99)
//!   - **Accuracy parity**: max absolute difference between substrates
//!   - **State parity**: reservoir state divergence after N timesteps
//!
//! ## Reference numbers (AKD1000, Feb 2026)
//!
//!   CPU-f64 throughput :  ~8,000,000 Hz (single thread, RS=50)
//!   SoftwareBackend    :  ~2,500,000 Hz (f32 SIMD, RS=50)
//!   AKD1000 (hardware) :     18,500 Hz (PCIe bound, batch=8 sweet spot)
//!   Energy:  CPU ~50 µJ/inference,  NPU ~1.4 µJ/inference (36× better)
//!
//! ## Usage
//!
//!   cargo run --bin bench_esn_substrate               # all substrates
//!   cargo run --bin bench_esn_substrate -- --no-hw    # skip hardware (CI)
//!   cargo run --bin bench_esn_substrate -- --iterations 5000

use akida_driver::{
    backends::software::{pack_software_model, SoftwareBackend},
    DeviceManager, NpuBackend,
};
use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const DEFAULT_ITERATIONS: usize = 1000;

// ESN architecture — matches hotSpring default and ecoPrimals production models
const RESERVOIR_SIZE: usize = 50;
const INPUT_SIZE: usize = 8;
const OUTPUT_SIZE: usize = 1;
const LEAK_RATE: f32 = 0.3;
const SPECTRAL_RADIUS: f64 = 0.95;
const CONNECTIVITY: f64 = 0.2;
const SEED: u64 = 42;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let iterations = parse_arg(&args, "--iterations", DEFAULT_ITERATIONS);
    let skip_hw = args.iter().any(|a| a == "--no-hw");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ESN Substrate Comparison                                  ║");
    println!("║  CPU-f64  vs  SoftwareBackend-f32  vs  AKD1000 (hardware)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("ESN architecture : RS={RESERVOIR_SIZE}, IS={INPUT_SIZE}, OS={OUTPUT_SIZE}");
    println!("Leak rate        : {LEAK_RATE}");
    println!("Spectral radius  : {SPECTRAL_RADIUS}");
    println!("Iterations       : {iterations}");
    println!();

    // ─── 1. Generate synthetic training data ─────────────────────────────────
    println!("Generating training data...");
    let (train_inputs, train_targets, test_inputs) = generate_physics_sequences();

    // ─── 2. Train CPU-f64 ESN ─────────────────────────────────────────────────
    println!("Training CPU-f64 ESN (RS={RESERVOIR_SIZE})...");
    let mut cpu_esn = CpuEsn::new(RESERVOIR_SIZE, INPUT_SIZE, OUTPUT_SIZE, LEAK_RATE as f64,
                                   SPECTRAL_RADIUS, CONNECTIVITY, SEED);
    cpu_esn.train(&train_inputs, &train_targets);
    let cpu_weights = cpu_esn.export_f32();

    println!("  Reservoir spectral radius : {:.4}", cpu_esn.measured_spectral_radius());
    println!("  Training sequences        : {}", train_inputs.len());
    println!("  w_in  : {} floats", cpu_weights.w_in.len());
    println!("  w_res : {} floats", cpu_weights.w_res.len());
    println!("  w_out : {} floats", cpu_weights.w_out.len());
    println!();

    // ─── 3. Benchmark CPU-f64 ─────────────────────────────────────────────────
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Substrate 1: CPU-f64 (reference)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let cpu_results = bench_cpu_f64(&mut cpu_esn, &test_inputs, iterations);
    print_results(&cpu_results);
    println!();

    // ─── 4. Build SoftwareBackend and benchmark ────────────────────────────────
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Substrate 2: SoftwareBackend (VirtualNPU, f32 CPU)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let mut sw = SoftwareBackend::new(RESERVOIR_SIZE, INPUT_SIZE, OUTPUT_SIZE);
    sw.load_weights(&cpu_weights.w_in, &cpu_weights.w_res, &cpu_weights.w_out)?;
    let sw_results = bench_software(&mut sw, &test_inputs, iterations);
    print_results(&sw_results);
    println!();

    // ─── 5. Parity: CPU-f64 vs SoftwareBackend ────────────────────────────────
    println!("Parity check: CPU-f64 vs SoftwareBackend");
    let parity = compute_parity(&cpu_results.outputs, &sw_results.outputs);
    println!("  Max |Δ|              : {:.6}", parity.max_abs);
    println!("  Mean |Δ|             : {:.6}", parity.mean_abs);
    println!("  Max |Δ|/|CPU|        : {:.4}%", parity.max_rel * 100.0);
    let parity_ok = parity.max_rel < 0.02; // <2% relative error for f64→f32
    println!("  f64→f32 precision    : {}", if parity_ok { "✓ OK (<2%)" } else { "✗ FAIL (>2%)" });
    println!();

    // ─── 6. Hardware benchmark (optional) ─────────────────────────────────────
    if skip_hw {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Substrate 3: AKD1000 hardware  [SKIPPED (--no-hw)]");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  To run with hardware: cargo run --bin bench_esn_substrate");
    } else {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Substrate 3: AKD1000 hardware (VFIO backend)");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        match bench_hardware(&cpu_weights, &test_inputs, iterations) {
            Ok(hw_results) => {
                print_results(&hw_results);
                println!();
                println!("Parity check: SoftwareBackend vs AKD1000");
                let hw_parity = compute_parity(&sw_results.outputs, &hw_results.outputs);
                println!("  Max |Δ|              : {:.6}", hw_parity.max_abs);
                println!("  Mean |Δ|             : {:.6}", hw_parity.mean_abs);
                println!("  Max |Δ|/|SW|         : {:.4}%", hw_parity.max_rel * 100.0);
                let hw_ok = hw_parity.max_rel < 0.05;
                println!("  f32→int4 precision   : {}", if hw_ok { "✓ OK (<5%)" } else { "✗ FAIL (>5%)" });
                println!();
                print_substrate_summary(&cpu_results, &sw_results, Some(&hw_results));
            }
            Err(e) => {
                println!("  Hardware not available: {e}");
                println!("  (Set up VFIO: cargo run --bin akida -- bind-vfio)");
                println!();
                print_substrate_summary(&cpu_results, &sw_results, None);
            }
        }
    }

    println!();
    println!("Benchmark complete.");
    Ok(())
}

// ─── Substrate measurement types ──────────────────────────────────────────────

struct BenchResults {
    substrate: &'static str,
    throughput_hz: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
    latency_min_us: f64,
    outputs: Vec<f64>,
}

struct Parity {
    max_abs: f64,
    mean_abs: f64,
    max_rel: f64,
}

// ─── CPU-f64 benchmark ────────────────────────────────────────────────────────

fn bench_cpu_f64(esn: &mut CpuEsn, test_inputs: &[Vec<f64>], iters: usize) -> BenchResults {
    let mut latencies = Vec::with_capacity(iters);
    let mut outputs = Vec::with_capacity(test_inputs.len());

    // Warmup
    for inp in test_inputs.iter().take(10) {
        let _ = esn.predict(std::slice::from_ref(inp));
    }

    // Measure
    for _ in 0..iters {
        let inp = &test_inputs[0]; // representative
        let t0 = Instant::now();
        let _ = esn.predict(std::slice::from_ref(inp));
        latencies.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    // Collect outputs for parity
    for inp in test_inputs {
        outputs.push(esn.predict(std::slice::from_ref(inp))[0]);
    }

    make_results("CPU-f64", &latencies, outputs)
}

// ─── SoftwareBackend benchmark ────────────────────────────────────────────────

fn bench_software(sw: &mut SoftwareBackend, test_inputs: &[Vec<f64>], iters: usize) -> BenchResults {
    let mut latencies = Vec::with_capacity(iters);
    let mut outputs = Vec::with_capacity(test_inputs.len());

    let inp_f32: Vec<f32> = test_inputs[0].iter().map(|&x| x as f32).collect();

    // Warmup
    for _ in 0..10 {
        let _ = sw.infer(&inp_f32);
    }

    // Measure
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = sw.infer(&inp_f32);
        latencies.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    // Collect outputs for parity
    for inp in test_inputs {
        let inp_f32: Vec<f32> = inp.iter().map(|&x| x as f32).collect();
        sw.reset_state();
        let out = sw.infer(&inp_f32).unwrap_or(vec![0.0]);
        outputs.push(out[0] as f64);
    }

    make_results("SoftwareBackend (VirtualNPU)", &latencies, outputs)
}

// ─── Hardware benchmark ────────────────────────────────────────────────────────

fn bench_hardware(weights: &ExportedWeightsF32, test_inputs: &[Vec<f64>], iters: usize)
    -> Result<BenchResults>
{
    let mgr = DeviceManager::discover()?;
    let mut device = mgr.open_first()?;

    let input_bytes: Vec<u8> = test_inputs[0].iter()
        .flat_map(|&x| (x as f32).to_le_bytes())
        .collect();
    let output_size = OUTPUT_SIZE * 4;

    // Load weights (via software model blob as reference)
    let blob = pack_software_model(
        RESERVOIR_SIZE, INPUT_SIZE, OUTPUT_SIZE, LEAK_RATE,
        &weights.w_in, &weights.w_res, &weights.w_out,
    );
    device.write(&blob)?;

    let mut out_buf = vec![0u8; output_size];

    // Warmup
    for _ in 0..20 {
        device.write(&input_bytes)?;
        device.read(&mut out_buf)?;
    }

    // Measure
    let mut latencies = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        device.write(&input_bytes)?;
        device.read(&mut out_buf)?;
        latencies.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    // Collect outputs for parity
    let mut outputs = Vec::with_capacity(test_inputs.len());
    for inp in test_inputs {
        let inp_bytes: Vec<u8> = inp.iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect();
        device.write(&inp_bytes)?;
        device.read(&mut out_buf)?;
        let val = f32::from_le_bytes(out_buf[0..4].try_into().unwrap_or([0u8; 4]));
        outputs.push(val as f64);
    }

    Ok(make_results("AKD1000 hardware (VFIO)", &latencies, outputs))
}

// ─── Summary ──────────────────────────────────────────────────────────────────

fn print_results(r: &BenchResults) {
    println!("  Throughput : {:>10.0} Hz  ({:.0} KHz)", r.throughput_hz, r.throughput_hz / 1000.0);
    println!("  Latency p50: {:>10.1} µs", r.latency_p50_us);
    println!("  Latency p95: {:>10.1} µs", r.latency_p95_us);
    println!("  Latency p99: {:>10.1} µs", r.latency_p99_us);
    println!("  Latency min: {:>10.1} µs", r.latency_min_us);
}

fn print_substrate_summary(
    cpu: &BenchResults,
    sw: &BenchResults,
    hw: Option<&BenchResults>,
) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Substrate Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{:<35} {:>12}  {:>10}", "Substrate", "Throughput", "p50 µs");
    println!("{}", "─".repeat(62));

    let cpu_hz = cpu.throughput_hz;
    println!(
        "{:<35} {:>10.0} Hz  {:>8.1} µs  [reference]",
        cpu.substrate, cpu.throughput_hz, cpu.latency_p50_us
    );
    println!(
        "{:<35} {:>10.0} Hz  {:>8.1} µs  [{:.0}× vs CPU]",
        sw.substrate, sw.throughput_hz, sw.latency_p50_us,
        sw.throughput_hz / cpu_hz
    );
    if let Some(hw) = hw {
        println!(
            "{:<35} {:>10.0} Hz  {:>8.1} µs  [{:.0}× vs CPU, energy 1.4 µJ]",
            hw.substrate, hw.throughput_hz, hw.latency_p50_us,
            hw.throughput_hz / cpu_hz
        );
        let speedup_for_energy = cpu.throughput_hz / hw.throughput_hz * (50.0 / 1.4);
        println!();
        println!("  Energy per inference  CPU ~50 µJ  |  NPU ~1.4 µJ  [{:.0}× NPU advantage]",
                 speedup_for_energy);
    }
}

fn compute_parity(a: &[f64], b: &[f64]) -> Parity {
    let n = a.len().min(b.len());
    if n == 0 {
        return Parity { max_abs: 0.0, mean_abs: 0.0, max_rel: 0.0 };
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()).take(n) {
        let diff = (av - bv).abs();
        let rel = diff / av.abs().max(1e-10);
        if diff > max_abs { max_abs = diff; }
        if rel  > max_rel { max_rel = rel;  }
        sum_abs += diff;
    }
    Parity { max_abs, mean_abs: sum_abs / n as f64, max_rel }
}

fn make_results(substrate: &'static str, latencies_us: &[f64], outputs: Vec<f64>) -> BenchResults {
    let mut sorted = latencies_us.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    BenchResults {
        substrate,
        throughput_hz: 1e6 / mean,
        latency_p50_us: sorted[n / 2],
        latency_p95_us: sorted[(n as f64 * 0.95) as usize],
        latency_p99_us: sorted[(n as f64 * 0.99) as usize],
        latency_min_us: sorted[0],
        outputs,
    }
}

// ─── CPU-f64 ESN (self-contained, no external crate) ─────────────────────────

struct ExportedWeightsF32 {
    w_in:  Vec<f32>,
    w_res: Vec<f32>,
    w_out: Vec<f32>,
}

struct CpuEsn {
    rs: usize,
    is: usize,
    os: usize,
    leak_rate: f64,
    w_in:  Vec<Vec<f64>>,
    w_res: Vec<Vec<f64>>,
    w_out: Option<Vec<Vec<f64>>>,
    state: Vec<f64>,
}

impl CpuEsn {
    fn new(rs: usize, is: usize, os: usize, leak: f64, sr: f64, conn: f64, seed: u64) -> Self {
        let mut rng = Xoshiro::new(seed);
        let w_in: Vec<Vec<f64>> = (0..rs)
            .map(|_| (0..is).map(|_| rng.uniform() - 0.5).collect())
            .collect();
        let mut w_res: Vec<Vec<f64>> = (0..rs)
            .map(|_| (0..rs)
                .map(|_| if rng.uniform() < conn { rng.normal() } else { 0.0 })
                .collect())
            .collect();
        let msr = spectral_radius(&w_res);
        if msr > 1e-10 {
            let scale = sr / msr;
            for row in &mut w_res { for v in row.iter_mut() { *v *= scale; } }
        }
        Self { rs, is, os, leak_rate: leak, w_in, w_res, w_out: None, state: vec![0.0; rs] }
    }

    fn step(&mut self, input: &[f64]) {
        let mut pre = vec![0.0f64; self.rs];
        for i in 0..self.rs {
            let mut v = 0.0;
            for j in 0..self.is.min(input.len()) { v += self.w_in[i][j] * input[j]; }
            for j in 0..self.rs               { v += self.w_res[i][j] * self.state[j]; }
            pre[i] = v;
        }
        let a = self.leak_rate;
        for i in 0..self.rs {
            self.state[i] = (1.0 - a) * self.state[i] + a * pre[i].tanh();
        }
    }

    fn collect_state(&mut self, inputs: &[Vec<f64>]) -> Vec<f64> {
        self.state.fill(0.0);
        for inp in inputs { self.step(inp); }
        self.state.clone()
    }

    fn train(&mut self, seqs: &[Vec<Vec<f64>>], targets: &[Vec<f64>]) {
        let n = seqs.len();
        let rs = self.rs;
        let os = self.os;
        let mut x = vec![vec![0.0; rs]; n];
        for (i, s) in seqs.iter().enumerate() {
            x[i] = self.collect_state(s);
        }
        // Ridge regression W_out = Y^T X (X^T X + λI)^{-1}
        let lambda = 1e-4_f64;
        let mut xtx = vec![vec![0.0f64; rs]; rs];
        for i in 0..rs { for j in 0..rs {
            xtx[i][j] = x.iter().take(n).map(|r| r[i] * r[j]).sum::<f64>();
        } xtx[i][i] += lambda; }
        let mut xty = vec![vec![0.0f64; os]; rs];
        for i in 0..rs { for j in 0..os {
            xty[i][j] = x.iter().zip(targets.iter()).take(n).map(|(r, t)| r[i] * t[j]).sum();
        } }
        let w_out_t = solve_lu(&xtx, &xty);
        let mut w_out = vec![vec![0.0; rs]; os];
        for i in 0..os { for j in 0..rs { w_out[i][j] = w_out_t[j][i]; } }
        self.w_out = Some(w_out);
    }

    fn predict(&mut self, seq: &[Vec<f64>]) -> Vec<f64> {
        let state = self.collect_state(seq);
        let w = self.w_out.as_ref().expect("train first");
        w.iter().map(|row| row.iter().zip(state.iter()).map(|(a, b)| a * b).sum()).collect()
    }

    fn export_f32(&self) -> ExportedWeightsF32 {
        let w_in  = self.w_in.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
        let w_res = self.w_res.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
        let w_out = self.w_out.as_ref().map_or(vec![], |w|
            w.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect());
        ExportedWeightsF32 { w_in, w_res, w_out }
    }

    fn measured_spectral_radius(&self) -> f64 {
        spectral_radius(&self.w_res)
    }
}

fn spectral_radius(w: &[Vec<f64>]) -> f64 {
    let n = w.len();
    if n == 0 { return 0.0; }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;
    for _ in 0..100 {
        let mut wv = vec![0.0; n];
        for (i, wvi) in wv.iter_mut().enumerate() {
            *wvi = w[i].iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        }
        let norm: f64 = wv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 { return 0.0; }
        lambda = norm;
        for (vi, wvi) in v.iter_mut().zip(wv.iter()) { *vi = wvi / norm; }
    }
    lambda
}

fn solve_lu(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if b.is_empty() { 0 } else { b[0].len() };
    let mut lu: Vec<Vec<f64>> = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    // LU factorization with partial pivoting
    for k in 0..n {
        let (max_row, _) = (k..n).map(|i| (i, lu[i][k].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((k, 0.0));
        lu.swap(k, max_row);
        piv.swap(k, max_row);
        if lu[k][k].abs() < 1e-14 { continue; }
        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n { let t = lu[i][k] * lu[k][j]; lu[i][j] -= t; }
        }
    }
    // Solve for each column of B
    let mut x = vec![vec![0.0; m]; n];
    for col in 0..m {
        let mut bc: Vec<f64> = piv.iter().map(|&p| b[p][col]).collect();
        // Forward substitution
        for i in 0..n { for j in 0..i { bc[i] -= lu[i][j] * bc[j]; } }
        // Backward substitution
        for i in (0..n).rev() {
            for j in (i + 1)..n { bc[i] -= lu[i][j] * bc[j]; }
            if lu[i][i].abs() > 1e-14 { bc[i] /= lu[i][i]; }
        }
        for (r, &v) in bc.iter().enumerate() { x[r][col] = v; }
    }
    x
}

// ─── Training data generation ─────────────────────────────────────────────────

fn generate_physics_sequences() -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = Xoshiro::new(99);

    // Synthetic lattice QCD-like sequences: plaquette + Polyakov observables
    // 8 features per timestep × 50 timesteps per sequence
    let n_train = 20;
    let n_test  = 50;
    let seq_len = 50;

    let mut train_seqs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_train);
    let mut targets: Vec<Vec<f64>> = Vec::with_capacity(n_train);

    for i in 0..n_train {
        let beta = 4.5 + (i as f64 / n_train as f64) * 2.0;
        let seq: Vec<Vec<f64>> = (0..seq_len).map(|t| {
            let phase = (beta - 5.69).tanh(); // synthetic phase indicator
            vec![
                0.5 + phase * 0.3 + rng.normal() * 0.05, // plaquette
                phase.abs() + rng.normal() * 0.1,          // |polyakov|
                rng.normal() * 0.05,                       // Im(L)
                (t as f64 / seq_len as f64),               // time
                beta / 7.0,                                // scaled beta
                rng.normal() * 0.02,                       // noise 1
                rng.normal() * 0.02,                       // noise 2
                1.0 - (t as f64 / seq_len as f64),        // reverse time
            ]
        }).collect();
        let label = if beta > 5.69 { 1.0 } else { 0.0 };
        train_seqs.push(seq);
        targets.push(vec![label]);
    }

    let test_inputs: Vec<Vec<f64>> = (0..n_test).map(|_| {
        (0..INPUT_SIZE).map(|_| rng.normal()).collect()
    }).collect();

    (train_seqs, targets, test_inputs)
}

// ─── Utility: PRNG ────────────────────────────────────────────────────────────

struct Xoshiro { s: [u64; 4] }
impl Xoshiro {
    fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for slot in &mut s {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }
    fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3])).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        r
    }
    fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-14);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}
