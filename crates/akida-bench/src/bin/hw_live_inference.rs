// SPDX-License-Identifier: AGPL-3.0-or-later

//! Live hardware inference — end-to-end VFIO path.
//!
//! Initializes the AKD1000 via VFIO (including reset/enable), loads a real
//! `.fbz` model, runs inference with synthetic input, and prints results.
//! This is the first full userspace inference without any kernel module.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin hw_live_inference
//! cargo run --bin hw_live_inference -- 0000:e2:00.0 ds_cnn_kws.fbz
//! ```

use akida_driver::{NpuBackend, VfioBackend};
use akida_models::Model;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const DEFAULT_MODEL: &str = "ds_cnn_kws.fbz";
const ZOO_DIR: &str = "baseCamp/zoo-artifacts";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("akida_driver=info".parse()?)
                .add_directive("warn".parse()?),
        )
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Akida Live Hardware Inference — Pure Userspace VFIO");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let args: Vec<String> = std::env::args().collect();
    let pcie_addr = args
        .get(1)
        .cloned()
        .or_else(discover_first_akida)
        .ok_or("No PCIe address given and no Akida device found")?;

    let model_name = args.get(2).map_or(DEFAULT_MODEL, |s| s.as_str());
    let model_path = format!("{ZOO_DIR}/{model_name}");

    // ── Parse the .fbz model ────────────────────────────────────────────
    println!("── Model Parsing ──────────────────────────────────────────────");
    let parse_start = Instant::now();
    let model = Model::from_file(&model_path)?;
    let parse_time = parse_start.elapsed();

    let file_size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    let raw_bytes: Vec<u8> = std::fs::read(&model_path)?;

    println!("  Model          : {model_name}");
    println!("  File size      : {file_size} bytes ({:.1} KB)", file_size as f64 / 1024.0);
    println!("  Layers         : {}", model.layer_count());
    println!("  Weight blocks  : {}", model.weights().len());
    println!("  Total weights  : ~{}", model.total_weight_count());
    println!("  Parse time     : {parse_time:?}");
    println!();

    // ── Initialize VFIO backend ─────────────────────────────────────────
    println!("── VFIO Backend Init ─────────────────────────────────────────");
    let init_start = Instant::now();
    let mut backend = VfioBackend::init(&pcie_addr)?;
    let init_time = init_start.elapsed();

    let caps = backend.capabilities();
    println!("  PCIe address   : {pcie_addr}");
    println!("  Init time      : {init_time:?}");
    println!("  Chip           : {:?}", caps.chip_version);
    println!("  NPUs           : {}", caps.npu_count);
    println!("  SRAM           : {} MB", caps.memory_mb);
    println!("  PCIe           : Gen{} x{}", caps.pcie.generation, caps.pcie.lanes);
    println!("  Ready          : {}", backend.is_ready());
    println!("  Backend        : {}", backend.backend_type());
    println!();

    if !backend.is_ready() {
        println!("── Manual Init Retry ─────────────────────────────────────────");
        println!("  Device not ready after automatic init, trying explicit reset...");
        match backend.reset_and_enable() {
            Ok(()) => println!("  Device is now READY"),
            Err(e) => {
                println!("  Reset failed: {e}");
                println!();
                println!("  The device did not reach READY state.");
                println!("  This may indicate the register offsets need empirical tuning,");
                println!("  or the chip requires a specific power-on sequence.");
                println!();
                println!("  Proceeding with model load attempt anyway...");
            }
        }
        println!();
    }

    // ── Load model via DMA ──────────────────────────────────────────────
    println!("── Model Load (DMA) ──────────────────────────────────────────");
    let load_start = Instant::now();
    match backend.load_model(&raw_bytes) {
        Ok(handle) => {
            let load_time = load_start.elapsed();
            println!("  Model loaded   : handle={}", handle.id());
            println!("  DMA transfer   : {} bytes in {load_time:?}", raw_bytes.len());
            println!(
                "  Throughput     : {:.1} MB/s",
                raw_bytes.len() as f64 / load_time.as_secs_f64() / 1_048_576.0
            );
            println!();

            // ── Verify load via SRAM readback ───────────────────────────
            println!("── Load Verification ─────────────────────────────────────────");
            match backend.verify_load(&raw_bytes) {
                Ok(v) if v.supported && v.verified => {
                    println!("  SRAM readback  : {} bytes verified  PASS", v.bytes_checked);
                }
                Ok(v) if v.supported => {
                    println!(
                        "  SRAM readback  : {}/{} bytes matched  MISMATCH",
                        v.bytes_matched, v.bytes_checked
                    );
                }
                Ok(_) => println!("  SRAM readback  : not supported on this path"),
                Err(e) => println!("  SRAM readback  : {e}"),
            }
            println!();

            // ── Run inference ───────────────────────────────────────────
            println!("── Live Inference ────────────────────────────────────────────");
            let input_size = 490; // DS-CNN KWS: 49 frames × 10 MFCC features
            let mut input = vec![0.0f32; input_size];
            let mut rng = 0xDEAD_BEEFu64;
            for x in &mut input {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *x = ((rng >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            }

            let infer_start = Instant::now();
            match backend.infer(&input) {
                Ok(output) => {
                    let infer_time = infer_start.elapsed();
                    println!("  Input size     : {input_size} floats");
                    println!("  Output size    : {} floats", output.len());
                    println!("  Inference time : {infer_time:?}");
                    println!("  Throughput     : {:.0} inferences/sec", 1.0 / infer_time.as_secs_f64());

                    if !output.is_empty() {
                        let max_idx = output
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        println!("  Top class      : {max_idx} (logit={:.4})", output[max_idx]);
                        println!("  Output sample  : {:?}", &output[..output.len().min(8)]);
                    }

                    // ── Latency benchmark ───────────────────────────────
                    println!();
                    println!("── Latency Benchmark (100 iterations) ──────────────────────");
                    let n = 100;
                    let bench_start = Instant::now();
                    for _ in 0..n {
                        let _ = backend.infer(&input)?;
                    }
                    let bench_total = bench_start.elapsed();
                    let avg_us = bench_total.as_micros() as f64 / n as f64;
                    println!("  Total time     : {bench_total:?}");
                    println!("  Avg latency    : {avg_us:.1} µs");
                    println!("  Throughput     : {:.0} inferences/sec", 1e6 / avg_us);
                }
                Err(e) => {
                    println!("  Inference error: {e}");
                    println!();
                    println!("  If 'Device not ready', the init sequence may need tuning.");
                    println!("  The register map is empirically derived; the exact reset");
                    println!("  protocol for AKD1000 over VFIO may differ from kernel path.");
                }
            }
        }
        Err(e) => {
            println!("  Model load failed: {e}");
            println!();
            println!("  This is expected if the device did not reach READY state.");
        }
    }

    // ── Power measurement ───────────────────────────────────────────────
    println!();
    println!("── Power Measurement ────────────────────────────────────────");
    match backend.measure_power() {
        Ok(w) => println!("  Power draw     : {w:.2} W"),
        Err(e) => println!("  Power          : unavailable ({e})"),
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Live hardware inference complete");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn discover_first_akida() -> Option<String> {
    let mgr = akida_driver::DeviceManager::discover().ok()?;
    mgr.devices().first().map(|d| d.pcie_address.clone())
}
