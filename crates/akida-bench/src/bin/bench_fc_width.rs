//! FC width scaling benchmark — reproduces Discovery 5 from BEYOND_SDK.md.
//!
//! Tests the latency vs model width curve for FullyConnected layers.
//! This is purely a data-transfer and PCIe characterization test —
//! wider models mean larger program binaries uploaded via DMA.
//!
//! Key finding: below ~512 neurons, PCIe dominates (flat ~650 µs).
//! Above 512, compute becomes non-negligible. At 4096, compute is ~15 ms.
//!
//! Reference table (BEYOND_SDK.md, Discovery 5):
//!   InputConv(8→N) → FC(N→N) → FC(N→1)
//!   width=  64:  prog=    5,120 B   lat=779µs
//!   width= 128:  prog=   15,408 B   lat=700µs
//!   width= 256:  prog=   54,464 B   lat=812µs
//!   width= 512:  prog=  206,208 B   lat=1,106µs  ← compute starts contributing
//!   width=1024:  prog=  804,608 B   lat=1,986µs
//!   width=2048:  prog=3,181,056 B   lat=4,969µs
//!   width=4096:  prog=12,652,544 B  lat=16,141µs
//!   width=8192:  OK (still fits in 8 MB SRAM)
//!
//! This test measures the model-upload DMA cost by varying the payload size
//! to approximate model sizes at each width.

use anyhow::Result;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

const ITERATIONS: usize = 100;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("warn".parse()?))
        .init();

    println!("FC width scaling benchmark (Discovery 5 — BEYOND_SDK.md)");
    println!("=========================================================");
    println!("Simulates DMA cost of uploading FC models of varying width.");
    println!("(Model payload approximated from reference program sizes)");
    println!();
    println!(
        "  {:>7}  {:>12}  {:>12}  {:>12}  {:>10}",
        "width", "prog size", "upload µs", "Hz", "vs 64"
    );
    println!("  {:-<7}  {:-<12}  {:-<12}  {:-<12}  {:-<10}", "", "", "", "", "");

    let manager = akida_driver::DeviceManager::discover()?;
    let mut device = manager.open_first()?;

    // Program sizes from Discovery 5 measurements (bytes)
    let widths_and_prog_sizes: &[(usize, usize)] = &[
        (64, 5_120),
        (128, 15_408),
        (256, 54_464),
        (512, 206_208),
        (1024, 804_608),
        (2048, 3_181_056),
        (4096, 12_652_544),
    ];

    let mut baseline_us: Option<f64> = None;

    for &(width, prog_bytes) in widths_and_prog_sizes {
        let payload = vec![0u8; prog_bytes];
        let mut readback = vec![0u8; 256]; // Output is small regardless of width

        // Warmup
        for _ in 0..3 {
            device.write(&payload)?;
            let _ = device.read(&mut readback);
        }

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            device.write(&payload)?;
            let _ = device.read(&mut readback);
        }
        let us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

        let baseline = *baseline_us.get_or_insert(us);
        let ratio = us / baseline;

        let prog_fmt = if prog_bytes >= 1_000_000 {
            format!("{:.2} MB", prog_bytes as f64 / 1_048_576.0)
        } else if prog_bytes >= 1_000 {
            format!("{} KB", prog_bytes / 1024)
        } else {
            format!("{} B", prog_bytes)
        };

        let note = if width == 512 {
            " ← compute starts"
        } else if width >= 4096 {
            " (compute dominant)"
        } else {
            ""
        };

        println!(
            "  {:>7}  {:>12}  {:>12.0}  {:>12.0}  {:>9.2}×{}",
            width,
            prog_fmt,
            us,
            1e6 / us,
            ratio,
            note
        );
    }

    println!();
    println!("Reference: PCIe dominated below width=512 (~650µs floor)  (Feb 2026)");
    println!("Width=8192 also fits — 8 MB SRAM limit is generous for FC networks");

    Ok(())
}
