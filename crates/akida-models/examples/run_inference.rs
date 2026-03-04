// SPDX-License-Identifier: AGPL-3.0-or-later

//! Example: Run inference on Akida device
//!
//! Demonstrates the complete inference workflow.

use akida_driver::DeviceManager;
use akida_models::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("🧠 Akida Inference Demo\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Get model path
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example run_inference -- <path_to_model.fbz>");
        eprintln!("Example: cargo run --example run_inference -- /path/to/model.fbz");
        std::process::exit(1);
    });

    println!("📂 Model: {}\n", model_path);

    // Step 1: Parse model
    println!("1️⃣  Parsing model...");
    let model = Model::from_file(&model_path)?;
    println!("   ✅ Parsed: {} layers\n", model.layer_count());

    // Step 2: Discover and open device
    println!("2️⃣  Discovering devices...");
    let manager = DeviceManager::discover()?;

    if manager.device_count() == 0 {
        println!("❌ No devices found!");
        return Ok(());
    }

    let mut device = manager.open_first()?;
    println!("   ✅ Opened device {}\n", device.index());

    // Step 3: Load model to device
    println!("3️⃣  Loading model to device...");
    let load_metrics = model.load_to_device(&mut device)?;
    println!("   ✅ Loaded in {:?}\n", load_metrics.duration);

    // Step 4: Prepare input
    println!("4️⃣  Preparing input data...");
    let input_size = model.input_size();
    println!("   Input size: {} bytes", input_size);

    // Create dummy input (zeros for demo)
    let input = vec![0u8; input_size];
    println!("   ✅ Input prepared\n");

    // Step 5: Run inference
    println!("5️⃣  Running inference...");
    let result = model.infer(&input, &mut device)?;

    // Step 6: Display results
    println!("\n✅ INFERENCE COMPLETE!\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📊 Inference Metrics:");
    println!("   Input transfer:  {:?}", result.input_transfer_duration);
    println!("   Output transfer: {:?}", result.output_transfer_duration);
    println!("   Total time:      {:?}", result.total_duration);
    println!("   Latency:         {:.1}µs", result.latency_us());
    println!(
        "   Throughput:      {:.1} inferences/sec",
        result.throughput_ips()
    );
    println!();

    println!("📦 Data Transfer:");
    println!("   Input bytes:  {}", result.input_bytes);
    println!("   Output bytes: {}", result.output_bytes);
    println!("   Output size:  {} elements\n", result.output.len());

    // Show first few output values
    println!("🎯 Output Preview (first 10 bytes):");
    print!("   ");
    for (i, &val) in result.output.iter().take(10).enumerate() {
        print!("{:3}", val);
        if i < 9 && i < result.output.len() - 1 {
            print!(" ");
        }
    }
    println!("\n");

    println!(
        "🎉 Demo complete! Model inference on device {} successful.",
        device.index()
    );
    println!("   Ready for production use!\n");

    Ok(())
}
