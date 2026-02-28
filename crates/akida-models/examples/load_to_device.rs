//! Example: Load model to Akida device
//!
//! Demonstrates the complete workflow of loading a parsed model to hardware.

use akida_driver::DeviceManager;
use akida_models::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("🧠 Akida Model Loading Demo\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Get model path from args
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example load_to_device -- <path_to_model.fbz>");
        eprintln!("Example: cargo run --example load_to_device -- /path/to/model.fbz");
        std::process::exit(1);
    });

    println!("📂 Loading model: {}\n", model_path);

    // Step 1: Parse model
    println!("1️⃣  Parsing model...");
    let model = Model::from_file(&model_path)?;
    println!(
        "   ✅ Parsed: {} layers, {} bytes\n",
        model.layer_count(),
        model.program_size()
    );

    // Step 2: Discover devices
    println!("2️⃣  Discovering Akida devices...");
    let manager = DeviceManager::discover()?;
    println!("   ✅ Found {} device(s)\n", manager.device_count());

    if manager.device_count() == 0 {
        println!("❌ No Akida devices found!");
        println!("   Make sure:");
        println!("   - Akida PCIe cards are installed");
        println!("   - Driver is loaded (lsmod | grep akida)");
        println!("   - Devices accessible (/dev/akida*)");
        return Ok(());
    }

    // Step 3: Select device
    let mut device = manager.open_first()?;
    println!("3️⃣  Selected device {}:", device.index());

    // Clone capabilities before mutable borrow
    let caps = device.info().capabilities().clone();
    println!("   Chip:   {:?}", caps.chip_version);
    println!("   NPUs:   {}", caps.npu_count);
    println!("   Memory: {} MB", caps.memory_mb);
    println!(
        "   PCIe:   Gen{} x{}\n",
        caps.pcie.generation, caps.pcie.lanes
    );

    // Step 4: Load model to device
    println!("4️⃣  Loading model to device...");
    let metrics = model.load_to_device(&mut device)?;

    // Step 5: Display results
    println!("\n✅ MODEL LOADED SUCCESSFULLY!\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📊 Load Metrics:");
    println!(
        "   Bytes transferred: {} bytes ({:.2} KB)",
        metrics.bytes_transferred,
        metrics.bytes_transferred as f64 / 1024.0
    );
    println!("   Chunks:           {}", metrics.chunks_transferred);
    println!("   Duration:         {:?}", metrics.duration);
    println!("   Throughput:       {:.2} MB/s", metrics.throughput_mbps);
    println!();

    // Compare with model size
    let efficiency = (metrics.bytes_transferred as f64 / model.program_size() as f64) * 100.0;
    println!("📈 Transfer Efficiency: {:.1}%", efficiency);

    // Device utilization
    let memory_used_pct =
        (metrics.bytes_transferred as f64 / (caps.memory_mb as f64 * 1024.0 * 1024.0)) * 100.0;
    println!("💾 Device Memory Used: {:.2}%\n", memory_used_pct);

    println!(
        "🎉 Demo complete! Model is now loaded on device {}",
        device.index()
    );
    println!("   Ready for inference (Phase 4).\n");

    Ok(())
}
