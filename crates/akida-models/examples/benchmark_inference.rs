//! Benchmark: Inference performance
//!
//! Comprehensive inference benchmarking suite.

use akida_driver::DeviceManager;
use akida_models::Model;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (quiet for benchmarking)
    tracing_subscriber::fmt().with_env_filter("warn").init();

    println!("🏃 Akida Inference Benchmark\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Get model path
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example benchmark_inference -- <path_to_model.fbz>");
        eprintln!("Example: cargo run --example benchmark_inference -- /path/to/model.fbz");
        std::process::exit(1);
    });

    println!("📂 Model: {}\n", model_path);

    // Parse model
    println!("1️⃣  Parsing model...");
    let model = Model::from_file(&model_path)?;
    println!(
        "   ✅ {} layers, {} bytes\n",
        model.layer_count(),
        model.program_size()
    );

    // Discover device
    println!("2️⃣  Discovering device...");
    let manager = DeviceManager::discover()?;
    if manager.device_count() == 0 {
        println!("❌ No devices found!");
        return Ok(());
    }
    println!("   ✅ Found {} device(s)\n", manager.device_count());

    // Load model
    println!("3️⃣  Loading model...");
    let mut device = manager.open_first()?;
    let load_metrics = model.load_to_device(&mut device)?;
    println!("   ✅ Loaded in {:?}\n", load_metrics.duration);

    // Prepare input
    let input = vec![0u8; model.input_size()];

    // Warmup (10 iterations)
    println!("4️⃣  Warming up (10 iterations)...");
    for _ in 0..10 {
        let _ = model.infer(&input, &mut device)?;
    }
    println!("   ✅ Warmup complete\n");

    // Benchmark: Single inference latency
    println!("5️⃣  Measuring latency (100 iterations)...");
    let mut latencies = Vec::with_capacity(100);

    for i in 0..100 {
        let start = Instant::now();
        let _ = model.infer(&input, &mut device)?;
        let elapsed = start.elapsed();
        latencies.push(elapsed);

        if i == 0 || i == 99 {
            println!("   Run {}: {:?}", i + 1, elapsed);
        } else if i == 1 {
            println!("   ...");
        }
    }

    // Calculate statistics
    let min_latency = latencies.iter().min().unwrap();
    let max_latency = latencies.iter().max().unwrap();
    let avg_latency = latencies.iter().sum::<std::time::Duration>() / latencies.len() as u32;
    let std_dev = calculate_std_dev(&latencies, avg_latency);

    println!(
        "\n   Min:     {:?} ({:.1}µs)",
        min_latency,
        min_latency.as_secs_f64() * 1_000_000.0
    );
    println!(
        "   Avg:     {:?} ({:.1}µs)",
        avg_latency,
        avg_latency.as_secs_f64() * 1_000_000.0
    );
    println!(
        "   Max:     {:?} ({:.1}µs)",
        max_latency,
        max_latency.as_secs_f64() * 1_000_000.0
    );
    println!("   Std Dev: {:.1}µs\n", std_dev * 1_000_000.0);

    // Benchmark: Throughput (1 second burst)
    println!("6️⃣  Measuring throughput (1 second burst)...");
    let burst_start = Instant::now();
    let mut burst_count = 0;

    while burst_start.elapsed().as_secs() < 1 {
        let _ = model.infer(&input, &mut device)?;
        burst_count += 1;
    }

    let burst_duration = burst_start.elapsed();
    let throughput = burst_count as f64 / burst_duration.as_secs_f64();

    println!("   Inferences: {}", burst_count);
    println!("   Duration:   {:?}", burst_duration);
    println!("   Throughput: {:.1} inferences/sec\n", throughput);

    // Results summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📊 BENCHMARK RESULTS\n");

    println!("Model:");
    println!(
        "   Size:    {} bytes ({:.2} KB)",
        model.program_size(),
        model.program_size() as f64 / 1024.0
    );
    println!("   Layers:  {}", model.layer_count());
    println!("   Weights: {} blocks\n", model.weights().len());

    println!("Loading:");
    println!("   Time:       {:?}", load_metrics.duration);
    println!("   Throughput: {:.2} MB/s\n", load_metrics.throughput_mbps);

    println!("Inference Latency (N=100):");
    println!(
        "   Min:     {:.1}µs",
        min_latency.as_secs_f64() * 1_000_000.0
    );
    println!(
        "   Avg:     {:.1}µs",
        avg_latency.as_secs_f64() * 1_000_000.0
    );
    println!(
        "   Max:     {:.1}µs",
        max_latency.as_secs_f64() * 1_000_000.0
    );
    println!("   Std Dev: {:.1}µs", std_dev * 1_000_000.0);
    println!(
        "   Variance: {:.1}%\n",
        (std_dev / avg_latency.as_secs_f64()) * 100.0
    );

    println!("Inference Throughput:");
    println!("   Rate:    {:.1} inferences/sec", throughput);
    println!(
        "   Period:  {:.1}µs per inference\n",
        1_000_000.0 / throughput
    );

    // Performance grade
    let grade = if avg_latency.as_micros() < 100 {
        "A+ (Excellent)"
    } else if avg_latency.as_micros() < 500 {
        "A (Very Good)"
    } else if avg_latency.as_micros() < 1000 {
        "B (Good)"
    } else {
        "C (Acceptable)"
    };

    println!("Performance Grade: {}\n", grade);

    println!("🎉 Benchmark complete!\n");

    Ok(())
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[std::time::Duration], mean: std::time::Duration) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean_secs = mean.as_secs_f64();
    let variance: f64 = values
        .iter()
        .map(|v| {
            let diff = v.as_secs_f64() - mean_secs;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}
