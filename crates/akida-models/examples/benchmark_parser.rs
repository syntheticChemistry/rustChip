//! Benchmark: Parse multiple Akida models
//!
//! Tests parser performance and accuracy across different model sizes.

use akida_models::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🏃 Akida Parser Benchmark\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test models - get from command line arguments
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: cargo run --example benchmark_parser -- <model1.fbz> [model2.fbz ...]");
        eprintln!("Example: cargo run --example benchmark_parser -- /path/to/model1.fbz /path/to/model2.fbz");
        std::process::exit(1);
    }
    let models = args;

    let mut total_time = std::time::Duration::ZERO;
    let mut total_size = 0usize;

    for model_path in &models {
        // Check if file exists
        if !std::path::Path::new(model_path).exists() {
            println!("⚠️  Skipping {} (not found)\n", model_path);
            continue;
        }

        println!("📂 Parsing: {}", model_path);

        // Measure parse time
        let start = Instant::now();
        let model = Model::from_file(model_path)?;
        let elapsed = start.elapsed();

        total_time += elapsed;
        total_size += model.program_size();

        // Display results
        println!(
            "   ⏱️  Parse time:   {:.3}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "   📊 Size:         {} bytes ({:.2} KB)",
            model.program_size(),
            model.program_size() as f64 / 1024.0
        );
        println!("   🏗️  Layers:       {}", model.layer_count());
        println!("   ⚖️  Weight blocks: {}", model.weights().len());
        println!("   📈 Total weights: ~{}", model.total_weight_count());

        // Show layers
        if model.layer_count() <= 5 {
            println!("   Layers:");
            for (i, layer) in model.layers().iter().enumerate() {
                println!("      {}. {} ({})", i + 1, layer.name, layer.layer_type);
            }
        }

        println!();
    }

    // Summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Benchmark Summary:");
    println!("   Total time:  {:.3}ms", total_time.as_secs_f64() * 1000.0);
    println!(
        "   Total size:  {} bytes ({:.2} KB)",
        total_size,
        total_size as f64 / 1024.0
    );
    println!(
        "   Avg speed:   {:.2} MB/s",
        (total_size as f64 / 1024.0 / 1024.0) / total_time.as_secs_f64()
    );
    println!("\n✨ Benchmark complete!\n");

    Ok(())
}
