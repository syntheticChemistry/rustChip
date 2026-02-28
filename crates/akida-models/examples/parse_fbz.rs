//! Example: Parse Akida .fbz model file
//!
//! Demonstrates parsing of Akida model files with pure Rust.

use akida_models::prelude::*;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("akida_models=debug")
        .init();

    println!("🧠 Akida Model Parser - Pure Rust\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Parse model file
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example parse_fbz -- <path_to_model.fbz>");
        eprintln!("Example: cargo run --example parse_fbz -- /path/to/model.fbz");
        std::process::exit(1);
    });

    println!("📂 Loading model: {}\n", model_path);

    let model = Model::from_file(&model_path)?;

    println!("✅ Model loaded successfully!\n");

    // Display model info
    println!("📊 Model Information:");
    println!("   Version:      {}", model.version());
    println!("   Layers:       {}", model.layer_count());
    println!(
        "   Program size: {} bytes ({:.2} KB)\n",
        model.program_size(),
        model.program_size() as f32 / 1024.0
    );

    // Display layers
    println!("🏗️  Model Architecture:");
    println!("┌────────────────────────────────────────────────┐");

    for (i, layer) in model.layers().iter().enumerate() {
        println!(
            "│  Layer {}: {:20} {:15} │",
            i,
            layer.name,
            format!("({})", layer.layer_type)
        );
    }

    println!("└────────────────────────────────────────────────┘\n");

    // Display weight information
    println!("⚖️  Weight Data:");
    println!("   Weight blocks:  {}", model.weights().len());
    println!("   Total weights:  ~{}\n", model.total_weight_count());

    if !model.weights().is_empty() {
        for (i, weight) in model.weights().iter().enumerate() {
            println!(
                "   Block {}: {} bytes ({}-bit quantization)",
                i,
                weight.data.len(),
                weight.quantization.bits
            );
        }
        println!();
    }

    println!("✨ Parse complete! Pure Rust FlatBuffers parsing working!\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
