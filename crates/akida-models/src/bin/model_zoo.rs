//! Akida Model Zoo CLI
//!
//! Manage Akida Model Zoo models - list, download, and validate.
//!
//! ## Usage
//!
//! ```bash
//! # List available models
//! model_zoo --list
//!
//! # Show zoo status
//! model_zoo --status
//!
//! # Create stub models for NeuroBench testing
//! model_zoo --init-stubs
//!
//! # Create specific stub model
//! model_zoo --create-stub ds_cnn_kws
//! ```

use akida_models::zoo::{ModelZoo, ZooModel};
use std::env;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber");

    if let Err(e) = run() {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> akida_models::Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut cache_dir = "models/akida".to_string();
    let mut show_list = false;
    let mut show_status = false;
    let mut init_stubs = false;
    let mut create_stub: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cache-dir" | "-c" => {
                i += 1;
                if i < args.len() {
                    cache_dir = args[i].clone();
                }
            }
            "--list" | "-l" => {
                show_list = true;
            }
            "--status" | "-s" => {
                show_status = true;
            }
            "--init-stubs" | "-i" => {
                init_stubs = true;
            }
            "--create-stub" => {
                i += 1;
                if i < args.len() {
                    create_stub = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            arg if !arg.starts_with('-') => {
                // Positional argument - treat as model name for stub creation
                create_stub = Some(arg.to_string());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Default to --status if no action specified
    if !show_list && !show_status && !init_stubs && create_stub.is_none() {
        show_status = true;
    }

    // Initialize model zoo
    info!("Initializing model zoo at: {}", cache_dir);
    let mut zoo = ModelZoo::new(&cache_dir)?;

    // Execute requested actions
    if show_list {
        list_models();
    }

    if show_status {
        zoo.print_status();
    }

    if init_stubs {
        info!("Creating NeuroBench stub models...");
        let paths = zoo.init_neurobench_stubs()?;
        println!("\nCreated {} stub models:", paths.len());
        for path in &paths {
            println!("  ✓ {}", path.display());
        }
    }

    if let Some(model_name) = create_stub {
        let model = parse_model_name(&model_name)?;
        info!("Creating stub for {:?}...", model);
        let path = zoo.create_stub_model(model)?;
        println!("\n✓ Created stub: {}", path.display());
    }

    Ok(())
}

fn parse_model_name(name: &str) -> akida_models::Result<ZooModel> {
    let name_lower = name.to_lowercase();
    let name_clean = name_lower.trim_end_matches(".fbz").replace('-', "_");

    match name_clean.as_str() {
        "akidanet_05_160" | "akidanet05" | "akidanet_05" => Ok(ZooModel::AkidaNet05_160),
        "akidanet_10_224" | "akidanet10" | "akidanet_10" | "akidanet" => {
            Ok(ZooModel::AkidaNet10_224)
        }
        "ds_cnn_kws" | "dscnn" | "kws" => Ok(ZooModel::DsCnnKws),
        "mobilenetv2" | "mobilenet" => Ok(ZooModel::MobileNetV2),
        "vit_tiny" | "vit" => Ok(ZooModel::ViTTiny),
        "yolov8n" | "yolo" => Ok(ZooModel::YoloV8n),
        "pointnet_plus" | "pointnet" | "pointnet++" => Ok(ZooModel::PointNetPlusPlus),
        "dvs_gesture" | "dvsgesture" | "gesture" => Ok(ZooModel::DvsGesture),
        "event_camera" | "eventcamera" => Ok(ZooModel::EventCamera),
        "esn_chaotic" | "esn" | "chaotic" => Ok(ZooModel::EsnChaotic),
        _ => Err(akida_models::AkidaModelError::loading_failed(format!(
            "Unknown model: {}. Use --list to see available models.",
            name
        ))),
    }
}

fn list_models() {
    println!("\nAvailable Akida Model Zoo Models:");
    println!("{}", "=".repeat(70));
    println!("{:<25} {:<15} {:<10} Description", "Model", "Task", "Size");
    println!("{}", "-".repeat(70));

    for model in ZooModel::all() {
        let task = format!("{:?}", model.task());
        let size = format_size(model.expected_size_bytes());

        println!(
            "{:<25} {:<15} {:<10} {}",
            model.filename(),
            task,
            size,
            model.description()
        );
    }

    println!("{}", "=".repeat(70));
    println!("\nNeuroBench models:");
    for model in ZooModel::neurobench_models() {
        println!("  - {}", model.filename());
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn print_help() {
    println!(
        r#"
Akida Model Zoo CLI

USAGE:
    model_zoo [OPTIONS] [MODEL_NAME]

OPTIONS:
    -c, --cache-dir <DIR>   Model cache directory (default: models/akida)
    -l, --list              List all available models
    -s, --status            Show zoo status (default action)
    -i, --init-stubs        Create stub models for NeuroBench testing
    --create-stub <MODEL>   Create a specific stub model
    -h, --help              Show this help message

MODELS:
    akidanet_05_160    AkidaNet ImageNet (0.5, 160×160)
    akidanet_10_224    AkidaNet ImageNet (1.0, 224×224)
    ds_cnn_kws         DS-CNN Keyword Spotting
    mobilenetv2        MobileNetV2 ImageNet
    vit_tiny           Vision Transformer (tiny)
    yolov8n            YOLOv8 nano
    pointnet_plus      PointNet++ 3D
    dvs_gesture        DVS Gesture recognition
    event_camera       Event camera detection
    esn_chaotic        ESN chaotic prediction

EXAMPLES:
    # Show zoo status
    model_zoo --status

    # Create all NeuroBench stubs
    model_zoo --init-stubs

    # Create specific stub
    model_zoo --create-stub ds_cnn_kws
"#
    );
}
