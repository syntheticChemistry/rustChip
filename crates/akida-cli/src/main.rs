// SPDX-License-Identifier: AGPL-3.0-or-later

//! `akida` — command-line interface for `BrainChip` Akida hardware and models.
//!
//! ```text
//! USAGE:
//!   akida enumerate                  List all devices and capabilities
//!   akida info <pcie-addr>           Detailed info for one device
//!   akida bind-vfio <pcie-addr>      Bind device to vfio-pci (root)
//!   akida unbind-vfio <pcie-addr>    Unbind from vfio-pci (root)
//!   akida parse <file.fbz>           Parse and inspect an Akida model file
//!   akida zoo-status [dir]           Show model zoo cache status
//!   akida guidestone [dir]           Run guideStone self-leveling validation
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "akida", about = "BrainChip Akida hardware CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all Akida devices and their capabilities.
    Enumerate,
    /// Print detailed information for one device.
    Info {
        /// `PCIe` address (e.g. 0000:a1:00.0) or device index (e.g. 0).
        device: String,
    },
    /// Bind a device to vfio-pci (requires root / `CAP_SYS_ADMIN`).
    BindVfio {
        /// `PCIe` address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
    /// Unbind a device from vfio-pci and re-bind to `akida_pcie` (if loaded).
    UnbindVfio {
        /// `PCIe` address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
    /// Query the IOMMU group for a device.
    IommuGroup {
        /// `PCIe` address (e.g. 0000:a1:00.0).
        pcie_addr: String,
    },
    /// Run full NPU setup: discover hardware, load module, enable PCIe, set permissions.
    Setup,
    /// Verify that Akida hardware is accessible and ready.
    Verify,
    /// Parse and inspect an Akida .fbz model file.
    Parse {
        /// Path to .fbz model file.
        file: String,
    },
    /// Show model zoo cache status.
    ZooStatus {
        /// Zoo artifacts directory (default: baseCamp/zoo-artifacts/).
        #[arg(default_value = "baseCamp/zoo-artifacts")]
        dir: String,
    },
    /// Run guideStone: self-leveling validation and benchmark of the model zoo.
    ///
    /// Parses all zoo + physics models, validates structure, computes SHA-256
    /// digests, benchmarks throughput, and emits a graded report.
    Guidestone {
        /// Zoo artifacts directory (default: baseCamp/zoo-artifacts/).
        #[arg(default_value = "baseCamp/zoo-artifacts")]
        dir: String,
    },
    /// Convert weights to an Akida .fbz model file (pure Rust, no Python).
    ///
    /// Reads float weights from .npy, .safetensors, or inline specification,
    /// quantizes to int1/2/4/8, serializes via FlatBuffer, and writes .fbz.
    Convert {
        /// Weight source: path to .npy or .safetensors file, or "zeros:N" for N zero weights.
        #[arg(long)]
        weights: String,
        /// Architecture spec: layer descriptions separated by spaces.
        /// e.g. "InputConv(50,1,1) FC(128) FC(1)"
        #[arg(long)]
        arch: String,
        /// Output .fbz file path.
        #[arg(long, short)]
        output: String,
        /// Weight bit width (1, 2, 4, or 8). Default: 4.
        #[arg(long, default_value = "4")]
        bits: u8,
    },
    /// Import an ONNX model: parse graph, extract weights, report Akida compatibility.
    ImportOnnx {
        /// Path to .onnx model file.
        #[arg(long)]
        weights: String,
        /// Optional: output .fbz file path. If given, quantizes and serializes.
        #[arg(long, short)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Cmd::Enumerate => cmd_enumerate()?,
        Cmd::Info { device } => cmd_info(&device)?,
        Cmd::BindVfio { pcie_addr } => cmd_bind_vfio(&pcie_addr)?,
        Cmd::UnbindVfio { pcie_addr } => cmd_unbind_vfio(&pcie_addr)?,
        Cmd::IommuGroup { pcie_addr } => cmd_iommu_group(&pcie_addr)?,
        Cmd::Setup => cmd_setup()?,
        Cmd::Verify => cmd_verify()?,
        Cmd::Parse { file } => cmd_parse(&file)?,
        Cmd::ZooStatus { dir } => cmd_zoo_status(&dir)?,
        Cmd::Guidestone { dir } => cmd_guidestone(&dir)?,
        Cmd::Convert {
            weights,
            arch,
            output,
            bits,
        } => cmd_convert(&weights, &arch, &output, bits)?,
        Cmd::ImportOnnx { weights, output } => cmd_import_onnx(&weights, output.as_deref())?,
    }

    Ok(())
}

fn cmd_enumerate() -> Result<()> {
    let mgr = akida_driver::DeviceManager::discover()?;

    println!("Akida devices: {}", mgr.device_count());
    println!();

    for info in mgr.devices() {
        let c = info.capabilities();
        let variant = match c.chip_version {
            akida_driver::ChipVersion::Akd1000 => "AKD1000",
            akida_driver::ChipVersion::Akd1500 => "AKD1500",
            akida_driver::ChipVersion::Unknown(_) => "Unknown",
        };

        println!("[{}] {} @ {}", info.index(), variant, info.pcie_address());
        println!(
            "     PCIe  Gen{} x{}  ({:.1} GB/s theoretical)",
            c.pcie.generation, c.pcie.lanes, c.pcie.bandwidth_gbps
        );
        println!("     NPUs  {}   SRAM  {} MB", c.npu_count, c.memory_mb);

        if let Some(m) = &c.mesh {
            println!(
                "     Mesh  {}×{}×{}  ({} functional)",
                m.x, m.y, m.z, m.functional_count
            );
        }
        if let Some(clock) = c.clock_mode {
            println!("     Clock {clock:?}");
        }
        if let Some(batch) = &c.batch {
            println!(
                "     Batch optimal={}  {:.1}× speedup",
                batch.optimal_batch, batch.optimal_speedup
            );
        }
        if let Some(pw) = c.power_mw {
            println!("     Power {pw} mW");
        }
        println!("     WeightMut {:?}", c.weight_mutation);
        println!();
    }

    Ok(())
}

fn cmd_info(device: &str) -> Result<()> {
    let mgr = akida_driver::DeviceManager::discover()?;

    // Accept index or PCIe address
    let info = if let Ok(idx) = device.parse::<usize>() {
        mgr.device(idx)?.clone()
    } else {
        mgr.devices()
            .iter()
            .find(|d| d.pcie_address() == device)
            .ok_or_else(|| anyhow::anyhow!("Device not found: {device}"))?
            .clone()
    };

    let c = info.capabilities();
    println!("Device       : {}", info.path().display());
    println!("PCIe address : {}", info.pcie_address());
    println!("Chip version : {:?}", c.chip_version);
    println!(
        "PCIe link    : Gen{} x{} ({:.1} GB/s)",
        c.pcie.generation, c.pcie.lanes, c.pcie.bandwidth_gbps
    );
    println!("NPUs         : {}", c.npu_count);
    println!("SRAM         : {} MB", c.memory_mb);

    if let Some(m) = &c.mesh {
        println!(
            "NP mesh      : {}×{}×{} ({} functional, {} disabled)",
            m.x,
            m.y,
            m.z,
            m.functional_count,
            (u32::from(m.x) * u32::from(m.y) * u32::from(m.z)).saturating_sub(m.functional_count)
        );
    }
    if let Some(clock) = c.clock_mode {
        println!("Clock mode   : {clock:?}");
    }
    if let Some(batch) = &c.batch {
        println!(
            "Batch        : max={} optimal={} ({:.1}× speedup)",
            batch.max_batch, batch.optimal_batch, batch.optimal_speedup
        );
    }
    if let Some(pw) = c.power_mw {
        println!("Power        : {pw} mW");
    }
    if let Some(t) = c.temperature_c {
        println!("Temperature  : {t:.1} °C");
    }
    println!("WeightMut    : {:?}", c.weight_mutation);

    // IOMMU group (useful for VFIO setup)
    match akida_driver::vfio::iommu_group(info.pcie_address()) {
        Ok(g) => println!("IOMMU group  : {g}"),
        Err(_) => println!("IOMMU group  : (not available — IOMMU disabled?)"),
    }

    Ok(())
}

fn cmd_bind_vfio(pcie_addr: &str) -> Result<()> {
    println!("Binding {pcie_addr} to vfio-pci ...");
    akida_driver::vfio::bind_to_vfio(pcie_addr)?;
    println!(
        "Done. IOMMU group: {}",
        akida_driver::vfio::iommu_group(pcie_addr)?
    );
    println!(
        "Grant access:  sudo chown $USER /dev/vfio/{}",
        akida_driver::vfio::iommu_group(pcie_addr)?
    );
    Ok(())
}

fn cmd_unbind_vfio(pcie_addr: &str) -> Result<()> {
    println!("Unbinding {pcie_addr} from vfio-pci ...");
    akida_driver::vfio::unbind_from_vfio(pcie_addr)?;
    println!("Done.");
    Ok(())
}

fn cmd_iommu_group(pcie_addr: &str) -> Result<()> {
    let group = akida_driver::vfio::iommu_group(pcie_addr)?;
    println!("IOMMU group for {pcie_addr}: {group}");
    println!("Device file: /dev/vfio/{group}");
    Ok(())
}

fn cmd_setup() -> Result<()> {
    println!("Akida NPU Setup");
    println!("================");
    let mut setup = akida_driver::setup::NpuSetup::new();
    setup.run().map_err(|e| anyhow::anyhow!("Setup failed: {e}"))?;
    println!("\nSetup complete. Run `akida verify` to confirm.");
    Ok(())
}

fn cmd_verify() -> Result<()> {
    println!("Akida NPU Verification");
    println!("======================\n");

    let mut all_ok = true;

    // 1. PCIe device discovery
    print!("PCIe discovery ... ");
    match akida_driver::DeviceManager::discover() {
        Ok(mgr) if mgr.device_count() > 0 => {
            println!("OK ({} device(s))", mgr.device_count());
            for info in mgr.devices() {
                let c = info.capabilities();
                println!(
                    "  [{:?}] {} — {} NPs, {} MB SRAM",
                    c.chip_version,
                    info.pcie_address(),
                    c.npu_count,
                    c.memory_mb
                );
                // IOMMU group
                match akida_driver::vfio::iommu_group(info.pcie_address()) {
                    Ok(g) => {
                        let vfio_dev = format!("/dev/vfio/{g}");
                        let accessible =
                            std::fs::metadata(&vfio_dev).map(|m| !m.permissions().readonly());
                        println!(
                            "  IOMMU group {g} — {}",
                            if accessible.unwrap_or(false) {
                                "accessible"
                            } else {
                                "NOT accessible (run: sudo chown $USER /dev/vfio/<group>)"
                            }
                        );
                    }
                    Err(_) => println!("  IOMMU — not available"),
                }
            }
        }
        Ok(_) => {
            println!("WARN — no devices found");
            all_ok = false;
        }
        Err(e) => {
            println!("FAIL — {e}");
            all_ok = false;
        }
    }

    // 2. Kernel module
    print!("Kernel module   ... ");
    match std::process::Command::new("lsmod").output() {
        Ok(output) => {
            let modules = String::from_utf8_lossy(&output.stdout);
            if modules.contains("akida_pcie") {
                println!("loaded");
            } else {
                println!("not loaded (optional — VFIO path does not need it)");
            }
        }
        Err(_) => println!("could not run lsmod"),
    }

    // 3. Device nodes
    print!("Device nodes    ... ");
    let dev_path = std::path::Path::new("/dev/akida0");
    if dev_path.exists() {
        println!("present ({})", dev_path.display());
    } else {
        println!("absent (normal if using VFIO path)");
    }

    // 4. VFIO container
    print!("VFIO container  ... ");
    let vfio_path = std::path::Path::new("/dev/vfio/vfio");
    if vfio_path.exists() {
        println!("present");
    } else {
        println!("MISSING — load vfio-pci module");
        all_ok = false;
    }

    println!();
    if all_ok {
        println!("All checks passed. Hardware is ready.");
    } else {
        println!("Some checks failed. Run `akida setup` or see docs/SETUP.md.");
    }

    Ok(())
}

fn cmd_parse(file: &str) -> Result<()> {
    let path = std::path::Path::new(file);
    if !path.exists() {
        anyhow::bail!("File not found: {file}");
    }

    let file_size = std::fs::metadata(path)?.len();
    let start = std::time::Instant::now();
    let model = akida_models::prelude::Model::from_file(file)
        .map_err(|e| anyhow::anyhow!("Parse failed: {e}"))?;
    let elapsed = start.elapsed();

    println!("Akida Model: {file}");
    println!("{}", "=".repeat(60));
    println!("File size         : {file_size} bytes ({:.2} KB)", file_size as f64 / 1024.0);
    println!(
        "Decompressed      : {} bytes ({:.2} KB)",
        model.program_size(),
        model.program_size() as f64 / 1024.0
    );
    let ratio = if file_size > 0 {
        model.program_size() as f64 / file_size as f64
    } else {
        0.0
    };
    println!("Compression ratio : {ratio:.2}x");
    println!("SDK version       : {}", model.version());
    println!("Layers            : {}", model.layer_count());
    println!("Weight blocks     : {}", model.weights().len());
    println!("Total weights     : ~{}", model.total_weight_count());
    println!("Parse time        : {elapsed:.3?}");

    if model.layer_count() > 0 {
        println!("\nLayers:");
        for (i, layer) in model.layers().iter().enumerate() {
            println!("  {i:3}. {:<30} {}", layer.name, layer.layer_type);
        }
    }

    println!("\n{}", "=".repeat(60));
    Ok(())
}

fn cmd_zoo_status(dir: &str) -> Result<()> {
    let zoo = akida_models::zoo::ModelZoo::new(dir)
        .map_err(|e| anyhow::anyhow!("Cannot open zoo: {e}"))?;
    zoo.print_status();
    Ok(())
}

fn cmd_guidestone(dir: &str) -> Result<()> {
    let gs = akida_models::guidestone::GuideStone::new(dir);
    let report = gs.run();
    report.print();

    if report.passed() {
        Ok(())
    } else {
        let (_, _, fail) = report.counts();
        Err(anyhow::anyhow!(
            "guideStone: {fail} check(s) failed — substrate not anchored"
        ))
    }
}

fn cmd_convert(weights_src: &str, arch: &str, output: &str, bits: u8) -> Result<()> {
    use akida_models::schema::{LayerDescriptor, ModelDescriptor, PropertyValue};

    if ![1, 2, 4, 8].contains(&bits) {
        anyhow::bail!("--bits must be 1, 2, 4, or 8 (got {bits})");
    }

    println!("Akida Model Conversion (pure Rust)");
    println!("{}", "=".repeat(60));

    // Phase 1: Import weights
    let imported_weights = import_weights(weights_src)?;
    println!(
        "Weights       : {} values from {}",
        imported_weights.len(),
        weights_src
    );

    // Phase 2: Quantize
    let quantized = akida_models::quantize::quantize_per_layer(&imported_weights, bits);
    println!(
        "Quantized     : {}-bit, scale={:.6}, packed={} bytes",
        quantized.bits,
        quantized.scale,
        quantized.packed.len()
    );

    // Phase 3: Parse architecture and build descriptor
    let layers = parse_arch_spec(arch)?;
    println!("Architecture  : {} layer(s)", layers.len());

    let mut desc = ModelDescriptor::new(akida_chip::program::SDK_VERSION_STR);

    let mut weight_offset = 0;
    for (i, (name, layer_type, props)) in layers.iter().enumerate() {
        let mut ld = LayerDescriptor::new(name)
            .with_str("layer_type", layer_type.as_str())
            .with_int("weights_bits", i64::from(bits));

        for (k, v) in props {
            ld.properties
                .push((k.clone(), PropertyValue::Int(*v)));
        }

        // Attach weight data to final layer (or split across layers in future)
        if i == layers.len() - 1 && !quantized.packed.is_empty() {
            let remaining = &quantized.packed[weight_offset..];
            ld = ld.with_weights(remaining.to_vec());
            weight_offset = quantized.packed.len();
        }

        desc.add_layer(ld);
    }

    // Phase 4: Serialize to .fbz
    let fbz = akida_models::schema::build_fbz(&desc);
    std::fs::write(output, &fbz)?;
    println!("Output        : {} ({} bytes)", output, fbz.len());

    // Phase 5: Verify round-trip
    let model = akida_models::Model::from_bytes(&fbz)
        .map_err(|e| anyhow::anyhow!("Round-trip verification failed: {e}"))?;
    println!(
        "Verified      : version={}, layers={}",
        model.version(),
        model.layer_count()
    );

    println!("{}", "=".repeat(60));
    println!("Conversion complete. No Python required.");
    Ok(())
}

fn import_weights(src: &str) -> Result<Vec<f32>> {
    if let Some(count_str) = src.strip_prefix("zeros:") {
        let n: usize = count_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid zeros count: {e}"))?;
        return Ok(vec![0.0_f32; n]);
    }

    if let Some(count_str) = src.strip_prefix("random:") {
        let n: usize = count_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid random count: {e}"))?;
        let mut weights = Vec::with_capacity(n);
        let mut state = 42u64;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            weights.push(val);
        }
        return Ok(weights);
    }

    let path = std::path::Path::new(src);
    if !path.exists() {
        anyhow::bail!("Weight file not found: {src}");
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "npy" => {
            let imported = akida_models::import::load_npy(path)
                .map_err(|e| anyhow::anyhow!("npy import: {e}"))?;
            Ok(imported.data)
        }
        "safetensors" => {
            let tensors = akida_models::import::load_safetensors(path)
                .map_err(|e| anyhow::anyhow!("safetensors import: {e}"))?;
            let all_weights: Vec<f32> = tensors.into_iter().flat_map(|t| t.data).collect();
            Ok(all_weights)
        }
        _ => anyhow::bail!(
            "Unknown weight format: .{ext} (supported: .npy, .safetensors, zeros:N, random:N)"
        ),
    }
}

fn parse_arch_spec(
    arch: &str,
) -> Result<Vec<(String, String, Vec<(String, i64)>)>> {
    let mut layers = Vec::new();

    for (i, token) in arch.split_whitespace().enumerate() {
        if let Some(args_str) = token.strip_prefix("InputConv(").and_then(|s| s.strip_suffix(')'))
        {
            let args: Vec<&str> = args_str.split(',').collect();
            let channels: i64 = args
                .first()
                .unwrap_or(&"1")
                .trim()
                .parse()
                .map_err(|e| anyhow::anyhow!("InputConv channels: {e}"))?;
            let kernel: i64 = args.get(1).unwrap_or(&"1").trim().parse().unwrap_or(1);
            let stride: i64 = args.get(2).unwrap_or(&"1").trim().parse().unwrap_or(1);

            layers.push((
                format!("conv_{i}"),
                "InputConv".to_string(),
                vec![
                    ("channels".to_string(), channels),
                    ("kernel_size".to_string(), kernel),
                    ("stride".to_string(), stride),
                ],
            ));
        } else if let Some(args_str) = token.strip_prefix("FC(").and_then(|s| s.strip_suffix(')'))
        {
            let neurons: i64 = args_str
                .trim()
                .parse()
                .map_err(|e| anyhow::anyhow!("FC neurons: {e}"))?;

            layers.push((
                format!("dense_{i}"),
                "FullyConnected".to_string(),
                vec![("units".to_string(), neurons)],
            ));
        } else if let Some(args_str) =
            token.strip_prefix("SepConv(").and_then(|s| s.strip_suffix(')'))
        {
            let args: Vec<&str> = args_str.split(',').collect();
            let filters: i64 = args
                .first()
                .unwrap_or(&"32")
                .trim()
                .parse()
                .unwrap_or(32);
            let kernel: i64 = args.get(1).unwrap_or(&"3").trim().parse().unwrap_or(3);
            let stride: i64 = args.get(2).unwrap_or(&"1").trim().parse().unwrap_or(1);

            layers.push((
                format!("pw_separable_{i}"),
                "SeparableConv".to_string(),
                vec![
                    ("filters".to_string(), filters),
                    ("kernel_size".to_string(), kernel),
                    ("stride".to_string(), stride),
                ],
            ));
        } else {
            anyhow::bail!(
                "Unknown layer spec: '{token}'. Expected: InputConv(ch,k,s), FC(n), SepConv(f,k,s)"
            );
        }
    }

    if layers.is_empty() {
        anyhow::bail!("Architecture spec is empty. Provide at least one layer.");
    }

    Ok(layers)
}

fn cmd_import_onnx(onnx_path: &str, output: Option<&str>) -> Result<()> {
    use akida_models::import::onnx;

    println!("ONNX Import");
    println!("===========");
    println!();

    let import = onnx::load_onnx(std::path::Path::new(onnx_path))?;

    println!("  File       : {onnx_path}");
    println!("  IR version : {}", import.ir_version);
    println!("  Opset      : {}", import.opset_version);
    println!("  Producer   : {}", import.producer);
    println!("  Graph      : {}", import.graph_name);
    println!("  Nodes      : {}", import.nodes.len());
    println!("  Weights    : {} tensors", import.weights.len());
    let total_params: usize = import.weights.iter().map(|w| w.data.len()).sum();
    println!("  Parameters : {total_params}");
    println!();

    if !import.inputs.is_empty() {
        println!("  Inputs:");
        for inp in &import.inputs {
            println!("    {} : {:?} ({})", inp.name, inp.shape, inp.elem_type);
        }
        println!();
    }

    if !import.outputs.is_empty() {
        println!("  Outputs:");
        for out in &import.outputs {
            println!("    {} : {:?} ({})", out.name, out.shape, out.elem_type);
        }
        println!();
    }

    let report = onnx::compatibility_report(&import);
    println!("  Akida Compatibility:");
    println!("    Supported ops   : {}/{} ({:.0}%)",
        report.supported_ops, report.total_ops, report.coverage * 100.0);
    if !report.unsupported_op_types.is_empty() {
        let mut unique: Vec<String> = report.unsupported_op_types.clone();
        unique.sort();
        unique.dedup();
        println!("    Unsupported     : {}", unique.join(", "));
    }
    println!();

    println!("  Op breakdown:");
    let mut op_list: Vec<_> = report.op_counts.iter().collect();
    op_list.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &op_list {
        let supported = onnx::SUPPORTED_OPS.iter().any(|s| *s == op.as_str());
        let marker = if supported { "+" } else { "-" };
        println!("    [{marker}] {op:>25} × {count}");
    }
    println!();

    if !import.weights.is_empty() {
        println!("  Weight tensors (top 10):");
        for w in import.weights.iter().take(10) {
            println!("    {:>40} : {:?} ({} params)", w.name, w.shape, w.data.len());
        }
        if import.weights.len() > 10 {
            println!("    ... and {} more", import.weights.len() - 10);
        }
        println!();
    }

    if let Some(out_path) = output {
        println!("  Quantizing weights (naive int8)...");
        let quantized = onnx::quantize_weights_naive(&import.weights);
        let total_bytes: usize = quantized.iter().map(|(_, d, _)| d.len()).sum();
        println!("  Quantized {total_params} f32 params → {total_bytes} int8 bytes");

        // Write quantized weights as a simple binary blob for now.
        // Full .fbz serialization requires mapping ONNX ops → Akida LayerSpec
        // (architecture-dependent), which is the next evolution step.
        let mut blob = Vec::new();
        for (name, data, shape) in &quantized {
            let name_bytes = name.as_bytes();
            blob.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            blob.extend_from_slice(name_bytes);
            blob.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                blob.extend_from_slice(&(dim as u32).to_le_bytes());
            }
            blob.extend_from_slice(&(data.len() as u32).to_le_bytes());
            blob.extend_from_slice(data);
        }
        std::fs::write(out_path, &blob)?;
        println!("  Wrote {} bytes to {out_path}", blob.len());
        println!("  (Binary weight blob — full .fbz conversion requires op mapping)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_arch_single_fc() {
        let layers = parse_arch_spec("FC(128)").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].1, "FullyConnected");
        assert_eq!(layers[0].2[0].1, 128);
    }

    #[test]
    fn parse_arch_input_conv_with_params() {
        let layers = parse_arch_spec("InputConv(50,3,2)").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].1, "InputConv");
        assert_eq!(layers[0].2[0].1, 50); // channels
        assert_eq!(layers[0].2[1].1, 3); // kernel
        assert_eq!(layers[0].2[2].1, 2); // stride
    }

    #[test]
    fn parse_arch_multi_layer_pipeline() {
        let layers = parse_arch_spec("InputConv(50,1,1) FC(128) FC(1)").unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].1, "InputConv");
        assert_eq!(layers[1].1, "FullyConnected");
        assert_eq!(layers[2].1, "FullyConnected");
    }

    #[test]
    fn parse_arch_separable_conv() {
        let layers = parse_arch_spec("SepConv(64,3,1)").unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].1, "SeparableConv");
        assert_eq!(layers[0].2[0].1, 64);
    }

    #[test]
    fn parse_arch_empty_fails() {
        assert!(parse_arch_spec("").is_err());
    }

    #[test]
    fn parse_arch_unknown_layer_fails() {
        assert!(parse_arch_spec("LSTM(128)").is_err());
    }

    #[test]
    fn parse_arch_input_conv_defaults() {
        let layers = parse_arch_spec("InputConv(32)").unwrap();
        assert_eq!(layers[0].2[0].1, 32); // channels
        assert_eq!(layers[0].2[1].1, 1); // kernel default
        assert_eq!(layers[0].2[2].1, 1); // stride default
    }

    #[test]
    fn parse_arch_complex_pipeline() {
        let layers =
            parse_arch_spec("InputConv(3,3,1) SepConv(64,3,2) FC(256) FC(128) FC(10)").unwrap();
        assert_eq!(layers.len(), 5);
        assert_eq!(layers[0].1, "InputConv");
        assert_eq!(layers[1].1, "SeparableConv");
        assert_eq!(layers[2].1, "FullyConnected");
        assert_eq!(layers[3].1, "FullyConnected");
        assert_eq!(layers[4].1, "FullyConnected");
        assert_eq!(layers[4].2[0].1, 10);
    }
}
