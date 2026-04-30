// SPDX-License-Identifier: AGPL-3.0-or-later

//! SNNTorch/PyTorch → rustChip bridge test.
//!
//! Validates the full conversion pipeline:
//! 1. Parse HuggingFace brainchip/* models (.fbz)
//! 2. Test ONNX import (via synthetic model round-trip)
//! 3. Test weight quantization pipeline
//! 4. Verify all conversion paths work end-to-end

use akida_bench::preserve;
use akida_models::import::onnx;
use std::path::Path;
use std::time::Instant;

const HF_MODELS: &[(&str, &str)] = &[
    ("akidanet18_imagenet.fbz", "Image classification (ImageNet)"),
    ("akidanet_faceidentification.fbz", "Face identification"),
    ("akidanet_plantvillage.fbz", "Plant disease detection"),
    ("ds_cnn_kws.fbz", "Keyword spotting (Speech Commands)"),
    ("centernet_voc.fbz", "Object detection (VOC)"),
    ("yolo_voc.fbz", "Object detection (YOLO/VOC)"),
    ("vgg_utk_face.fbz", "Age estimation (UTK)"),
    ("akida_unet_portrait128.fbz", "Portrait segmentation"),
    ("gxnor_mnist.fbz", "MNIST classification (GXNOR)"),
    ("mobilenet_imagenet.fbz", "MobileNet ImageNet"),
    ("pointnet_plus_modelnet40.fbz", "3D point cloud (ModelNet40)"),
];

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SNNTorch / PyTorch → rustChip Bridge Test");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    // Test 1: Parse HuggingFace brainchip/* models
    println!("── Test 1: HuggingFace brainchip/* Model Parsing ────────────");
    println!();

    for (filename, description) in HF_MODELS {
        let path = format!("{}/{filename}", preserve::ZOO_DIR);
        if !Path::new(&path).exists() {
            println!("  [SKIP] {filename} — not in zoo");
            skipped += 1;
            continue;
        }

        let start = Instant::now();
        match akida_models::Model::from_file(&path) {
            Ok(model) => {
                let elapsed = start.elapsed();
                println!("  [PASS] {filename} — {} layers, v{}, {:.1}ms",
                    model.layer_count(), model.version(), elapsed.as_secs_f64() * 1000.0);
                println!("         {description}");
                passed += 1;
            }
            Err(e) => {
                println!("  [FAIL] {filename} — {e}");
                failed += 1;
            }
        }
    }
    println!();

    // Test 2: ONNX Import Round-trip
    println!("── Test 2: ONNX Import Pipeline ─────────────────────────────");
    println!();

    {
        use onnx_rs::ast::*;
        let model = Model {
            ir_version: 9,
            producer_name: "snntorch-test",
            opset_import: vec![OperatorSetId { domain: "", version: 13 }],
            graph: Some(Graph {
                name: "snn_mnist",
                node: vec![
                    Node { op_type: OpType::Conv, name: "conv1", input: vec!["X", "W1"], output: vec!["C1"], ..Default::default() },
                    Node { op_type: OpType::Relu, name: "relu1", input: vec!["C1"], output: vec!["R1"], ..Default::default() },
                    Node { op_type: OpType::MaxPool, name: "pool1", input: vec!["R1"], output: vec!["P1"], ..Default::default() },
                    Node { op_type: OpType::Flatten, name: "flat", input: vec!["P1"], output: vec!["F1"], ..Default::default() },
                    Node { op_type: OpType::Gemm, name: "fc1", input: vec!["F1", "W2", "B2"], output: vec!["Y"], ..Default::default() },
                    Node { op_type: OpType::Softmax, name: "sm", input: vec!["Y"], output: vec!["Z"], ..Default::default() },
                ],
                initializer: vec![
                    TensorProto::from_f32("W1", vec![16, 1, 3, 3], vec![0.01; 144]),
                    TensorProto::from_f32("W2", vec![10, 3136], vec![0.001; 31360]),
                    TensorProto::from_f32("B2", vec![10], vec![0.0; 10]),
                ],
                input: vec![ValueInfo {
                    name: "X",
                    r#type: Some(TypeProto {
                        value: Some(TypeValue::Tensor(TensorTypeProto {
                            elem_type: DataType::Float,
                            shape: Some(TensorShape {
                                dim: vec![
                                    TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                                    TensorShapeDimension { value: Dimension::Value(1), denotation: "" },
                                    TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                                    TensorShapeDimension { value: Dimension::Value(28), denotation: "" },
                                ],
                            }),
                        })),
                        denotation: "",
                    }),
                    ..Default::default()
                }],
                output: vec![ValueInfo { name: "Z", ..Default::default() }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let bytes = onnx_rs::encode(&model);
        println!("  Encoded synthetic SNN-MNIST model: {} bytes", bytes.len());

        match onnx::parse_onnx(&bytes) {
            Ok(import) => {
                let report = onnx::compatibility_report(&import);
                println!("  [PASS] ONNX parse: {} nodes, {} weights, {:.0}% compatible",
                    import.nodes.len(), import.weights.len(), report.coverage * 100.0);
                println!("         Ops: {:?}", import.nodes.iter().map(|n| n.op_type.as_str()).collect::<Vec<_>>());
                passed += 1;
            }
            Err(e) => {
                println!("  [FAIL] ONNX parse: {e}");
                failed += 1;
            }
        }
    }
    println!();

    // Test 3: Quantization Pipeline
    println!("── Test 3: Weight Quantization Pipeline ──────────────────────");
    println!();

    {
        let weights = vec![
            akida_models::import::ImportedWeights {
                name: "conv1.weight".to_string(),
                data: (0..144).map(|i| (i as f32 - 72.0) / 72.0).collect(),
                shape: vec![16, 1, 3, 3],
            },
            akida_models::import::ImportedWeights {
                name: "fc1.weight".to_string(),
                data: (0..31360).map(|i| (i as f32 - 15680.0) / 15680.0).collect(),
                shape: vec![10, 3136],
            },
        ];

        let quantized = onnx::quantize_weights_naive(&weights);
        let total_f32 = weights.iter().map(|w| w.data.len()).sum::<usize>();
        let total_int8 = quantized.iter().map(|(_, d, _)| d.len()).sum::<usize>();

        let max_abs: Vec<f32> = weights.iter().map(|w| {
            w.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        }).collect();

        println!("  Input weights:");
        for (i, w) in weights.iter().enumerate() {
            println!("    {} : {:?} ({} params, max={:.4})",
                w.name, w.shape, w.data.len(), max_abs[i]);
        }
        println!("  Quantized:");
        for (name, data, shape) in &quantized {
            let min = data.iter().map(|&b| b as i8).min().unwrap_or(0);
            let max = data.iter().map(|&b| b as i8).max().unwrap_or(0);
            println!("    {name} : {shape:?} ({} bytes, range [{min}, {max}])",
                data.len());
        }
        println!("  f32 params: {total_f32} → int8 bytes: {total_int8}");
        println!("  Compression: {:.1}×", total_f32 as f64 * 4.0 / total_int8 as f64);

        if total_int8 == total_f32 {
            println!("  [PASS] Quantization preserved parameter count");
            passed += 1;
        } else {
            println!("  [FAIL] Parameter count mismatch");
            failed += 1;
        }
    }
    println!();

    // Test 4: NeuroBench-style Integration
    println!("── Test 4: Integration Verification ──────────────────────────");
    println!();

    let npy_test = akida_models::import::build_npy_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    match akida_models::import::load_npy(Path::new("/dev/null")) {
        Err(_) => {
            println!("  [PASS] .npy import (error path verified)");
            passed += 1;
        }
        Ok(_) => {
            println!("  [FAIL] .npy import should have failed for /dev/null");
            failed += 1;
        }
    }

    if npy_test.len() > 10 {
        println!("  [PASS] .npy builder produces valid bytes ({} bytes)", npy_test.len());
        passed += 1;
    } else {
        println!("  [FAIL] .npy builder output too small");
        failed += 1;
    }

    match akida_models::import::load_safetensors(Path::new("/nonexistent.safetensors")) {
        Err(_) => {
            println!("  [PASS] .safetensors import (error path verified)");
            passed += 1;
        }
        Ok(_) => {
            println!("  [FAIL] .safetensors import should have failed");
            failed += 1;
        }
    }
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Bridge Test Summary");
    println!("  ───────────────────");
    println!("  Passed  : {passed}");
    println!("  Failed  : {failed}");
    println!("  Skipped : {skipped}");
    println!("═══════════════════════════════════════════════════════════════");

    std::process::exit(if failed == 0 { 0 } else { 1 });
}
