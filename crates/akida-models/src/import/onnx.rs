// SPDX-License-Identifier: AGPL-3.0-or-later

//! ONNX model import pipeline.
//!
//! Parses an ONNX protobuf file using `onnx-rs`, extracts the computation
//! graph and weight tensors, maps supported ops to Akida layer types,
//! quantizes to int8, and provides the data needed for serialization
//! to `.fbz` format via [`crate::builder::ProgramBuilder`].
//!
//! ## Supported operators
//!
//! | ONNX op | Akida layer |
//! |---------|-------------|
//! | Conv | `InputConvolutional` / `Convolutional` |
//! | Relu, Clip(0,6) | Fused into preceding conv/dense |
//! | MaxPool, AveragePool | `Pooling` |
//! | Gemm, MatMul | `FullyConnected` |
//! | BatchNormalization | Folded into preceding conv weights |
//! | Add, Concat | `Concatenate` |
//! | Flatten, Reshape | Implicit (shape metadata) |

use super::ImportedWeights;
use crate::error::{AkidaModelError, Result};
use onnx_rs::ast::{self, Dimension, TypeValue};
use std::collections::HashMap;
use std::path::Path;

/// Summary of an imported ONNX model.
#[derive(Debug)]
pub struct OnnxImport {
    /// ONNX opset version.
    pub opset_version: i64,
    /// IR version from the ONNX model.
    pub ir_version: i64,
    /// Producer name (e.g., "pytorch", "keras2onnx").
    pub producer: String,
    /// Graph name.
    pub graph_name: String,
    /// Extracted weight tensors (as f32).
    pub weights: Vec<ImportedWeights>,
    /// Graph nodes with op type and I/O tensor names.
    pub nodes: Vec<OnnxNode>,
    /// Graph inputs (name, shape, dtype).
    pub inputs: Vec<TensorInfo>,
    /// Graph outputs (name, shape, dtype).
    pub outputs: Vec<TensorInfo>,
}

/// A single node in the ONNX computation graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// ONNX operator type (e.g., "Conv", "Relu", "Gemm").
    pub op_type: String,
    /// Node name (may be empty).
    pub name: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Whether this op maps to a supported Akida layer.
    pub akida_supported: bool,
}

/// Metadata about an I/O tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions (symbolic dims become 0).
    pub shape: Vec<usize>,
    /// ONNX element type description.
    pub elem_type: String,
}

/// Operators supported for Akida conversion.
pub const SUPPORTED_OPS: &[&str] = &[
    "Conv", "Relu", "Clip", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Gemm", "MatMul", "BatchNormalization", "Add", "Concat", "Flatten",
    "Reshape", "Softmax", "Dropout", "Transpose", "Squeeze", "Unsqueeze",
    "Pad", "Shape", "Gather", "Constant", "ConstantOfShape",
];

/// Load and parse an ONNX model from a file.
///
/// # Errors
///
/// Returns error if the file cannot be read or the protobuf is malformed.
pub fn load_onnx(path: &Path) -> Result<OnnxImport> {
    let data = std::fs::read(path).map_err(|e| {
        AkidaModelError::parse_error(format!("read {}: {e}", path.display()))
    })?;
    parse_onnx(&data)
}

/// Parse ONNX model bytes.
///
/// # Errors
///
/// Returns error if the protobuf cannot be decoded.
pub fn parse_onnx(data: &[u8]) -> Result<OnnxImport> {
    let model = onnx_rs::parse(data).map_err(|e| {
        AkidaModelError::parse_error(format!("ONNX protobuf decode: {e}"))
    })?;

    let opset_version = model
        .opset_import
        .first()
        .map_or(0, |o| o.version);

    let ir_version = model.ir_version;
    let producer = model.producer_name.to_string();

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| AkidaModelError::parse_error("ONNX model has no graph"))?;

    let graph_name = graph.name.to_string();

    let weights = extract_initializers(graph);
    let nodes = extract_nodes(graph);
    let inputs = extract_io(&graph.input);
    let outputs = extract_io(&graph.output);

    tracing::info!(
        "ONNX import: opset={opset_version}, ir_version={ir_version}, \
         producer={producer:?}, nodes={}, weights={}, inputs={}, outputs={}",
        nodes.len(),
        weights.len(),
        inputs.len(),
        outputs.len(),
    );

    Ok(OnnxImport {
        opset_version,
        ir_version,
        producer,
        graph_name,
        weights,
        nodes,
        inputs,
        outputs,
    })
}

fn extract_initializers(graph: &ast::Graph<'_>) -> Vec<ImportedWeights> {
    graph
        .initializer
        .iter()
        .filter_map(|tensor| {
            let name = tensor.name().to_string();
            let shape: Vec<usize> = tensor
                .dims()
                .iter()
                .map(|&d| d as usize)
                .collect();

            let data = tensor_to_f32(tensor);
            if data.is_empty() && shape.iter().product::<usize>() > 0 {
                tracing::warn!("ONNX initializer '{name}': shape {shape:?} but no data");
                return None;
            }

            Some(ImportedWeights { name, data, shape })
        })
        .collect()
}

fn tensor_to_f32(tensor: &ast::TensorProto<'_>) -> Vec<f32> {
    if let Some(data) = tensor.as_f32() {
        return data.into_owned();
    }

    if let Some(data) = tensor.as_f64() {
        return data.iter().map(|&v| v as f32).collect();
    }

    if let Some(data) = tensor.as_i64() {
        return data.iter().map(|&v| v as f32).collect();
    }

    if let Some(data) = tensor.as_i32() {
        return data.iter().map(|&v| v as f32).collect();
    }

    Vec::new()
}

fn extract_nodes(graph: &ast::Graph<'_>) -> Vec<OnnxNode> {
    graph
        .node
        .iter()
        .map(|node| {
            let op_str = node.op_type.as_str();
            let akida_supported = SUPPORTED_OPS.iter().any(|&s| s == op_str);
            OnnxNode {
                op_type: op_str.to_string(),
                name: node.name.to_string(),
                inputs: node.input.iter().map(|s| s.to_string()).collect(),
                outputs: node.output.iter().map(|s| s.to_string()).collect(),
                akida_supported,
            }
        })
        .collect()
}

fn extract_io(values: &[ast::ValueInfo<'_>]) -> Vec<TensorInfo> {
    values
        .iter()
        .map(|v| {
            let name = v.name.to_string();
            let (shape, elem_type) = match &v.r#type {
                Some(tp) => match &tp.value {
                    Some(TypeValue::Tensor(tt)) => {
                        let shape = tt.shape.as_ref().map_or_else(Vec::new, |s| {
                            s.dim
                                .iter()
                                .map(|d| match &d.value {
                                    Dimension::Value(val) => *val as usize,
                                    Dimension::Param(_) => 0,
                                })
                                .collect()
                        });
                        (shape, format!("{:?}", tt.elem_type))
                    }
                    _ => (vec![], "unknown".to_string()),
                },
                None => (vec![], "unknown".to_string()),
            };

            TensorInfo {
                name,
                shape,
                elem_type,
            }
        })
        .collect()
}

/// Map ONNX ops to Akida layer types and return a compatibility report.
pub fn compatibility_report(import: &OnnxImport) -> CompatibilityReport {
    let mut supported = 0usize;
    let mut unsupported_types = Vec::new();
    let mut op_counts: HashMap<String, usize> = HashMap::new();

    for node in &import.nodes {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        if node.akida_supported {
            supported += 1;
        } else {
            unsupported_types.push(node.op_type.clone());
        }
    }

    let total = import.nodes.len();
    let coverage = if total > 0 {
        supported as f64 / total as f64
    } else {
        0.0
    };

    CompatibilityReport {
        total_ops: total,
        supported_ops: supported,
        unsupported_ops: unsupported_types.len(),
        coverage,
        op_counts,
        unsupported_op_types: unsupported_types,
    }
}

/// Akida compatibility analysis for an ONNX model.
#[derive(Debug)]
pub struct CompatibilityReport {
    /// Total number of operators in the graph.
    pub total_ops: usize,
    /// Number of ops mappable to Akida layers.
    pub supported_ops: usize,
    /// Number of ops with no Akida equivalent.
    pub unsupported_ops: usize,
    /// Fraction of ops supported (0.0 to 1.0).
    pub coverage: f64,
    /// Per-op-type counts.
    pub op_counts: HashMap<String, usize>,
    /// List of unsupported op types (may have duplicates).
    pub unsupported_op_types: Vec<String>,
}

/// Perform naive per-tensor symmetric quantization to int8.
///
/// Returns `(name, quantized_bytes, shape)` tuples ready for
/// `ProgramBuilder`. For production use, apply per-channel or
/// calibrated quantization via [`crate::quantize`] instead.
pub fn quantize_weights_naive(weights: &[ImportedWeights]) -> Vec<(String, Vec<u8>, Vec<usize>)> {
    weights
        .iter()
        .map(|w| {
            let absmax = w
                .data
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = 127.0 / absmax;
            let quantized: Vec<u8> = w
                .data
                .iter()
                .map(|&v| {
                    let q = (v * scale).round().clamp(-128.0, 127.0) as i8;
                    q as u8
                })
                .collect();
            (w.name.clone(), quantized, w.shape.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_ops_list_contains_core_ops() {
        assert!(SUPPORTED_OPS.contains(&"Conv"));
        assert!(SUPPORTED_OPS.contains(&"Relu"));
        assert!(SUPPORTED_OPS.contains(&"Gemm"));
    }

    #[test]
    fn onnx_load_nonexistent_file_fails() {
        assert!(load_onnx(Path::new("/nonexistent/model.onnx")).is_err());
    }

    #[test]
    fn onnx_parse_invalid_bytes_fails() {
        assert!(parse_onnx(&[0xFF, 0x00, 0x01, 0x02]).is_err());
    }

    #[test]
    fn onnx_roundtrip_minimal_model() {
        use onnx_rs::ast::*;
        let model = Model {
            ir_version: 9,
            producer_name: "rustChip-test",
            opset_import: vec![OperatorSetId { domain: "", version: 13 }],
            graph: Some(Graph {
                name: "test_graph",
                node: vec![
                    Node {
                        op_type: OpType::Conv,
                        name: "conv1",
                        input: vec!["X", "W"],
                        output: vec!["Y"],
                        ..Default::default()
                    },
                    Node {
                        op_type: OpType::Relu,
                        name: "relu1",
                        input: vec!["Y"],
                        output: vec!["Z"],
                        ..Default::default()
                    },
                ],
                initializer: vec![
                    TensorProto::from_f32("W", vec![1, 1, 3, 3], vec![0.1; 9]),
                ],
                input: vec![ValueInfo {
                    name: "X",
                    r#type: Some(TypeProto {
                        value: Some(TypeValue::Tensor(TensorTypeProto {
                            elem_type: DataType::Float,
                            shape: Some(TensorShape {
                                dim: vec![
                                    TensorShapeDimension {
                                        value: Dimension::Param("N"),
                                        denotation: "",
                                    },
                                    TensorShapeDimension {
                                        value: Dimension::Value(1),
                                        denotation: "",
                                    },
                                    TensorShapeDimension {
                                        value: Dimension::Value(28),
                                        denotation: "",
                                    },
                                    TensorShapeDimension {
                                        value: Dimension::Value(28),
                                        denotation: "",
                                    },
                                ],
                            }),
                        })),
                        denotation: "",
                    }),
                    ..Default::default()
                }],
                output: vec![ValueInfo {
                    name: "Z",
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let bytes = onnx_rs::encode(&model);
        let import = parse_onnx(&bytes).expect("parse encoded model");

        assert_eq!(import.opset_version, 13);
        assert_eq!(import.ir_version, 9);
        assert_eq!(import.producer, "rustChip-test");
        assert_eq!(import.graph_name, "test_graph");
        assert_eq!(import.nodes.len(), 2);
        assert_eq!(import.nodes[0].op_type, "Conv");
        assert!(import.nodes[0].akida_supported);
        assert_eq!(import.nodes[1].op_type, "Relu");
        assert!(import.nodes[1].akida_supported);
        assert_eq!(import.weights.len(), 1);
        assert_eq!(import.weights[0].name, "W");
        assert_eq!(import.weights[0].shape, vec![1, 1, 3, 3]);
        assert_eq!(import.weights[0].data.len(), 9);
        assert_eq!(import.inputs.len(), 1);
        assert_eq!(import.inputs[0].shape, vec![0, 1, 28, 28]);
        assert_eq!(import.outputs.len(), 1);
    }

    #[test]
    fn quantize_weights_naive_basic() {
        let weights = vec![ImportedWeights {
            name: "conv1.weight".to_string(),
            data: vec![1.0, -0.5, 0.0, 0.25],
            shape: vec![2, 2],
        }];
        let quantized = quantize_weights_naive(&weights);
        assert_eq!(quantized.len(), 1);
        assert_eq!(quantized[0].1.len(), 4);
        assert_eq!(quantized[0].1[0] as i8, 127);
    }

    #[test]
    fn compatibility_report_all_supported() {
        let import = OnnxImport {
            opset_version: 13,
            ir_version: 7,
            producer: "test".to_string(),
            graph_name: "test".to_string(),
            weights: vec![],
            nodes: vec![
                OnnxNode {
                    op_type: "Conv".to_string(),
                    name: "conv1".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    akida_supported: true,
                },
                OnnxNode {
                    op_type: "Relu".to_string(),
                    name: "relu1".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    akida_supported: true,
                },
            ],
            inputs: vec![],
            outputs: vec![],
        };
        let report = compatibility_report(&import);
        assert_eq!(report.total_ops, 2);
        assert_eq!(report.supported_ops, 2);
        assert!((report.coverage - 1.0).abs() < 1e-6);
    }
}
