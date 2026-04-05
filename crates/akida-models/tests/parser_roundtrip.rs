// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parser round-trip and model creation tests.
//!
//! Verifies that models can be parsed, queried, and that stub models
//! created by the zoo pass validation.

use akida_models::{Model, zoo::ModelZoo, zoo::ZooModel};

fn make_stub_fbz(version: &str) -> Vec<u8> {
    let mut data = Vec::with_capacity(512);
    data.extend_from_slice(&[0x80, 0x44, 0x04, 0x10]);
    data.extend_from_slice(&[0x00; 26]);
    data.extend_from_slice(version.as_bytes());
    data.push(0x00);
    while data.len() < 512 {
        data.push(0x00);
    }
    data
}

#[test]
fn test_parse_stub_model() {
    let data = make_stub_fbz("2.18.2");
    let model = Model::from_bytes(&data).unwrap();
    assert_eq!(model.version(), "2.18.2");
    assert!(model.program_size() >= 512);
}

#[test]
fn test_parse_different_versions() {
    for version in ["2.18.2", "3.0.0", "1.5.12"] {
        let data = make_stub_fbz(version);
        let model = Model::from_bytes(&data).unwrap();
        assert_eq!(model.version(), version, "Version mismatch for {version}");
    }
}

#[test]
fn test_invalid_magic_rejected() {
    let mut data = make_stub_fbz("2.18.2");
    data[0] = 0xFF;
    assert!(Model::from_bytes(&data).is_err());
}

#[test]
fn test_too_small_rejected() {
    let data = vec![0x80, 0x44, 0x04, 0x10];
    assert!(Model::from_bytes(&data).is_err());
}

#[test]
fn test_model_io_sizes_positive() {
    let data = make_stub_fbz("2.18.2");
    let model = Model::from_bytes(&data).unwrap();
    assert!(model.input_size() > 0, "Input size should be > 0");
    assert!(model.output_size() > 0, "Output size should be > 0");
}

#[test]
fn test_zoo_stub_round_trip() {
    let temp = tempfile::TempDir::new().unwrap();
    let mut zoo = ModelZoo::new(temp.path()).unwrap();
    let path = zoo.create_reference_model(ZooModel::MinimalFc).unwrap();

    assert!(path.exists());

    let model = Model::from_file(&path).unwrap();
    assert_eq!(model.version(), "2.18.2");
    assert!(model.program_size() > 0);
}

#[test]
fn test_weight_data_decode() {
    let data = make_stub_fbz("2.18.2");
    let model = Model::from_bytes(&data).unwrap();

    for w in model.weights() {
        let decoded = w.decode().unwrap();
        assert!(decoded.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn test_layer_type_inference() {
    let data = make_stub_fbz("2.18.2");
    let model = Model::from_bytes(&data).unwrap();
    for layer in model.layers() {
        assert!(!layer.name.is_empty() || model.layer_count() == 0);
    }
}
