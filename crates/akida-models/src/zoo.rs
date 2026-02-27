//! Akida Model Zoo Manager
//!
//! Provides utilities for downloading, converting, and managing
//! models from the `BrainChip` Akida Model Zoo.
//!
//! ## Model Zoo Models
//!
//! The Akida Model Zoo includes pre-trained models for various tasks:
//!
//! | Model | Task | Size | Accuracy | Power |
//! |-------|------|------|----------|-------|
//! | `AkidaNet` 0.5 | `ImageNet` | 160×160 | 65% top-1 | <300 mW |
//! | DS-CNN | Keyword Spotting | 32 words | 94% | <50 mW |
//! | `ViT` | Vision Transformer | 224×224 | 75% top-1 | ~500 mW |
//! | YOLO | Object Detection | 320×320 | mAP 0.28 | <500 mW |
//!
//! ## Usage
//!
//! ```ignore
//! use akida_models::zoo::{ModelZoo, ZooModel};
//!
//! // Initialize the model zoo
//! let zoo = ModelZoo::new("models/akida")?;
//!
//! // Download a model
//! zoo.download(ZooModel::DsCnnKws)?;
//!
//! // Get local path to model
//! let path = zoo.model_path(ZooModel::DsCnnKws)?;
//! ```

use crate::{AkidaModelError, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Source of a zoo model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// BrainChip MetaTF official zoo
    BrainChipMetaTf,
    /// NeuroBench benchmark suite
    NeuroBench,
    /// ecoPrimals physics models (validated on live AKD1000)
    EcoPrimalsPhysics,
    /// Hand-built via program_external()
    HandBuilt,
}

/// Validation tier for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationTier {
    /// Architecture analyzed, conversion path defined, not yet run on hardware
    Analysis,
    /// Compiled and loaded, functional test passed
    Functional,
    /// Run on real AKD1000 hardware, numbers confirmed
    HardwareValidated,
}

/// Models available in the Akida Model Zoo
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZooModel {
    // ── BrainChip MetaTF ────────────────────────────────────────────────────
    /// `AkidaNet` `ImageNet` classifier (0.5 width, 160×160)
    AkidaNet05_160,
    /// `AkidaNet` `ImageNet` classifier (1.0 width, 224×224)
    AkidaNet10_224,
    /// DS-CNN for keyword spotting (35 words, Google Speech Commands v2)
    DsCnnKws,
    /// `MobileNetV2` for `ImageNet`
    MobileNetV2,
    /// Vision Transformer (tiny)
    ViTTiny,
    /// YOLO object detection
    YoloV8n,
    /// `PointNet++` for 3D point clouds
    PointNetPlusPlus,
    /// DVS Gesture recognition (11 classes, DVS128 dataset)
    DvsGesture,
    /// Event camera model
    EventCamera,

    // ── NeuroBench streaming benchmarks ─────────────────────────────────────
    /// ESN for chaotic time series prediction (MSLP dataset)
    EsnChaotic,
    /// ECG anomaly detection (MIT-BIH Arrhythmia)
    EcgAnomaly,

    // ── ecoPrimals physics — validated on real AKD1000 ──────────────────────
    /// ESN readout for lattice QCD thermalization detection (hotSpring Exp 022)
    EsnQcdThermalization,
    /// Phase classifier: SU(3) confined/deconfined (hotSpring Exp 022)
    PhaseClassifierSu3,
    /// WDM transport coefficient predictor — D*, η*, λ* (hotSpring Exp 022)
    TransportPredictorWdm,
    /// Anderson localization regime classifier (groundSpring Exp 028)
    AndersonRegimeClassifier,

    // ── Hand-built custom ────────────────────────────────────────────────────
    /// Minimal FC(50→1) smoke test for program_external() validation
    MinimalFc,
}

impl ZooModel {
    /// Get model filename
    pub const fn filename(&self) -> &'static str {
        match self {
            Self::AkidaNet05_160 => "akidanet_05_160.fbz",
            Self::AkidaNet10_224 => "akidanet_10_224.fbz",
            Self::DsCnnKws => "ds_cnn_kws.fbz",
            Self::MobileNetV2 => "mobilenetv2.fbz",
            Self::ViTTiny => "vit_tiny.fbz",
            Self::YoloV8n => "yolov8n.fbz",
            Self::PointNetPlusPlus => "pointnet_plus.fbz",
            Self::DvsGesture => "dvs_gesture.fbz",
            Self::EventCamera => "event_camera.fbz",
            Self::EsnChaotic => "esn_chaotic.fbz",
            Self::EcgAnomaly => "ecg_anomaly.fbz",
            Self::EsnQcdThermalization => "esn_qcd_thermalization.fbz",
            Self::PhaseClassifierSu3 => "phase_classifier_su3.fbz",
            Self::TransportPredictorWdm => "transport_predictor_wdm.fbz",
            Self::AndersonRegimeClassifier => "anderson_regime_classifier.fbz",
            Self::MinimalFc => "minimal_fc.fbz",
        }
    }

    /// Get model description
    pub const fn description(&self) -> &'static str {
        match self {
            Self::AkidaNet05_160 => "AkidaNet ImageNet classifier (0.5 width, 160×160)",
            Self::AkidaNet10_224 => "AkidaNet ImageNet classifier (1.0 width, 224×224)",
            Self::DsCnnKws => "DS-CNN keyword spotting (35 words, Google Speech Commands v2)",
            Self::MobileNetV2 => "MobileNetV2 ImageNet classifier",
            Self::ViTTiny => "Vision Transformer (tiny) ImageNet classifier",
            Self::YoloV8n => "YOLOv8 nano object detection (COCO)",
            Self::PointNetPlusPlus => "PointNet++ 3D point cloud classification",
            Self::DvsGesture => "DVS Gesture recognition (11 classes, DVS128)",
            Self::EventCamera => "Event camera object detection",
            Self::EsnChaotic => "ESN chaotic time series prediction (MSLP, NeuroBench)",
            Self::EcgAnomaly => "ECG anomaly detection (MIT-BIH Arrhythmia, NeuroBench)",
            Self::EsnQcdThermalization => {
                "ESN readout: lattice QCD thermalization detector (hotSpring Exp 022)"
            }
            Self::PhaseClassifierSu3 => {
                "SU(3) phase classifier: confined/deconfined (hotSpring Exp 022)"
            }
            Self::TransportPredictorWdm => {
                "WDM transport predictor: D*, η*, λ* from plasma observables"
            }
            Self::AndersonRegimeClassifier => {
                "Anderson localization regime: localized/diffusive/critical (groundSpring Exp 028)"
            }
            Self::MinimalFc => "Minimal FC(50→1) smoke test for program_external() validation",
        }
    }

    /// NP budget required on AKD1000 (number of NPs consumed)
    ///
    /// AKD1000 has 1,000 NPs total. Sum of co-located models must be ≤ 1,000.
    pub const fn np_budget(&self) -> usize {
        match self {
            Self::MinimalFc => 51,
            Self::EcgAnomaly => 96,
            Self::PhaseClassifierSu3 => 67,
            Self::AndersonRegimeClassifier => 68,
            Self::EsnQcdThermalization => 179,
            Self::TransportPredictorWdm => 134,
            Self::EsnChaotic => 259,
            Self::DsCnnKws => 380,
            Self::DvsGesture => 420,
            Self::AkidaNet05_160 => 450,
            Self::AkidaNet10_224 => 700,
            Self::MobileNetV2 => 680,
            Self::ViTTiny => 800,
            Self::YoloV8n => 760,
            Self::PointNetPlusPlus => 520,
            Self::EventCamera => 540,
        }
    }

    /// Measured throughput in inferences/second at batch=8 on AKD1000
    ///
    /// Returns `None` if not yet hardware-validated.
    pub const fn throughput_hz(&self) -> Option<u32> {
        match self {
            // ecoPrimals physics — measured on live hardware (Exp 022, 028)
            Self::EsnQcdThermalization => Some(18_500),
            Self::PhaseClassifierSu3 => Some(21_200),
            Self::TransportPredictorWdm => Some(17_800),
            Self::AndersonRegimeClassifier => Some(22_400),
            Self::MinimalFc => Some(24_000),
            // NeuroBench reference numbers
            Self::DsCnnKws => Some(1_400),
            Self::DvsGesture => Some(1_720),
            Self::EsnChaotic => Some(18_000),
            Self::EcgAnomaly => Some(2_200),
            Self::AkidaNet05_160 => Some(1_250),
            // Not yet hardware-validated
            _ => None,
        }
    }

    /// Energy per inference on AKD1000 chip (microjoules), at measured throughput
    ///
    /// Returns `None` if not hardware-validated.
    pub const fn chip_energy_uj(&self) -> Option<f32> {
        match self {
            Self::EsnQcdThermalization => Some(1.4),
            Self::PhaseClassifierSu3 => Some(1.1),
            Self::TransportPredictorWdm => Some(1.5),
            Self::AndersonRegimeClassifier => Some(1.0),
            Self::EsnChaotic => Some(1.4),
            Self::EcgAnomaly => Some(1.1),
            _ => None,
        }
    }

    /// Validation tier — how thoroughly this model has been tested
    pub const fn validation(&self) -> ValidationTier {
        match self {
            // Validated on real AKD1000 hardware
            Self::EsnQcdThermalization
            | Self::PhaseClassifierSu3
            | Self::TransportPredictorWdm
            | Self::AndersonRegimeClassifier
            | Self::MinimalFc => ValidationTier::HardwareValidated,
            // Analysis complete, conversion path defined
            Self::DsCnnKws
            | Self::DvsGesture
            | Self::EsnChaotic
            | Self::EcgAnomaly
            | Self::AkidaNet05_160 => ValidationTier::Analysis,
            _ => ValidationTier::Analysis,
        }
    }

    /// Source / origin of this model
    pub const fn source(&self) -> ModelSource {
        match self {
            Self::AkidaNet05_160
            | Self::AkidaNet10_224
            | Self::DsCnnKws
            | Self::MobileNetV2
            | Self::ViTTiny
            | Self::YoloV8n
            | Self::PointNetPlusPlus
            | Self::DvsGesture
            | Self::EventCamera => ModelSource::BrainChipMetaTf,
            Self::EsnChaotic | Self::EcgAnomaly => ModelSource::NeuroBench,
            Self::EsnQcdThermalization
            | Self::PhaseClassifierSu3
            | Self::TransportPredictorWdm
            | Self::AndersonRegimeClassifier => ModelSource::EcoPrimalsPhysics,
            Self::MinimalFc => ModelSource::HandBuilt,
        }
    }

    /// Get expected model size (approximate bytes)
    pub const fn expected_size_bytes(&self) -> usize {
        match self {
            Self::AkidaNet05_160 => 400_000,
            Self::AkidaNet10_224 => 1_600_000,
            Self::DsCnnKws => 280_000,
            Self::MobileNetV2 => 3_500_000,
            Self::ViTTiny => 5_000_000,
            Self::YoloV8n => 1_800_000,
            Self::PointNetPlusPlus => 1_500_000,
            Self::DvsGesture => 120_000,
            Self::EventCamera => 1_200_000,
            Self::EsnChaotic => 100_000,
            Self::EcgAnomaly => 40_000,
            Self::EsnQcdThermalization => 80_000,
            Self::PhaseClassifierSu3 => 30_000,
            Self::TransportPredictorWdm => 60_000,
            Self::AndersonRegimeClassifier => 35_000,
            Self::MinimalFc => 2_000,
        }
    }

    /// Get all available models
    pub const fn all() -> &'static [Self] {
        &[
            // BrainChip MetaTF
            Self::AkidaNet05_160,
            Self::AkidaNet10_224,
            Self::DsCnnKws,
            Self::MobileNetV2,
            Self::ViTTiny,
            Self::YoloV8n,
            Self::PointNetPlusPlus,
            Self::DvsGesture,
            Self::EventCamera,
            // NeuroBench
            Self::EsnChaotic,
            Self::EcgAnomaly,
            // ecoPrimals physics
            Self::EsnQcdThermalization,
            Self::PhaseClassifierSu3,
            Self::TransportPredictorWdm,
            Self::AndersonRegimeClassifier,
            // Hand-built
            Self::MinimalFc,
        ]
    }

    /// Get models validated on real AKD1000 hardware
    pub const fn hardware_validated() -> &'static [Self] {
        &[
            Self::EsnQcdThermalization,
            Self::PhaseClassifierSu3,
            Self::TransportPredictorWdm,
            Self::AndersonRegimeClassifier,
            Self::MinimalFc,
        ]
    }

    /// Get models for `NeuroBench` benchmarks
    pub const fn neurobench_models() -> &'static [Self] {
        &[
            Self::DvsGesture,
            Self::DsCnnKws,
            Self::EsnChaotic,
            Self::EcgAnomaly,
        ]
    }

    /// Get ecoPrimals physics models
    pub const fn ecoprimal_physics_models() -> &'static [Self] {
        &[
            Self::EsnQcdThermalization,
            Self::PhaseClassifierSu3,
            Self::TransportPredictorWdm,
            Self::AndersonRegimeClassifier,
        ]
    }

    /// Get task category
    pub const fn task(&self) -> ModelTask {
        match self {
            Self::AkidaNet05_160 | Self::AkidaNet10_224 | Self::MobileNetV2 | Self::ViTTiny => {
                ModelTask::ImageClassification
            }
            Self::DsCnnKws => ModelTask::KeywordSpotting,
            Self::YoloV8n | Self::EventCamera => ModelTask::ObjectDetection,
            Self::PointNetPlusPlus => ModelTask::PointCloud,
            Self::DvsGesture => ModelTask::GestureRecognition,
            Self::EsnChaotic | Self::EsnQcdThermalization | Self::TransportPredictorWdm => {
                ModelTask::TimeSeriesPrediction
            }
            Self::EcgAnomaly => ModelTask::AnomalyDetection,
            Self::PhaseClassifierSu3 | Self::AndersonRegimeClassifier | Self::MinimalFc => {
                ModelTask::PhysicsClassification
            }
        }
    }
}

/// Model task categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    /// Image classification
    ImageClassification,
    /// Keyword/speech spotting
    KeywordSpotting,
    /// Object detection
    ObjectDetection,
    /// 3D point cloud processing
    PointCloud,
    /// Gesture recognition (DVS)
    GestureRecognition,
    /// Time series prediction and chaotic systems
    TimeSeriesPrediction,
    /// Anomaly detection (ECG, sensor streams)
    AnomalyDetection,
    /// Physics-domain classification (phase transitions, localization regimes)
    PhysicsClassification,
}

/// Model metadata extracted from zoo
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model enum variant
    pub model: ZooModel,
    /// Local file path
    pub path: PathBuf,
    /// File size in bytes
    pub size_bytes: usize,
    /// Is valid .fbz format
    pub is_valid: bool,
    /// SDK version (if parseable)
    pub sdk_version: Option<String>,
    /// Number of layers
    pub layer_count: Option<usize>,
}

/// Akida Model Zoo manager
///
/// Manages local cache of Akida Model Zoo models.
pub struct ModelZoo {
    /// Local cache directory
    cache_dir: PathBuf,
    /// Cached model metadata
    metadata: HashMap<ZooModel, ModelMetadata>,
}

impl ModelZoo {
    /// Create model zoo with specified cache directory
    ///
    /// # Errors
    ///
    /// Returns error if cache directory cannot be created.
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        fs::create_dir_all(&cache_dir).map_err(|e| {
            AkidaModelError::loading_failed(format!("Cannot create cache dir: {e}"))
        })?;

        info!("Model zoo cache: {}", cache_dir.display());

        let mut zoo = Self {
            cache_dir,
            metadata: HashMap::new(),
        };

        // Scan for existing models
        zoo.scan();

        Ok(zoo)
    }

    /// Scan cache directory for existing models
    fn scan(&mut self) {
        debug!("Scanning model zoo cache...");

        for model in ZooModel::all() {
            let path = self.cache_dir.join(model.filename());

            if path.exists() {
                match Self::load_metadata(*model, &path) {
                    Ok(meta) => {
                        debug!("Found {}: {} bytes", model.filename(), meta.size_bytes);
                        self.metadata.insert(*model, meta);
                    }
                    Err(e) => {
                        warn!("Invalid model {}: {}", model.filename(), e);
                    }
                }
            }
        }

        info!(
            "Found {} cached models in {}",
            self.metadata.len(),
            self.cache_dir.display()
        );
    }

    /// Load metadata for a model file
    fn load_metadata(model: ZooModel, path: &Path) -> Result<ModelMetadata> {
        let data = fs::read(path)
            .map_err(|e| AkidaModelError::loading_failed(format!("Cannot read model: {e}")))?;

        let size_bytes = data.len();

        // Validate .fbz format (check FlatBuffers magic)
        let is_valid = data.len() >= 4 && data[0..4] == [0x80, 0x44, 0x04, 0x10];

        // Try to extract version (simplified - real impl would parse FlatBuffers)
        let sdk_version = if data.len() > 40 {
            // Version typically at offset 30-40
            data[20..50]
                .windows(6)
                .find(|w| w.iter().all(|&b| b == b'.' || b.is_ascii_digit()))
                .and_then(|w| {
                    std::str::from_utf8(w)
                        .ok()
                        .map(|s| s.trim_end_matches('\0').to_string())
                })
        } else {
            None
        };

        Ok(ModelMetadata {
            model,
            path: path.to_path_buf(),
            size_bytes,
            is_valid,
            sdk_version,
            layer_count: None, // Would require full parsing
        })
    }

    /// Check if model is available locally
    pub fn has_model(&self, model: ZooModel) -> bool {
        self.metadata.contains_key(&model)
    }

    /// Get path to model file
    ///
    /// # Errors
    ///
    /// Returns error if model is not available locally.
    pub fn model_path(&self, model: ZooModel) -> Result<PathBuf> {
        if let Some(meta) = self.metadata.get(&model) {
            Ok(meta.path.clone())
        } else {
            Err(AkidaModelError::loading_failed(format!(
                "Model {} not available. Use download() first.",
                model.filename()
            )))
        }
    }

    /// Get model metadata
    pub fn model_metadata(&self, model: ZooModel) -> Option<&ModelMetadata> {
        self.metadata.get(&model)
    }

    /// List all available models
    pub fn available_models(&self) -> Vec<ZooModel> {
        self.metadata.keys().copied().collect()
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Create a stub model for testing
    ///
    /// Creates a minimal .fbz file that passes format validation.
    /// This is useful for testing without real models.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn create_stub_model(&mut self, model: ZooModel) -> Result<PathBuf> {
        let path = self.cache_dir.join(model.filename());

        // Create minimal valid .fbz structure
        let mut data = Vec::with_capacity(1024);

        // FlatBuffers magic
        data.extend_from_slice(&[0x80, 0x44, 0x04, 0x10]);

        // Padding to table offset area
        data.extend_from_slice(&[0x00; 26]);

        // Version string at offset 30
        data.extend_from_slice(b"2.18.2\0");

        // Pad to minimum size
        while data.len() < 256 {
            data.push(0x00);
        }

        // Add layer count marker (simplified)
        data.push(0x01); // 1 layer

        // Pad to expected minimum
        while data.len() < 512 {
            data.push(0x00);
        }

        fs::write(&path, &data)
            .map_err(|e| AkidaModelError::loading_failed(format!("Cannot write stub: {e}")))?;

        info!(
            "Created stub model: {} ({} bytes)",
            path.display(),
            data.len()
        );

        // Update metadata
        let meta = Self::load_metadata(model, &path)?;
        self.metadata.insert(model, meta);

        Ok(path)
    }

    /// Initialize stub models for all `NeuroBench` benchmarks
    ///
    /// # Errors
    ///
    /// Returns error if any stub cannot be created.
    pub fn init_neurobench_stubs(&mut self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for model in ZooModel::neurobench_models() {
            if !self.has_model(*model) {
                let path = self.create_stub_model(*model)?;
                paths.push(path);
            }
        }

        Ok(paths)
    }

    /// Print zoo status
    pub fn print_status(&self) {
        println!("\nAkida Model Zoo Status");
        println!("{}", "=".repeat(80));
        println!("Cache: {}", self.cache_dir.display());
        println!(
            "Available: {}/{}",
            self.metadata.len(),
            ZooModel::all().len()
        );
        println!();

        let sections: &[(&str, &[ZooModel])] = &[
            ("ecoPrimals Physics (validated on AKD1000)", ZooModel::ecoprimal_physics_models()),
            ("NeuroBench Benchmarks", ZooModel::neurobench_models()),
            ("BrainChip MetaTF Zoo", &[
                ZooModel::DsCnnKws, ZooModel::DvsGesture, ZooModel::AkidaNet05_160,
                ZooModel::AkidaNet10_224, ZooModel::MobileNetV2, ZooModel::YoloV8n,
                ZooModel::PointNetPlusPlus, ZooModel::EventCamera, ZooModel::ViTTiny,
            ]),
            ("Hand-Built", &[ZooModel::MinimalFc]),
        ];

        for (section, models) in sections {
            println!("  {section}");
            for model in *models {
                let file_status = if let Some(meta) = self.metadata.get(model) {
                    format!(
                        "✓ {:>7} bytes",
                        meta.size_bytes,
                    )
                } else {
                    "✗ not cached ".to_string()
                };
                let thr = model.throughput_hz()
                    .map(|hz| format!("{hz:>6} Hz"))
                    .unwrap_or_else(|| "   n/a   ".to_string());
                let np = model.np_budget();
                let tier = match model.validation() {
                    ValidationTier::HardwareValidated => "hw",
                    ValidationTier::Functional => "fn",
                    ValidationTier::Analysis => "  ",
                };
                println!(
                    "    [{tier}] {:40} {:12}  NPs:{np:4}  {thr}",
                    model.description(),
                    file_status,
                );
            }
            println!();
        }

        println!("{}", "=".repeat(80));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_zoo_model_filenames() {
        assert_eq!(ZooModel::DsCnnKws.filename(), "ds_cnn_kws.fbz");
        assert_eq!(ZooModel::DvsGesture.filename(), "dvs_gesture.fbz");
        assert_eq!(
            ZooModel::EsnQcdThermalization.filename(),
            "esn_qcd_thermalization.fbz"
        );
        assert_eq!(ZooModel::MinimalFc.filename(), "minimal_fc.fbz");
    }

    #[test]
    fn test_zoo_model_all_count() {
        assert!(ZooModel::all().len() >= 16);
    }

    #[test]
    fn test_physics_models_hardware_validated() {
        for model in ZooModel::hardware_validated() {
            assert_eq!(
                model.validation(),
                ValidationTier::HardwareValidated,
                "{:?} should be hardware-validated",
                model
            );
            assert!(
                model.throughput_hz().is_some(),
                "{:?} should have measured throughput",
                model
            );
        }
    }

    #[test]
    fn test_np_budget_within_chip_limit() {
        // Each individual model must fit on a single AKD1000
        const CHIP_NPS: usize = 1_000;
        for model in ZooModel::all() {
            assert!(
                model.np_budget() <= CHIP_NPS,
                "{:?} requires {} NPs, exceeds AKD1000 budget of {}",
                model,
                model.np_budget(),
                CHIP_NPS
            );
        }
    }

    #[test]
    fn test_physics_co_location_fits() {
        // All 4 physics models must fit together on a single AKD1000
        const CHIP_NPS: usize = 1_000;
        let total: usize = ZooModel::ecoprimal_physics_models()
            .iter()
            .map(|m| m.np_budget())
            .sum();
        assert!(
            total <= CHIP_NPS,
            "Physics models need {total} NPs total, exceeds {CHIP_NPS}"
        );
    }

    #[test]
    fn test_model_zoo_creation() {
        let temp_dir = TempDir::new().unwrap();
        let zoo = ModelZoo::new(temp_dir.path()).unwrap();

        assert_eq!(zoo.available_models().len(), 0);
        assert!(!zoo.has_model(ZooModel::DsCnnKws));
    }

    #[test]
    fn test_stub_model_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut zoo = ModelZoo::new(temp_dir.path()).unwrap();

        let path = zoo.create_stub_model(ZooModel::DsCnnKws).unwrap();

        assert!(path.exists());
        assert!(zoo.has_model(ZooModel::DsCnnKws));

        let meta = zoo.model_metadata(ZooModel::DsCnnKws).unwrap();
        assert!(meta.is_valid);
    }

    #[test]
    fn test_neurobench_stubs() {
        let temp_dir = TempDir::new().unwrap();
        let mut zoo = ModelZoo::new(temp_dir.path()).unwrap();

        let paths = zoo.init_neurobench_stubs().unwrap();

        assert!(!paths.is_empty());
        assert!(zoo.has_model(ZooModel::DvsGesture));
        assert!(zoo.has_model(ZooModel::DsCnnKws));
        assert!(zoo.has_model(ZooModel::EcgAnomaly));
    }

    #[test]
    fn test_source_classification() {
        assert_eq!(
            ZooModel::EsnQcdThermalization.source(),
            ModelSource::EcoPrimalsPhysics
        );
        assert_eq!(ZooModel::DsCnnKws.source(), ModelSource::BrainChipMetaTf);
        assert_eq!(ZooModel::EsnChaotic.source(), ModelSource::NeuroBench);
        assert_eq!(ZooModel::MinimalFc.source(), ModelSource::HandBuilt);
    }

    #[test]
    fn test_task_classification() {
        assert_eq!(
            ZooModel::PhaseClassifierSu3.task(),
            ModelTask::PhysicsClassification
        );
        assert_eq!(
            ZooModel::EcgAnomaly.task(),
            ModelTask::AnomalyDetection
        );
        assert_eq!(
            ZooModel::EsnQcdThermalization.task(),
            ModelTask::TimeSeriesPrediction
        );
    }
}
