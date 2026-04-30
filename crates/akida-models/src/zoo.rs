// SPDX-License-Identifier: AGPL-3.0-or-later

//! Akida Model Zoo Manager
//!
//! Provides utilities for downloading, converting, and managing
//! models from the `BrainChip` Akida Model Zoo.
//! References to hotSpring in model descriptions are ecosystem context — not a runtime dependency.
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
    /// `BrainChip` `MetaTF` official zoo
    BrainChipMetaTf,
    /// `NeuroBench` benchmark suite
    NeuroBench,
    /// ecoPrimals physics models (validated on live AKD1000)
    EcoPrimalsPhysics,
    /// Hand-built via `program_external()`
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
    // ── BrainChip MetaTF (exported via scripts/export_zoo.py) ───────────────
    /// `AkidaNet` `ImageNet` classifier (1.0 width, 224×224)
    AkidaNetImagenet,
    /// `AkidaNet` 18-layer `ImageNet` classifier (224×224)
    AkidaNet18Imagenet,
    /// `AkidaNet` PlantVillage disease classifier
    AkidaNetPlantvillage,
    /// `AkidaNet` Visual Wake Words (96×96)
    AkidaNetVww,
    /// `AkidaNet` face identification
    AkidaNetFaceId,
    /// `Akida` U-Net portrait segmentation (128×128)
    AkidaUnetPortrait128,
    /// `CenterNet` VOC object detection
    CenterNetVoc,
    /// ConvTiny DVS gesture recognition (11 classes)
    ConvtinyGesture,
    /// ConvTiny Samsung Handy gesture recognition
    ConvtinyHandySamsung,
    /// DS-CNN for keyword spotting (35 words, Google Speech Commands v2)
    DsCnnKws,
    /// GXNOR MNIST digit classification
    GxnorMnist,
    /// `MobileNet` `ImageNet` classifier
    MobileNetImagenet,
    /// `PointNet++` ModelNet40 3D point cloud classification
    PointNetPlusModelnet40,
    /// TENN recurrent speech commands (12 classes)
    TennRecurrentSc12,
    /// TENN recurrent UORED
    TennRecurrentUored,
    /// TENN spatiotemporal DVS128
    TennSpatiotemporalDvs128,
    /// TENN spatiotemporal eye tracking
    TennSpatiotemporalEye,
    /// TENN spatiotemporal Jester gesture
    TennSpatiotemporalJester,
    /// VGG UTK face age regression
    VggUtkFace,
    /// YOLO VOC object detection
    YoloVoc,
    /// YOLO WiderFace detection
    YoloWiderface,

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
    /// Minimal FC(50→1) smoke test for `program_external()` validation
    MinimalFc,
}

impl ZooModel {
    /// Get model filename (matches output of `scripts/export_zoo.py`)
    pub const fn filename(&self) -> &'static str {
        match self {
            Self::AkidaNetImagenet => "akidanet_imagenet.fbz",
            Self::AkidaNet18Imagenet => "akidanet18_imagenet.fbz",
            Self::AkidaNetPlantvillage => "akidanet_plantvillage.fbz",
            Self::AkidaNetVww => "akidanet_vww.fbz",
            Self::AkidaNetFaceId => "akidanet_faceidentification.fbz",
            Self::AkidaUnetPortrait128 => "akida_unet_portrait128.fbz",
            Self::CenterNetVoc => "centernet_voc.fbz",
            Self::ConvtinyGesture => "convtiny_gesture.fbz",
            Self::ConvtinyHandySamsung => "convtiny_handy_samsung.fbz",
            Self::DsCnnKws => "ds_cnn_kws.fbz",
            Self::GxnorMnist => "gxnor_mnist.fbz",
            Self::MobileNetImagenet => "mobilenet_imagenet.fbz",
            Self::PointNetPlusModelnet40 => "pointnet_plus_modelnet40.fbz",
            Self::TennRecurrentSc12 => "tenn_recurrent_sc12.fbz",
            Self::TennRecurrentUored => "tenn_recurrent_uored.fbz",
            Self::TennSpatiotemporalDvs128 => "tenn_spatiotemporal_dvs128.fbz",
            Self::TennSpatiotemporalEye => "tenn_spatiotemporal_eye.fbz",
            Self::TennSpatiotemporalJester => "tenn_spatiotemporal_jester.fbz",
            Self::VggUtkFace => "vgg_utk_face.fbz",
            Self::YoloVoc => "yolo_voc.fbz",
            Self::YoloWiderface => "yolo_widerface.fbz",
            Self::EsnChaotic => "esn_chaotic.fbz",
            Self::EcgAnomaly => "ecg_anomaly.fbz",
            Self::EsnQcdThermalization => "esn_readout.fbz",
            Self::PhaseClassifierSu3 => "phase_classifier.fbz",
            Self::TransportPredictorWdm => "transport_predictor.fbz",
            Self::AndersonRegimeClassifier => "anderson_classifier.fbz",
            Self::MinimalFc => "minimal_fc.fbz",
        }
    }

    /// Get model description
    pub const fn description(&self) -> &'static str {
        match self {
            Self::AkidaNetImagenet => "AkidaNet ImageNet classifier (1.0 width, 224x224)",
            Self::AkidaNet18Imagenet => "AkidaNet18 ImageNet classifier (224x224)",
            Self::AkidaNetPlantvillage => "AkidaNet PlantVillage disease classifier (224x224)",
            Self::AkidaNetVww => "AkidaNet Visual Wake Words (96x96)",
            Self::AkidaNetFaceId => "AkidaNet face identification (112x96)",
            Self::AkidaUnetPortrait128 => "Akida U-Net portrait segmentation (128x128)",
            Self::CenterNetVoc => "CenterNet VOC object detection (384x384)",
            Self::ConvtinyGesture => "ConvTiny DVS gesture recognition (11 classes, 64x64)",
            Self::ConvtinyHandySamsung => "ConvTiny Samsung Handy gesture (120x160)",
            Self::DsCnnKws => "DS-CNN keyword spotting (35 words, Google Speech Commands v2)",
            Self::GxnorMnist => "GXNOR MNIST digit classification (28x28)",
            Self::MobileNetImagenet => "MobileNet ImageNet classifier (224x224)",
            Self::PointNetPlusModelnet40 => "PointNet++ ModelNet40 3D point cloud (8x256x3)",
            Self::TennRecurrentSc12 => "TENN recurrent speech commands (12 classes)",
            Self::TennRecurrentUored => "TENN recurrent UORED (4 classes)",
            Self::TennSpatiotemporalDvs128 => "TENN spatiotemporal DVS128 gesture (128x128x2)",
            Self::TennSpatiotemporalEye => "TENN spatiotemporal eye tracking (80x106x2)",
            Self::TennSpatiotemporalJester => "TENN spatiotemporal Jester gesture (100x100x3)",
            Self::VggUtkFace => "VGG UTK face age regression (32x32x3)",
            Self::YoloVoc => "YOLO AkidaNet VOC object detection (224x224)",
            Self::YoloWiderface => "YOLO AkidaNet WiderFace detection (224x224)",
            Self::EsnChaotic => "ESN chaotic time series prediction (MSLP, NeuroBench)",
            Self::EcgAnomaly => "ECG anomaly detection (MIT-BIH Arrhythmia, NeuroBench)",
            Self::EsnQcdThermalization => {
                "ESN readout: lattice QCD thermalization detector (hotSpring Exp 022)"
            }
            Self::PhaseClassifierSu3 => {
                "SU(3) phase classifier: confined/deconfined (hotSpring Exp 022)"
            }
            Self::TransportPredictorWdm => {
                "WDM transport predictor: D*, eta*, lambda* from plasma observables"
            }
            Self::AndersonRegimeClassifier => {
                "Anderson localization regime: loc/diff/critical (groundSpring Exp 028)"
            }
            Self::MinimalFc => "Minimal FC(50->1) smoke test for program_external() validation",
        }
    }

    /// NP budget required on AKD1000 (number of NPs consumed)
    ///
    /// AKD1000 has 1,000 NPs total. Sum of co-located models must be ≤ 1,000.
    /// Estimates for MetaTF zoo models based on architecture complexity.
    pub const fn np_budget(&self) -> usize {
        match self {
            Self::MinimalFc => 51,
            Self::PhaseClassifierSu3 => 67,
            Self::AndersonRegimeClassifier => 68,
            Self::EcgAnomaly => 96,
            Self::TransportPredictorWdm => 134,
            Self::EsnQcdThermalization => 179,
            Self::TennRecurrentUored => 120,
            Self::TennRecurrentSc12 => 180,
            Self::VggUtkFace => 200,
            Self::GxnorMnist => 220,
            Self::EsnChaotic => 259,
            Self::AkidaNetVww => 300,
            Self::ConvtinyGesture => 350,
            Self::ConvtinyHandySamsung => 350,
            Self::DsCnnKws => 380,
            Self::TennSpatiotemporalDvs128 => 400,
            Self::TennSpatiotemporalEye => 420,
            Self::PointNetPlusModelnet40 => 450,
            Self::AkidaNetPlantvillage => 500,
            Self::TennSpatiotemporalJester => 520,
            Self::AkidaUnetPortrait128 => 550,
            Self::AkidaNetFaceId => 600,
            Self::CenterNetVoc => 650,
            Self::MobileNetImagenet => 680,
            Self::AkidaNetImagenet => 700,
            Self::AkidaNet18Imagenet => 700,
            Self::YoloVoc => 760,
            Self::YoloWiderface => 760,
        }
    }

    /// Measured throughput in inferences/second at batch=8 on AKD1000
    ///
    /// Returns `None` if not yet hardware-validated.
    pub const fn throughput_hz(&self) -> Option<u32> {
        match self {
            Self::EsnQcdThermalization => Some(18_500),
            Self::PhaseClassifierSu3 => Some(21_200),
            Self::TransportPredictorWdm => Some(17_800),
            Self::AndersonRegimeClassifier => Some(22_400),
            Self::MinimalFc => Some(24_000),
            Self::DsCnnKws => Some(1_400),
            Self::EsnChaotic => Some(18_000),
            Self::EcgAnomaly => Some(2_200),
            _ => None,
        }
    }

    /// Energy per inference on AKD1000 chip (microjoules), at measured throughput
    ///
    /// Returns `None` if not hardware-validated.
    pub const fn chip_energy_uj(&self) -> Option<f32> {
        match self {
            Self::EsnQcdThermalization | Self::EsnChaotic => Some(1.4),
            Self::PhaseClassifierSu3 | Self::EcgAnomaly => Some(1.1),
            Self::TransportPredictorWdm => Some(1.5),
            Self::AndersonRegimeClassifier => Some(1.0),
            _ => None,
        }
    }

    /// Validation tier — how thoroughly this model has been tested
    pub const fn validation(&self) -> ValidationTier {
        match self {
            Self::EsnQcdThermalization
            | Self::PhaseClassifierSu3
            | Self::TransportPredictorWdm
            | Self::AndersonRegimeClassifier
            | Self::MinimalFc => ValidationTier::HardwareValidated,
            // All MetaTF zoo models: exported and parsed successfully
            Self::AkidaNetImagenet
            | Self::AkidaNet18Imagenet
            | Self::AkidaNetPlantvillage
            | Self::AkidaNetVww
            | Self::AkidaNetFaceId
            | Self::AkidaUnetPortrait128
            | Self::CenterNetVoc
            | Self::ConvtinyGesture
            | Self::ConvtinyHandySamsung
            | Self::DsCnnKws
            | Self::GxnorMnist
            | Self::MobileNetImagenet
            | Self::PointNetPlusModelnet40
            | Self::TennRecurrentSc12
            | Self::TennRecurrentUored
            | Self::TennSpatiotemporalDvs128
            | Self::TennSpatiotemporalEye
            | Self::TennSpatiotemporalJester
            | Self::VggUtkFace
            | Self::YoloVoc
            | Self::YoloWiderface => ValidationTier::Functional,
            _ => ValidationTier::Analysis,
        }
    }

    /// Source / origin of this model
    pub const fn source(&self) -> ModelSource {
        match self {
            Self::AkidaNetImagenet
            | Self::AkidaNet18Imagenet
            | Self::AkidaNetPlantvillage
            | Self::AkidaNetVww
            | Self::AkidaNetFaceId
            | Self::AkidaUnetPortrait128
            | Self::CenterNetVoc
            | Self::ConvtinyGesture
            | Self::ConvtinyHandySamsung
            | Self::DsCnnKws
            | Self::GxnorMnist
            | Self::MobileNetImagenet
            | Self::PointNetPlusModelnet40
            | Self::TennRecurrentSc12
            | Self::TennRecurrentUored
            | Self::TennSpatiotemporalDvs128
            | Self::TennSpatiotemporalEye
            | Self::TennSpatiotemporalJester
            | Self::VggUtkFace
            | Self::YoloVoc
            | Self::YoloWiderface => ModelSource::BrainChipMetaTf,
            Self::EsnChaotic | Self::EcgAnomaly => ModelSource::NeuroBench,
            Self::EsnQcdThermalization
            | Self::PhaseClassifierSu3
            | Self::TransportPredictorWdm
            | Self::AndersonRegimeClassifier => ModelSource::EcoPrimalsPhysics,
            Self::MinimalFc => ModelSource::HandBuilt,
        }
    }

    /// Get expected model size (approximate bytes, from actual exports)
    pub const fn expected_size_bytes(&self) -> usize {
        match self {
            Self::AkidaNetImagenet => 5_269_000,
            Self::AkidaNet18Imagenet => 2_827_000,
            Self::AkidaNetPlantvillage => 1_402_000,
            Self::AkidaNetVww => 304_000,
            Self::AkidaNetFaceId => 2_834_000,
            Self::AkidaUnetPortrait128 => 1_302_000,
            Self::CenterNetVoc => 2_864_000,
            Self::ConvtinyGesture => 174_000,
            Self::ConvtinyHandySamsung => 172_000,
            Self::DsCnnKws => 41_000,
            Self::GxnorMnist => 203_000,
            Self::MobileNetImagenet => 5_028_000,
            Self::PointNetPlusModelnet40 => 343_000,
            Self::TennRecurrentSc12 => 70_000,
            Self::TennRecurrentUored => 37_000,
            Self::TennSpatiotemporalDvs128 => 225_000,
            Self::TennSpatiotemporalEye => 275_000,
            Self::TennSpatiotemporalJester => 1_611_000,
            Self::VggUtkFace => 150_000,
            Self::YoloVoc => 4_368_000,
            Self::YoloWiderface => 4_239_000,
            Self::EsnChaotic => 100_000,
            Self::EcgAnomaly => 40_000,
            Self::EsnQcdThermalization => 107_000,
            Self::PhaseClassifierSu3 => 4_000,
            Self::TransportPredictorWdm => 8_000,
            Self::AndersonRegimeClassifier => 5_000,
            Self::MinimalFc => 2_000,
        }
    }

    /// Get all available models
    pub const fn all() -> &'static [Self] {
        &[
            // BrainChip MetaTF (21 models, exported via scripts/export_zoo.py)
            Self::AkidaNetImagenet,
            Self::AkidaNet18Imagenet,
            Self::AkidaNetPlantvillage,
            Self::AkidaNetVww,
            Self::AkidaNetFaceId,
            Self::AkidaUnetPortrait128,
            Self::CenterNetVoc,
            Self::ConvtinyGesture,
            Self::ConvtinyHandySamsung,
            Self::DsCnnKws,
            Self::GxnorMnist,
            Self::MobileNetImagenet,
            Self::PointNetPlusModelnet40,
            Self::TennRecurrentSc12,
            Self::TennRecurrentUored,
            Self::TennSpatiotemporalDvs128,
            Self::TennSpatiotemporalEye,
            Self::TennSpatiotemporalJester,
            Self::VggUtkFace,
            Self::YoloVoc,
            Self::YoloWiderface,
            // NeuroBench
            Self::EsnChaotic,
            Self::EcgAnomaly,
            // ecoPrimals physics (4 models, exported via scripts/export_physics.py)
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

    /// Get the 21 BrainChip MetaTF zoo models
    pub const fn brainchip_zoo() -> &'static [Self] {
        &[
            Self::AkidaNetImagenet,
            Self::AkidaNet18Imagenet,
            Self::AkidaNetPlantvillage,
            Self::AkidaNetVww,
            Self::AkidaNetFaceId,
            Self::AkidaUnetPortrait128,
            Self::CenterNetVoc,
            Self::ConvtinyGesture,
            Self::ConvtinyHandySamsung,
            Self::DsCnnKws,
            Self::GxnorMnist,
            Self::MobileNetImagenet,
            Self::PointNetPlusModelnet40,
            Self::TennRecurrentSc12,
            Self::TennRecurrentUored,
            Self::TennSpatiotemporalDvs128,
            Self::TennSpatiotemporalEye,
            Self::TennSpatiotemporalJester,
            Self::VggUtkFace,
            Self::YoloVoc,
            Self::YoloWiderface,
        ]
    }

    /// Get models for `NeuroBench` benchmarks
    pub const fn neurobench_models() -> &'static [Self] {
        &[Self::DsCnnKws, Self::EsnChaotic, Self::EcgAnomaly]
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
            Self::AkidaNetImagenet
            | Self::AkidaNet18Imagenet
            | Self::AkidaNetPlantvillage
            | Self::AkidaNetVww
            | Self::MobileNetImagenet
            | Self::GxnorMnist => ModelTask::ImageClassification,
            Self::AkidaNetFaceId | Self::VggUtkFace => ModelTask::FaceAnalysis,
            Self::AkidaUnetPortrait128 => ModelTask::Segmentation,
            Self::CenterNetVoc | Self::YoloVoc | Self::YoloWiderface => {
                ModelTask::ObjectDetection
            }
            Self::ConvtinyGesture
            | Self::ConvtinyHandySamsung
            | Self::TennSpatiotemporalDvs128
            | Self::TennSpatiotemporalJester => ModelTask::GestureRecognition,
            Self::DsCnnKws | Self::TennRecurrentSc12 | Self::TennRecurrentUored => {
                ModelTask::KeywordSpotting
            }
            Self::TennSpatiotemporalEye => ModelTask::EyeTracking,
            Self::PointNetPlusModelnet40 => ModelTask::PointCloud,
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
    /// Face identification or age regression
    FaceAnalysis,
    /// Semantic segmentation
    Segmentation,
    /// Keyword/speech spotting
    KeywordSpotting,
    /// Object detection
    ObjectDetection,
    /// 3D point cloud processing
    PointCloud,
    /// Gesture recognition (DVS / video)
    GestureRecognition,
    /// Eye tracking
    EyeTracking,
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

    /// Load metadata for a model file.
    ///
    /// Attempts a full parse via [`Model::from_bytes`] to determine validity,
    /// version, and layer count. Falls back to size-only metadata if parsing fails.
    fn load_metadata(model: ZooModel, path: &Path) -> Result<ModelMetadata> {
        let data = fs::read(path)
            .map_err(|e| AkidaModelError::loading_failed(format!("Cannot read model: {e}")))?;

        let size_bytes = data.len();

        match crate::Model::from_bytes(&data) {
            Ok(parsed) => Ok(ModelMetadata {
                model,
                path: path.to_path_buf(),
                size_bytes,
                is_valid: true,
                sdk_version: Some(parsed.version().to_string()),
                layer_count: Some(parsed.layer_count()),
            }),
            Err(e) => {
                debug!("Model {} failed to parse: {e}", model.filename());
                Ok(ModelMetadata {
                    model,
                    path: path.to_path_buf(),
                    size_bytes,
                    is_valid: false,
                    sdk_version: None,
                    layer_count: None,
                })
            }
        }
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
        self.metadata.get(&model).map_or_else(
            || {
                Err(AkidaModelError::loading_failed(format!(
                    "Model {} not available. Use download() first.",
                    model.filename()
                )))
            },
            |meta| Ok(meta.path.clone()),
        )
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

    /// Create a **reference** model artifact (minimal valid `.fbz`) for tooling and CI.
    ///
    /// This is the fallback path when no hardware or pre-built zoo artifact is present: it writes
    /// a minimal `FlatBuffers` payload that passes format validation so parsers and benches can run.
    /// It is not a substitute for hardware-validated models in production.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn create_reference_model(&mut self, model: ZooModel) -> Result<PathBuf> {
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

        fs::write(&path, &data).map_err(|e| {
            AkidaModelError::loading_failed(format!("Cannot write reference model: {e}"))
        })?;

        info!(
            "Created reference model: {} ({} bytes)",
            path.display(),
            data.len()
        );

        // Update metadata
        let meta = Self::load_metadata(model, &path)?;
        self.metadata.insert(model, meta);

        Ok(path)
    }

    /// Initialize reference models for all `NeuroBench` benchmarks (see [`Self::create_reference_model`]).
    ///
    /// # Errors
    ///
    /// Returns error if any reference artifact cannot be created.
    pub fn init_neurobench_stubs(&mut self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for model in ZooModel::neurobench_models() {
            if !self.has_model(*model) {
                let path = self.create_reference_model(*model)?;
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
            (
                "ecoPrimals Physics (validated on AKD1000)",
                ZooModel::ecoprimal_physics_models(),
            ),
            ("NeuroBench Benchmarks", ZooModel::neurobench_models()),
            ("BrainChip MetaTF Zoo", ZooModel::brainchip_zoo()),
            ("Hand-Built", &[ZooModel::MinimalFc]),
        ];

        for (section, models) in sections {
            println!("  {section}");
            for model in *models {
                let file_status = self.metadata.get(model).map_or_else(
                    || "✗ not cached ".to_string(),
                    |meta| format!("✓ {:>7} bytes", meta.size_bytes),
                );
                let thr = model
                    .throughput_hz()
                    .map_or_else(|| "   n/a   ".to_string(), |hz| format!("{hz:>6} Hz"));
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
        assert_eq!(ZooModel::ConvtinyGesture.filename(), "convtiny_gesture.fbz");
        assert_eq!(
            ZooModel::EsnQcdThermalization.filename(),
            "esn_readout.fbz"
        );
        assert_eq!(ZooModel::MinimalFc.filename(), "minimal_fc.fbz");
        assert_eq!(ZooModel::YoloVoc.filename(), "yolo_voc.fbz");
    }

    #[test]
    fn test_zoo_model_all_count() {
        assert!(ZooModel::all().len() >= 28);
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
    fn test_reference_model_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut zoo = ModelZoo::new(temp_dir.path()).unwrap();

        let path = zoo.create_reference_model(ZooModel::DsCnnKws).unwrap();

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
        assert_eq!(ZooModel::EcgAnomaly.task(), ModelTask::AnomalyDetection);
        assert_eq!(
            ZooModel::EsnQcdThermalization.task(),
            ModelTask::TimeSeriesPrediction
        );
    }

    #[test]
    fn validation_tier_ordering_places_hardware_above_analysis() {
        assert!(ValidationTier::HardwareValidated > ValidationTier::Analysis);
        assert!(ValidationTier::Functional > ValidationTier::Analysis);
    }

    #[test]
    fn model_path_errors_when_missing() {
        let temp_dir = TempDir::new().unwrap();
        let zoo = ModelZoo::new(temp_dir.path()).unwrap();
        let err = zoo.model_path(ZooModel::AkidaNetImagenet).unwrap_err();
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn all_zoo_models_have_nonzero_expected_size() {
        for m in ZooModel::all() {
            assert!(m.expected_size_bytes() > 0, "{m:?}");
        }
    }

    #[test]
    fn zoo_listings_are_consistent_lengths() {
        assert!(ZooModel::hardware_validated().len() <= ZooModel::all().len());
        assert!(ZooModel::neurobench_models().len() <= ZooModel::all().len());
        assert!(ZooModel::ecoprimal_physics_models().len() <= ZooModel::all().len());
    }

    #[test]
    fn every_zoo_model_exposes_nonempty_metadata() {
        for m in ZooModel::all() {
            assert!(!m.filename().is_empty());
            assert!(m.filename().ends_with(".fbz"));
            assert!(!m.description().is_empty());
            assert!(m.np_budget() > 0);
            let _ = m.source();
            let _ = m.task();
            let _ = m.validation();
            let _ = m.expected_size_bytes();
        }
    }

    #[test]
    fn validation_tier_functional_between_analysis_and_hardware() {
        assert!(ValidationTier::Analysis < ValidationTier::Functional);
        assert!(ValidationTier::Functional < ValidationTier::HardwareValidated);
    }

    #[test]
    fn model_zoo_init_neurobench_stubs_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let mut zoo = ModelZoo::new(temp_dir.path()).unwrap();
        let first = zoo.init_neurobench_stubs().unwrap();
        let second = zoo.init_neurobench_stubs().unwrap();
        assert!(!first.is_empty() || !second.is_empty());
        assert!(
            second.is_empty(),
            "second call should skip existing artifacts"
        );
    }

    #[test]
    fn model_zoo_indexes_invalid_fbz_then_reference_fixes_magic() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join(ZooModel::GxnorMnist.filename());
        fs::write(&path, b"not a flatbuffer").unwrap();
        let mut zoo = ModelZoo::new(temp_dir.path()).unwrap();
        assert!(zoo.has_model(ZooModel::GxnorMnist));
        assert!(!zoo.model_metadata(ZooModel::GxnorMnist).unwrap().is_valid);
        zoo.create_reference_model(ZooModel::GxnorMnist).unwrap();
        assert!(zoo.model_metadata(ZooModel::GxnorMnist).unwrap().is_valid);
    }

    #[test]
    fn model_zoo_cache_dir_accessor() {
        let temp_dir = TempDir::new().unwrap();
        let zoo = ModelZoo::new(temp_dir.path()).unwrap();
        assert_eq!(zoo.cache_dir(), temp_dir.path());
    }

    #[test]
    fn print_status_smoke_output() {
        let temp_dir = TempDir::new().unwrap();
        let zoo = ModelZoo::new(temp_dir.path()).unwrap();
        zoo.print_status();
    }
}
