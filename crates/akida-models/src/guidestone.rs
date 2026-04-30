// SPDX-License-Identifier: AGPL-3.0-or-later

//! guideStone — self-leveling verification artifact for NPU compute.
//!
//! A guideStone run parses a set of `.fbz` models, validates their structure
//! against reference expectations, computes SHA-256 digests, benchmarks parse
//! throughput, and emits a graded report. The result anchors all subsequent
//! work on that hardware or software build to a known-good baseline.
//!
//! # Usage
//!
//! ```ignore
//! use akida_models::guidestone::GuideStone;
//!
//! let gs = GuideStone::new("baseCamp/zoo-artifacts");
//! let report = gs.run();
//! report.print();
//! std::process::exit(if report.passed() { 0 } else { 1 });
//! ```

use crate::zoo::ZooModel;
use crate::Model;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Grade assigned to a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    /// Check passed within tolerance.
    Pass,
    /// Check passed with warnings (e.g., missing optional data).
    Warn,
    /// Check failed.
    Fail,
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Warn => write!(f, "WARN"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Result of a single guideStone check.
#[derive(Debug, Clone)]
pub struct Check {
    /// Human-readable name.
    pub name: String,
    /// Grade.
    pub grade: Grade,
    /// Detail message.
    pub detail: String,
}

/// Result of parsing one model in the guideStone run.
#[derive(Debug, Clone)]
pub struct ModelResult {
    /// Which model.
    pub model: ZooModel,
    /// File found on disk.
    pub file_present: bool,
    /// SHA-256 hex digest of the raw `.fbz` file.
    pub sha256: Option<String>,
    /// Parse succeeded.
    pub parsed: bool,
    /// SDK version extracted.
    pub version: Option<String>,
    /// Layer count extracted.
    pub layer_count: Option<usize>,
    /// File size in bytes.
    pub file_size: u64,
    /// Decompressed (program) size in bytes.
    pub program_size: Option<usize>,
    /// Parse wall time.
    pub parse_time: Option<Duration>,
    /// Per-check grades.
    pub checks: Vec<Check>,
}

/// The full guideStone report.
#[derive(Debug)]
pub struct Report {
    /// Artifact directory that was validated.
    pub artifact_dir: PathBuf,
    /// Per-model results.
    pub models: Vec<ModelResult>,
    /// Aggregate parse throughput (bytes/sec).
    pub throughput_bytes_per_sec: f64,
    /// Total wall time for the run.
    pub total_time: Duration,
    /// The models that were included in this run.
    pub model_set: Vec<ZooModel>,
}

impl Report {
    /// Did every check pass (no FAILs)?
    #[must_use]
    pub fn passed(&self) -> bool {
        self.models
            .iter()
            .flat_map(|m| &m.checks)
            .all(|c| c.grade != Grade::Fail)
    }

    /// Count of checks by grade.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut pass = 0;
        let mut warn = 0;
        let mut fail = 0;
        for c in self.models.iter().flat_map(|m| &m.checks) {
            match c.grade {
                Grade::Pass => pass += 1,
                Grade::Warn => warn += 1,
                Grade::Fail => fail += 1,
            }
        }
        (pass, warn, fail)
    }

    /// Print the report to stdout.
    pub fn print(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║              rustChip guideStone Report                     ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("  Artifact dir : {}", self.artifact_dir.display());
        println!("  Models       : {}", self.model_set.len());
        println!("  Total time   : {:.2?}", self.total_time);
        println!(
            "  Throughput   : {:.1} MB/s",
            self.throughput_bytes_per_sec / 1_048_576.0
        );
        println!();

        for mr in &self.models {
            let status = if !mr.file_present {
                "MISSING"
            } else if mr.parsed {
                "OK"
            } else {
                "FAIL"
            };
            println!(
                "  [{:>7}] {:<45} {}",
                status,
                mr.model.description().chars().take(45).collect::<String>(),
                mr.model.filename(),
            );

            if let Some(ref sha) = mr.sha256 {
                println!("            sha256: {sha}");
            }
            if let Some(v) = &mr.version {
                println!(
                    "            version: {v}  layers: {}  program: {} bytes  parse: {:.2?}",
                    mr.layer_count.unwrap_or(0),
                    mr.program_size.unwrap_or(0),
                    mr.parse_time.unwrap_or_default(),
                );
            }
            for check in &mr.checks {
                let icon = match check.grade {
                    Grade::Pass => "✓",
                    Grade::Warn => "⚠",
                    Grade::Fail => "✗",
                };
                println!("            {icon} {} — {}", check.name, check.detail);
            }
            println!();
        }

        let (pass, warn, fail) = self.counts();
        println!("  ────────────────────────────────────────────────────────");
        println!(
            "  Total checks : {}  ( {} PASS, {} WARN, {} FAIL )",
            pass + warn + fail,
            pass,
            warn,
            fail,
        );
        println!();

        if self.passed() {
            println!("  ██ guideStone: ALL CHECKS PASSED ██");
            println!();
            println!("  This substrate is anchored. Subsequent work on this build");
            println!("  can reference this guideStone run as the baseline.");
        } else {
            println!("  ██ guideStone: {} CHECK(S) FAILED ██", fail);
            println!();
            println!("  This substrate is NOT anchored. Fix the failures above");
            println!("  before referencing this run as a baseline.");
        }
        println!();
    }
}

/// The guideStone runner.
pub struct GuideStone {
    artifact_dir: PathBuf,
}

impl GuideStone {
    /// Create a new guideStone targeting the given artifact directory.
    #[must_use]
    pub fn new<P: AsRef<Path>>(artifact_dir: P) -> Self {
        Self {
            artifact_dir: artifact_dir.as_ref().to_path_buf(),
        }
    }

    /// The guideStone model subset: all BrainChip zoo models plus physics
    /// models. These are the models the guidestone validates — a cross-section
    /// of the full zoo covering every architecture class.
    #[must_use]
    pub fn model_set() -> Vec<ZooModel> {
        let mut set = Vec::new();
        set.extend_from_slice(ZooModel::brainchip_zoo());
        set.extend_from_slice(ZooModel::ecoprimal_physics_models());
        set
    }

    /// Run the full guideStone validation.
    #[must_use]
    pub fn run(&self) -> Report {
        let model_set = Self::model_set();
        let run_start = Instant::now();
        let mut results = Vec::with_capacity(model_set.len());
        let mut total_bytes: u64 = 0;
        let mut total_parse_time = Duration::ZERO;

        for model in &model_set {
            let mr = self.validate_model(*model);
            if mr.file_present {
                total_bytes += mr.file_size;
            }
            if let Some(pt) = mr.parse_time {
                total_parse_time += pt;
            }
            results.push(mr);
        }

        let throughput = if total_parse_time.as_secs_f64() > 0.0 {
            total_bytes as f64 / total_parse_time.as_secs_f64()
        } else {
            0.0
        };

        Report {
            artifact_dir: self.artifact_dir.clone(),
            models: results,
            throughput_bytes_per_sec: throughput,
            total_time: run_start.elapsed(),
            model_set,
        }
    }

    fn validate_model(&self, model: ZooModel) -> ModelResult {
        let path = self.artifact_dir.join(model.filename());
        let mut checks = Vec::new();

        // Check 1: file exists
        if !path.exists() {
            checks.push(Check {
                name: "file_present".into(),
                grade: Grade::Fail,
                detail: format!("{} not found", model.filename()),
            });
            return ModelResult {
                model,
                file_present: false,
                sha256: None,
                parsed: false,
                version: None,
                layer_count: None,
                file_size: 0,
                program_size: None,
                parse_time: None,
                checks,
            };
        }
        checks.push(Check {
            name: "file_present".into(),
            grade: Grade::Pass,
            detail: format!("{} found", model.filename()),
        });

        // Read raw bytes for hash + parse
        let raw = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                checks.push(Check {
                    name: "file_readable".into(),
                    grade: Grade::Fail,
                    detail: format!("read error: {e}"),
                });
                return ModelResult {
                    model,
                    file_present: true,
                    sha256: None,
                    parsed: false,
                    version: None,
                    layer_count: None,
                    file_size: 0,
                    program_size: None,
                    parse_time: None,
                    checks,
                };
            }
        };

        let file_size = raw.len() as u64;

        // Check 2: SHA-256
        let sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(&raw);
            format!("{:x}", hasher.finalize())
        };

        checks.push(Check {
            name: "sha256".into(),
            grade: Grade::Pass,
            detail: sha256.clone(),
        });

        // Check 3: file size sanity
        let expected = model.expected_size_bytes();
        let size_ratio = file_size as f64 / expected as f64;
        if (0.5..=2.0).contains(&size_ratio) {
            checks.push(Check {
                name: "file_size".into(),
                grade: Grade::Pass,
                detail: format!(
                    "{file_size} bytes (expected ~{expected}, ratio {size_ratio:.2})"
                ),
            });
        } else if file_size > 0 {
            checks.push(Check {
                name: "file_size".into(),
                grade: Grade::Warn,
                detail: format!(
                    "{file_size} bytes (expected ~{expected}, ratio {size_ratio:.2} — unusual)"
                ),
            });
        } else {
            checks.push(Check {
                name: "file_size".into(),
                grade: Grade::Fail,
                detail: "empty file".into(),
            });
        }

        // Check 4: parse
        let parse_start = Instant::now();
        let parse_result = Model::from_bytes(&raw);
        let parse_time = parse_start.elapsed();

        match parse_result {
            Ok(parsed) => {
                let version = parsed.version().to_string();
                let layer_count = parsed.layer_count();
                let program_size = parsed.program_size();

                checks.push(Check {
                    name: "parse".into(),
                    grade: Grade::Pass,
                    detail: format!("OK in {parse_time:.2?}"),
                });

                // Check 5: version non-empty
                if version.is_empty() {
                    checks.push(Check {
                        name: "version".into(),
                        grade: Grade::Warn,
                        detail: "empty version string".into(),
                    });
                } else {
                    checks.push(Check {
                        name: "version".into(),
                        grade: Grade::Pass,
                        detail: version.clone(),
                    });
                }

                // Check 6: layer count > 0
                if layer_count > 0 {
                    checks.push(Check {
                        name: "layers".into(),
                        grade: Grade::Pass,
                        detail: format!("{layer_count} layers"),
                    });
                } else {
                    checks.push(Check {
                        name: "layers".into(),
                        grade: Grade::Warn,
                        detail: "0 layers extracted (heuristic may miss small models)".into(),
                    });
                }

                // Check 7: decompression ratio sane
                if file_size > 0 {
                    let ratio = program_size as f64 / file_size as f64;
                    if ratio >= 0.5 {
                        checks.push(Check {
                            name: "decompress_ratio".into(),
                            grade: Grade::Pass,
                            detail: format!("{ratio:.2}x ({program_size} / {file_size})"),
                        });
                    } else {
                        checks.push(Check {
                            name: "decompress_ratio".into(),
                            grade: Grade::Warn,
                            detail: format!(
                                "{ratio:.2}x — unexpectedly small decompressed output"
                            ),
                        });
                    }
                }

                // Check 8: weights present
                let weight_count = parsed.total_weight_count();
                if weight_count > 0 || layer_count == 0 {
                    checks.push(Check {
                        name: "weights".into(),
                        grade: Grade::Pass,
                        detail: format!("{weight_count} weight values in {} blocks", parsed.weights().len()),
                    });
                } else {
                    checks.push(Check {
                        name: "weights".into(),
                        grade: Grade::Warn,
                        detail: "no weight blocks found".into(),
                    });
                }

                // Check 9: NP budget fits AKD1000
                let np = model.np_budget();
                if np <= 1000 {
                    checks.push(Check {
                        name: "np_budget".into(),
                        grade: Grade::Pass,
                        detail: format!("{np} NPs (≤ 1000 AKD1000 limit)"),
                    });
                } else {
                    checks.push(Check {
                        name: "np_budget".into(),
                        grade: Grade::Fail,
                        detail: format!("{np} NPs — exceeds AKD1000 limit of 1000"),
                    });
                }

                ModelResult {
                    model,
                    file_present: true,
                    sha256: Some(sha256),
                    parsed: true,
                    version: Some(version),
                    layer_count: Some(layer_count),
                    file_size,
                    program_size: Some(program_size),
                    parse_time: Some(parse_time),
                    checks,
                }
            }
            Err(e) => {
                checks.push(Check {
                    name: "parse".into(),
                    grade: Grade::Fail,
                    detail: format!("parse error: {e}"),
                });

                ModelResult {
                    model,
                    file_present: true,
                    sha256: Some(sha256),
                    parsed: false,
                    version: None,
                    layer_count: None,
                    file_size,
                    program_size: None,
                    parse_time: Some(parse_time),
                    checks,
                }
            }
        }
    }
}
