// SPDX-License-Identifier: AGPL-3.0-or-later

//! Online weight evolution via direct SRAM mutation.
//!
//! Leverages `NpuBackend::mutate_weights()` for zero-DMA weight updates,
//! enabling 136+ generations/sec evolutionary weight optimization on-chip.
//!
//! # Architecture
//!
//! ```text
//! NpuEvolver
//!   ├── base weights (loaded via DMA once)
//!   ├── population of WeightPatch variants
//!   └── fitness evaluation loop:
//!       1. Apply patch to SRAM (mutate_weights)
//!       2. Run inference (infer)
//!       3. Evaluate fitness (FitnessEvaluator)
//!       4. Select best, breed, mutate
//! ```
//!
//! # Performance
//!
//! Weight mutation via `set_variable()`: ~86 µs optimal (Discovery 6).
//! At this rate, a population of 10 can complete ~136 gen/sec including
//! inference (54 µs) and fitness evaluation.
//!
//! # Status
//!
//! Scaffolded — the evolution loop and fitness evaluation interface are
//! defined. Requires a loaded model and `NpuBackend` with SRAM support
//! for production use.

use crate::backend::NpuBackend;
use crate::error::Result;

/// Online weight evolver for NPU-accelerated evolutionary optimization.
///
/// Manages a population of weight patches and iterates through
/// mutation → evaluation → selection cycles using direct SRAM access.
pub struct NpuEvolver<F: FitnessEvaluator> {
    config: EvolutionConfig,
    population: Vec<Individual>,
    generation: u64,
    best_fitness: f64,
    fitness_evaluator: F,
}

impl<F: FitnessEvaluator> NpuEvolver<F> {
    /// Create a new evolver with the given configuration and fitness function.
    #[must_use]
    pub fn new(config: EvolutionConfig, fitness_evaluator: F) -> Self {
        let population = (0..config.population_size)
            .map(|i| Individual {
                _id: i,
                patches: Vec::new(),
                fitness: f64::NEG_INFINITY,
            })
            .collect();

        Self {
            config,
            population,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            fitness_evaluator,
        }
    }

    /// Run one evolution step: mutate, evaluate, select.
    ///
    /// For each individual in the population:
    /// 1. Apply its weight patches to SRAM
    /// 2. Run inference on the evaluation input
    /// 3. Compute fitness
    /// 4. After all evaluated, select and breed the next generation
    ///
    /// # Errors
    ///
    /// Returns error if weight mutation or inference fails.
    pub fn evolve_step(
        &mut self,
        backend: &mut dyn NpuBackend,
        eval_input: &[f32],
    ) -> Result<EvolutionResult> {
        let mut fitnesses = Vec::with_capacity(self.population.len());

        for individual in &self.population {
            // Apply weight patches
            for patch in &individual.patches {
                backend.mutate_weights(patch.offset, &patch.data)?;
            }

            // Run inference
            let output = backend.infer(eval_input)?;

            // Evaluate fitness
            let fitness = self.fitness_evaluator.evaluate(eval_input, &output);
            fitnesses.push(fitness);
        }

        // Update fitness values
        for (individual, &fitness) in self.population.iter_mut().zip(fitnesses.iter()) {
            individual.fitness = fitness;
        }

        // Select and breed
        self.select_and_breed();

        self.generation += 1;
        let best = self.best_fitness;

        Ok(EvolutionResult {
            generation: self.generation,
            best_fitness: best,
            mean_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            population_size: self.population.len(),
        })
    }

    /// Current generation number.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Best fitness achieved so far.
    #[must_use]
    pub const fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Get the best individual's weight patches.
    #[must_use]
    pub fn best_patches(&self) -> Option<&[WeightPatch]> {
        self.population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|ind| ind.patches.as_slice())
    }

    fn select_and_breed(&mut self) {
        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(best) = self.population.first()
            && best.fitness > self.best_fitness
        {
            self.best_fitness = best.fitness;
        }

        // Tournament selection: keep top half, mutate to fill bottom half
        let survivors = self.config.population_size / 2;
        let survivor_patches: Vec<Vec<WeightPatch>> = self.population[..survivors]
            .iter()
            .map(|ind| ind.patches.clone())
            .collect();

        for i in survivors..self.population.len() {
            let parent_idx = i % survivors;
            self.population[i].patches = mutate_patches(
                &survivor_patches[parent_idx],
                self.config.mutation_rate,
                self.config.mutation_magnitude,
            );
            self.population[i].fitness = f64::NEG_INFINITY;
        }
    }

    /// Seed the population with initial weight patches.
    ///
    /// Call this before the first `evolve_step()` to provide starting
    /// mutations. Without seeding, evolution starts from zero-diff patches.
    pub fn seed_population(&mut self, base_patches: &[WeightPatch]) {
        for individual in &mut self.population {
            individual.patches = mutate_patches(
                base_patches,
                self.config.mutation_rate,
                self.config.mutation_magnitude,
            );
        }
    }
}

/// A small weight mutation: offset + replacement bytes.
///
/// Applied to on-chip SRAM via `NpuBackend::mutate_weights()`.
/// Designed to be small (tens of bytes) for zero-DMA speed.
#[derive(Debug, Clone)]
pub struct WeightPatch {
    /// Byte offset within the model's weight region.
    pub offset: usize,
    /// Replacement data.
    pub data: Vec<u8>,
}

impl WeightPatch {
    /// Create a new weight patch.
    #[must_use]
    pub const fn new(offset: usize, data: Vec<u8>) -> Self {
        Self { offset, data }
    }

    /// Size of this patch in bytes.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.data.len()
    }
}

/// Configuration for the evolution process.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Number of individuals in the population.
    pub population_size: usize,
    /// Probability of mutating each byte in a patch (0.0–1.0).
    pub mutation_rate: f64,
    /// Maximum magnitude of a single byte mutation (0–255).
    pub mutation_magnitude: u8,
    /// Maximum number of generations before stopping.
    pub max_generations: u64,
    /// Target fitness — stop early if achieved.
    pub target_fitness: Option<f64>,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            mutation_rate: 0.05,
            mutation_magnitude: 16,
            max_generations: 1000,
            target_fitness: None,
        }
    }
}

/// Result of a single evolution step.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Current generation number.
    pub generation: u64,
    /// Best fitness in this generation.
    pub best_fitness: f64,
    /// Mean fitness across the population.
    pub mean_fitness: f64,
    /// Population size.
    pub population_size: usize,
}

/// Trait for evaluating the fitness of an inference result.
///
/// Implement this for your specific optimization objective.
pub trait FitnessEvaluator: Send + Sync {
    /// Evaluate the fitness of an inference result.
    ///
    /// # Arguments
    ///
    /// * `input` — the input that was fed to the NPU
    /// * `output` — the inference result
    ///
    /// Returns a fitness score (higher is better).
    fn evaluate(&self, input: &[f32], output: &[f32]) -> f64;
}

/// An individual in the evolution population.
#[derive(Debug, Clone)]
struct Individual {
    _id: usize,
    patches: Vec<WeightPatch>,
    fitness: f64,
}

/// Apply random mutations to a set of weight patches.
fn mutate_patches(patches: &[WeightPatch], mutation_rate: f64, magnitude: u8) -> Vec<WeightPatch> {
    patches
        .iter()
        .map(|patch| {
            let mut new_data = patch.data.clone();
            // Simple deterministic-ish mutation using index as seed
            // (production version should use a proper PRNG)
            for (i, byte) in new_data.iter_mut().enumerate() {
                let pseudo_rand =
                    ((i as u64).wrapping_mul(0x5851_f42d_4c95_7f2d) >> 56) as f64 / 256.0;
                if pseudo_rand < mutation_rate {
                    let delta =
                        ((pseudo_rand * f64::from(magnitude) * 2.0) as i16) - i16::from(magnitude);
                    *byte = byte.wrapping_add(delta as u8);
                }
            }
            WeightPatch {
                offset: patch.offset,
                data: new_data,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendType, ModelHandle, NpuBackend};
    use crate::backends::software::SoftwareBackend;

    struct MaxOutputFitness;

    impl FitnessEvaluator for MaxOutputFitness {
        fn evaluate(&self, _input: &[f32], output: &[f32]) -> f64 {
            output.iter().map(|&v| f64::from(v)).sum()
        }
    }

    /// Wraps `SoftwareBackend` so `mutate_weights` is a no-op (evolver unit tests without SRAM).
    #[derive(Debug)]
    struct EvolverTestBackend {
        inner: SoftwareBackend,
    }

    impl EvolverTestBackend {
        fn new_loaded() -> Self {
            let mut inner = SoftwareBackend::new(4, 2, 1);
            inner
                .load_weights(&[0.1f32; 8], &[0.05f32; 16], &[0.2f32; 4])
                .unwrap();
            Self { inner }
        }
    }

    impl NpuBackend for EvolverTestBackend {
        fn init(_device_id: &str) -> Result<Self> {
            Ok(Self::new_loaded())
        }

        fn capabilities(&self) -> &crate::capabilities::Capabilities {
            self.inner.capabilities()
        }

        fn load_model(&mut self, model: &[u8]) -> Result<ModelHandle> {
            self.inner.load_model(model)
        }

        fn load_reservoir(&mut self, w_in: &[f32], w_res: &[f32]) -> Result<()> {
            self.inner.load_reservoir(w_in, w_res)
        }

        fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
            self.inner.infer(input)
        }

        fn measure_power(&self) -> Result<f32> {
            self.inner.measure_power()
        }

        fn backend_type(&self) -> BackendType {
            self.inner.backend_type()
        }

        fn is_ready(&self) -> bool {
            self.inner.is_ready()
        }

        fn mutate_weights(&mut self, _offset: usize, _data: &[u8]) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn default_config() {
        let config = EvolutionConfig::default();
        assert_eq!(config.population_size, 10);
        assert!(config.mutation_rate > 0.0);
    }

    #[test]
    fn weight_patch_size() {
        let patch = WeightPatch::new(0x100, vec![0u8; 64]);
        assert_eq!(patch.size(), 64);
    }

    #[test]
    fn evolver_creation() {
        let config = EvolutionConfig::default();
        let evolver = NpuEvolver::new(config, MaxOutputFitness);
        assert_eq!(evolver.generation(), 0);
        assert_eq!(evolver.best_fitness(), f64::NEG_INFINITY);
    }

    #[test]
    fn mutate_patches_preserves_count() {
        let patches = vec![
            WeightPatch::new(0, vec![1, 2, 3, 4]),
            WeightPatch::new(100, vec![5, 6, 7, 8]),
        ];
        let mutated = mutate_patches(&patches, 0.5, 10);
        assert_eq!(mutated.len(), 2);
        assert_eq!(mutated[0].offset, 0);
        assert_eq!(mutated[1].offset, 100);
    }

    #[test]
    fn evolve_step_runs_without_error() {
        let config = EvolutionConfig {
            population_size: 4,
            ..EvolutionConfig::default()
        };
        let mut evolver = NpuEvolver::new(config, MaxOutputFitness);
        let mut backend = EvolverTestBackend::new_loaded();
        let input = [0.5f32, -0.25];
        let res = evolver.evolve_step(&mut backend, &input).unwrap();
        assert_eq!(res.generation, 1);
        assert_eq!(res.population_size, 4);
        assert!(res.mean_fitness.is_finite());
    }

    #[test]
    fn seed_population_then_evolve_advances_generation() {
        let config = EvolutionConfig {
            population_size: 4,
            ..EvolutionConfig::default()
        };
        let mut evolver = NpuEvolver::new(config, MaxOutputFitness);
        evolver.seed_population(&[WeightPatch::new(0, vec![1, 2, 3])]);
        let mut backend = EvolverTestBackend::new_loaded();
        let input = [0.0f32, 0.0];
        let _ = evolver.evolve_step(&mut backend, &input).unwrap();
        let _ = evolver.evolve_step(&mut backend, &input).unwrap();
        assert_eq!(evolver.generation(), 2);
    }

    #[test]
    fn best_patches_returns_some_after_evolution() {
        let config = EvolutionConfig {
            population_size: 4,
            ..EvolutionConfig::default()
        };
        let mut evolver = NpuEvolver::new(config, MaxOutputFitness);
        evolver.seed_population(&[WeightPatch::new(0, vec![1u8])]);
        let mut backend = EvolverTestBackend::new_loaded();
        let _ = evolver.evolve_step(&mut backend, &[0.1f32, 0.2]).unwrap();
        assert!(evolver.best_patches().is_some());
    }

    #[test]
    fn evolution_result_clone() {
        let r = EvolutionResult {
            generation: 2,
            best_fitness: 1.0,
            mean_fitness: 0.5,
            population_size: 4,
        };
        assert_eq!(r.clone().generation, 2);
    }
}
