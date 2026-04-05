// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure CPU f32 + tanh ESN step (software fallback).

#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

use crate::error::{AkidaError, Result};

use super::EsnWeights;

/// Pure f32 + tanh ESN executor (powers `PureSoftware` mode).
pub(super) struct SoftwareEsnExecutor {
    input_dim: usize,
    reservoir_dim: usize,
    output_dim: usize,
    w_in: Vec<f32>,
    w_res: Vec<f32>,
    w_out: Vec<f32>,
    state: Vec<f32>,
    leak: f32,
}

impl SoftwareEsnExecutor {
    pub(super) fn new(w: &EsnWeights) -> Self {
        Self {
            input_dim: w.input_dim,
            reservoir_dim: w.reservoir_dim,
            output_dim: w.output_dim,
            w_in: w.w_in.clone(),
            w_res: w.w_res.clone(),
            w_out: w.w_out.clone(),
            state: vec![0.0f32; w.reservoir_dim],
            leak: w.leak_rate,
        }
    }

    pub(super) fn step(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let rs = self.reservoir_dim;
        let is = self.input_dim;
        if input.len() != is {
            return Err(AkidaError::capability_query_failed(format!(
                "step: input len {} != input_dim {}",
                input.len(),
                is
            )));
        }
        let alpha = self.leak;
        let mut pre = vec![0.0f32; rs];
        for i in 0..rs {
            for j in 0..is {
                pre[i] += self.w_in[i * is + j] * input[j];
            }
            for j in 0..rs {
                pre[i] += self.w_res[i * rs + j] * self.state[j];
            }
        }
        for i in 0..rs {
            self.state[i] = (1.0 - alpha) * self.state[i] + alpha * pre[i].tanh();
        }
        let os = self.output_dim;
        Ok((0..os)
            .map(|i| {
                self.w_out[i * rs..(i + 1) * rs]
                    .iter()
                    .zip(self.state.iter())
                    .map(|(w, s)| w * s)
                    .sum()
            })
            .collect())
    }

    pub(super) fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub(super) fn reservoir_state(&self) -> Vec<f32> {
        self.state.clone()
    }
}
