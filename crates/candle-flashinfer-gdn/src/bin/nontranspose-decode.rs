//! Eager numerical acceptance test for small- and large-batch non-transposed decode.

use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use candle::{DType, Device, Tensor};
use candle_flashinfer_gdn::{NontransposeDecodeInputs, NontransposeDecodePlan};
use flashinfer_gdn::{NontransposeDecodeCompiler, NontransposeDecodeSpecialization};

const H: usize = 1;
const HV: usize = 1;
const K: usize = 128;
const V: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let python = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    let cache_root = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let device = Device::new_cuda_with_stream(0)?;
    let cuda_device = device.as_cuda_device()?.clone();
    let (major, minor) = cuda_device.cuda_stream().context().compute_capability()?;
    let gpu_arch = format!("sm_{major}{minor}a");

    let small = run_case(2, &python, &cache_root, &gpu_arch, &device)?;
    let large = run_case(32, &python, &cache_root, &gpu_arch, &device)?;
    println!(
        "non-transposed decode passed: small output={:e}, state={:e}; large output={:e}, state={:e}",
        small.0, small.1, large.0, large.1
    );
    Ok(())
}

fn run_case(
    batch: usize,
    python: &Path,
    cache_root: &Path,
    gpu_arch: &str,
    device: &Device,
) -> Result<(f32, f32), Box<dyn Error>> {
    let cuda_device = device.as_cuda_device()?.clone();
    let specialization = NontransposeDecodeSpecialization::new(gpu_arch, H, HV, K, V, batch)?;
    let compiler = NontransposeDecodeCompiler::from_managed_python(python, cache_root)?
        .specialization(specialization)?;
    let plan = NontransposeDecodePlan::prepare(&compiler, &cuda_device, batch)?;

    let pool_size = batch + 5;
    let pool_indices: Vec<usize> = (0..batch).map(|index| pool_size - 1 - index).collect();
    let device_indices: Vec<i32> = pool_indices
        .iter()
        .map(|index| i32::try_from(*index))
        .collect::<Result<_, _>>()?;
    let initial_state = values(pool_size * HV * K * V, 0.013, 0.031);
    let a_log_values = vec![-0.4; HV];
    let a_values = values(batch * HV, 0.2, 0.13);
    let dt_bias_values = vec![0.05; HV];
    let q_values = values(batch * H * K, 0.021, 0.017);
    let k_values = values(batch * H * K, 0.018, 0.023);
    let v_values = values(batch * HV * V, 0.11, 0.019);
    let beta_values = values(batch * HV, 0.3, 0.11);

    let state = Tensor::from_vec(initial_state.clone(), (pool_size, HV, K, V), device)?;
    let a_log = Tensor::from_vec(a_log_values.clone(), HV, device)?;
    let (a, a_quantized) = bf16_tensor(a_values, (batch, 1, HV), device)?;
    let dt_bias = Tensor::from_vec(dt_bias_values.clone(), HV, device)?;
    let (q, q_quantized) = bf16_tensor(q_values, (batch, 1, H, K), device)?;
    let (key, k_quantized) = bf16_tensor(k_values, (batch, 1, H, K), device)?;
    let (value, v_quantized) = bf16_tensor(v_values, (batch, 1, HV, V), device)?;
    let (beta, beta_quantized) = bf16_tensor(beta_values, (batch, 1, HV), device)?;
    let state_indices = Tensor::from_vec(device_indices, batch, device)?;
    let output = plan.forward(&NontransposeDecodeInputs {
        state: &state,
        a_log: &a_log,
        a: &a,
        dt_bias: &dt_bias,
        q: &q,
        k: &key,
        v: &value,
        beta: &beta,
        state_indices: &state_indices,
    })?;
    device.synchronize()?;

    let (expected_output, expected_state) = reference(
        batch,
        &initial_state,
        &a_log_values,
        &a_quantized,
        &dt_bias_values,
        &q_quantized,
        &k_quantized,
        &v_quantized,
        &beta_quantized,
        plan.specialization().scale,
        &pool_indices,
    );
    let output_error = max_abs_error(&f32_values(&output)?, &expected_output);
    let state_error = max_abs_error(&f32_values(&state)?, &expected_state);
    if output_error > 2.0e-3 || state_error > 2.0e-4 {
        return Err(io::Error::other(format!(
            "batch {batch} numerical mismatch: output={output_error:e}, state={state_error:e}"
        ))
        .into());
    }
    Ok((output_error, state_error))
}

fn usage() -> io::Error {
    io::Error::other("usage: nontranspose-decode <compiler-python> <cache-root>")
}

fn values(length: usize, amplitude: f32, phase: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index as f32 + 1.0) * phase).sin() * amplitude)
        .collect()
}

fn bf16_tensor<S: candle::shape::ShapeWithOneHole>(
    values: Vec<f32>,
    shape: S,
    device: &Device,
) -> candle::Result<(Tensor, Vec<f32>)> {
    let quantized = Tensor::from_vec(values, shape, &Device::Cpu)?.to_dtype(DType::BF16)?;
    let values = f32_values(&quantized)?;
    Ok((quantized.to_device(device)?, values))
}

fn f32_values(tensor: &Tensor) -> candle::Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

#[allow(clippy::too_many_arguments)]
fn reference(
    batch_size: usize,
    initial_state: &[f32],
    a_log: &[f32],
    a: &[f32],
    dt_bias: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    scale: f32,
    state_indices: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0; batch_size * HV * V];
    for (batch, &state_slot) in state_indices.iter().enumerate().take(batch_size) {
        for value_head in 0..HV {
            let query_head = value_head / (HV / H);
            let q_offset = (batch * H + query_head) * K;
            let k_offset = (batch * H + query_head) * K;
            let q_norm = (q[q_offset..q_offset + K]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            let k_norm = (k[k_offset..k_offset + K]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                + 1.0e-6)
                .sqrt();
            let gate_index = batch * HV + value_head;
            let softplus = (1.0 + (a[gate_index] + dt_bias[value_head]).exp()).ln();
            let decay = (-a_log[value_head].exp() * softplus).exp();
            let update_gate = 1.0 / (1.0 + (-beta[gate_index]).exp());
            for value_index in 0..V {
                let mut state_key = 0.0;
                for key_index in 0..K {
                    let state_index =
                        ((state_slot * HV + value_head) * K + key_index) * V + value_index;
                    let state_value = initial_state[state_index] * decay;
                    state[state_index] = state_value;
                    state_key += state_value * (k[k_offset + key_index] / k_norm);
                }
                let delta = (v[gate_index * V + value_index] - state_key) * update_gate;
                let mut state_query = 0.0;
                for key_index in 0..K {
                    let state_index =
                        ((state_slot * HV + value_head) * K + key_index) * V + value_index;
                    state[state_index] += (k[k_offset + key_index] / k_norm) * delta;
                    state_query += state[state_index] * (q[q_offset + key_index] / q_norm) * scale;
                }
                output[gate_index * V + value_index] = state_query;
            }
        }
    }
    (output, state)
}

fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}
