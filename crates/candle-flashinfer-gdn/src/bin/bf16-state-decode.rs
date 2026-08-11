//! Eager numerical acceptance for ILP4 and wide-vector BF16-state T=1 decode.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use candle::cuda_backend::cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use candle::{DType, Device, Tensor};
use candle_flashinfer_gdn::{DecodeInputs, GdnDecode, GdnDecodeConfig, GdnHandle};

const H: usize = 1;
const HV: usize = 1;
const K: usize = 128;
const V: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let cache_root = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err(usage().into());
    }
    let handle = match cache_root {
        Some(cache_root) => GdnHandle::with_cache_root(cache_root)?,
        None => GdnHandle::new()?,
    };

    let device = Device::new_cuda_with_stream(0)?;
    // SAFETY: this acceptance binary synchronizes the single owned stream before
    // capture and before dropping any tensors used by the captured graph.
    unsafe {
        device.as_cuda_device()?.disable_event_tracking();
    }
    let decode = GdnDecode::new(&handle, &device, GdnDecodeConfig::new(H, HV, K, V))?;
    let fallback = run_case(2, &decode, &device)?;
    let wide = run_case(512, &decode, &device)?;
    println!(
        "BF16-state decode passed: ILP4 output={:e}, state={:e}; wide output={:e}, state={:e}",
        fallback.0, fallback.1, wide.0, wide.1
    );
    Ok(())
}

fn run_case(
    batch: usize,
    decode: &GdnDecode,
    device: &Device,
) -> Result<(f32, f32), Box<dyn Error>> {
    let x = Tensor::zeros((batch, 1), DType::BF16, device)?;
    let plan = decode.prepare(&x)?;
    let pool_size = batch + 5;
    let pool_indices: Vec<usize> = (0..batch).map(|index| pool_size - 1 - index).collect();
    let device_indices: Vec<i32> = pool_indices
        .iter()
        .map(|index| i32::try_from(*index))
        .collect::<Result<_, _>>()?;
    let initial_values = values(pool_size * HV * V * K, 0.013, 0.031);
    let a_log_values = vec![-0.4; HV];
    let a_values = values(batch * HV, 0.2, 0.13);
    let dt_bias_values = vec![0.05; HV];
    let q_values = values(batch * H * K, 0.021, 0.017);
    let k_values = values(batch * H * K, 0.018, 0.023);
    let v_values = values(batch * HV * V, 0.11, 0.019);
    let beta_values = values(batch * HV, 0.3, 0.11);

    let (state, initial_state) = bf16_tensor(initial_values, (pool_size, HV, V, K), device)?;
    let a_log = Tensor::from_vec(a_log_values.clone(), HV, device)?;
    let (a, a_quantized) = bf16_tensor(a_values, (batch, 1, HV), device)?;
    let dt_bias = Tensor::from_vec(dt_bias_values.clone(), HV, device)?;
    let (q, q_quantized) = bf16_tensor(q_values, (batch, 1, H, K), device)?;
    let (key, k_quantized) = bf16_tensor(k_values, (batch, 1, H, K), device)?;
    // Exercise the model adapter's zero-byte-offset normalization with a
    // contiguous subview, matching split QKV projections in real models.
    let key = Tensor::cat(&[Tensor::zeros((1, 1, H, K), DType::BF16, device)?, key], 0)?
        .narrow(0, 1, batch)?;
    let (value, v_quantized) = bf16_tensor(v_values, (batch, 1, HV, V), device)?;
    let (beta, beta_quantized) = bf16_tensor(beta_values, (batch, 1, HV), device)?;
    let state_indices = Tensor::from_vec(device_indices, batch, device)?;

    let inputs = DecodeInputs {
        state: &state,
        a_log: &a_log,
        a: &a,
        dt_bias: &dt_bias,
        q: &q,
        k: &key,
        v: &value,
        beta: &beta,
        state_indices: &state_indices,
        checkpoint_indices: None,
    };
    let output = plan.forward(&inputs)?;
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
        plan.scale(),
        &pool_indices,
    );
    let expected_output = quantize_bf16(&expected_output)?;
    let expected_state = quantize_bf16(&expected_state)?;
    let output_error = max_abs_error(&f32_values(&output)?, &expected_output);
    let state_error = max_abs_error(&f32_values(&state)?, &expected_state);
    if output_error > 2.0e-3 || state_error > 2.0e-3 {
        return Err(io::Error::other(format!(
            "batch {batch} numerical mismatch: output={output_error:e}, state={state_error:e}"
        ))
        .into());
    }
    if batch == 2 {
        let stream = device.as_cuda_device()?.cuda_stream();
        stream.synchronize()?;
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
        let captured_output = plan.forward(&inputs)?;
        drop(captured_output);
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
            .ok_or_else(|| io::Error::other("CUDA graph capture returned no graph"))?;
        graph.launch()?;
        stream.synchronize()?;
    }
    Ok((output_error, state_error))
}

fn usage() -> io::Error {
    io::Error::other("usage: bf16-state-decode [cache-root]")
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

fn quantize_bf16(values: &[f32]) -> candle::Result<Vec<f32>> {
    let tensor =
        Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu)?.to_dtype(DType::BF16)?;
    f32_values(&tensor)
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
                let state_offset = ((state_slot * HV + value_head) * V + value_index) * K;
                let mut state_key = 0.0;
                for key_index in 0..K {
                    let state_value = initial_state[state_offset + key_index] * decay;
                    state[state_offset + key_index] = state_value;
                    state_key += state_value * (k[k_offset + key_index] / k_norm);
                }
                let delta = (v[gate_index * V + value_index] - state_key) * update_gate;
                let mut state_query = 0.0;
                for key_index in 0..K {
                    state[state_offset + key_index] += (k[k_offset + key_index] / k_norm) * delta;
                    state_query += state[state_offset + key_index]
                        * (q[q_offset + key_index] / q_norm)
                        * scale;
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
