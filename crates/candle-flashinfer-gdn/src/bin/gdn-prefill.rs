//! Eager numerical acceptance for architecture-dispatched GDN prefill.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use candle_flashinfer_gdn::{GdnHandle, GdnPrefill, GdnPrefillConfig, PrefillInputs};

const H: usize = 1;
const HV: usize = 1;
const K: usize = 128;
const V: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let cache_root = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err(io::Error::other("usage: gdn-prefill [cache-root]").into());
    }
    let handle = match cache_root {
        Some(cache_root) => GdnHandle::with_cache_root(cache_root)?,
        None => GdnHandle::new()?,
    };
    let device = Device::new_cuda_with_stream(0)?;
    let checkpointed = env::var_os("GDN_PREFILL_CHECKPOINTS").is_some();
    let prefill = GdnPrefill::new(&handle, &device, GdnPrefillConfig::new(H, HV, K, V))?;

    let batch = 2;
    let tokens_per_sequence = if checkpointed { 64 } else { 1 };
    let total_tokens = batch * tokens_per_sequence;
    let pool_size = 5;
    let mut qk_values = vec![0.0_f32; total_tokens * H * K];
    for token in 0..total_tokens {
        qk_values[token * H * K] = 1.0;
    }
    let q = bf16_tensor(qk_values.clone(), (total_tokens, H, K), &device)?;
    let k = bf16_tensor(qk_values, (total_tokens, H, K), &device)?;
    let mut v_values = vec![0.0_f32; total_tokens * HV * V];
    for token in 0..total_tokens {
        let value = if token < tokens_per_sequence {
            0.25
        } else {
            -0.125
        };
        v_values[token * HV * V..(token + 1) * HV * V].fill(value);
    }
    let v = bf16_tensor(v_values.clone(), (total_tokens, HV, V), &device)?;
    let alpha = Tensor::ones((total_tokens, HV), DType::F32, &device)?;
    let beta = Tensor::ones((total_tokens, HV), DType::F32, &device)?;
    let state = Tensor::zeros((pool_size, HV, V, K), DType::BF16, &device)?;
    let state_indices = Tensor::from_vec(vec![3_i32, 1], batch, &device)?;
    let cu_seqlens = Tensor::from_vec(
        vec![0_i32, tokens_per_sequence as i32, total_tokens as i32],
        batch + 1,
        &device,
    )?;
    let state_checkpoints = checkpointed
        .then(|| Tensor::zeros((batch, HV, V, K), DType::BF16, &device))
        .transpose()?;
    let checkpoint_cu_starts = checkpointed
        .then(|| Tensor::from_vec(vec![0_i32, 1, 2], batch + 1, &device))
        .transpose()?;
    let plan = if checkpointed {
        prefill.prepare_checkpointed(&q, &cu_seqlens, 64, batch)?
    } else {
        prefill.prepare(&q, &cu_seqlens)?
    };
    let output = plan.forward(&PrefillInputs {
        state: &state,
        state_indices: &state_indices,
        q: &q,
        k: &k,
        v: &v,
        alpha: &alpha,
        beta: &beta,
        cu_seqlens: &cu_seqlens,
        state_checkpoints: state_checkpoints.as_ref(),
        checkpoint_cu_starts: checkpoint_cu_starts.as_ref(),
    })?;
    device.synchronize()?;

    let scale = (K as f32).sqrt().recip();
    let expected_output = bf16_values(
        v_values.iter().map(|value| value * scale).collect(),
        (total_tokens, HV, V),
    )?;
    let mut expected_state = vec![0.0_f32; pool_size * HV * V * K];
    for token in 0..total_tokens {
        let slot = if token < tokens_per_sequence {
            3_usize
        } else {
            1_usize
        };
        for value_index in 0..V {
            let state_offset = (slot * HV * V + value_index) * K;
            expected_state[state_offset] = v_values[token * V + value_index];
        }
    }
    let expected_state = bf16_values(expected_state, (pool_size, HV, V, K))?;
    let output_error = max_abs_error(&f32_values(&output)?, &expected_output);
    let state_error = max_abs_error(&f32_values(&state)?, &expected_state);
    let checkpoint_error = if let Some(checkpoints) = state_checkpoints.as_ref() {
        let mut expected = vec![0.0_f32; batch * HV * V * K];
        for sequence in 0..batch {
            let token = (sequence + 1) * tokens_per_sequence - 1;
            for value_index in 0..V {
                let offset = (sequence * HV * V + value_index) * K;
                expected[offset] = v_values[token * HV * V + value_index];
            }
        }
        max_abs_error(&f32_values(checkpoints)?, &expected)
    } else {
        0.0
    };
    if output_error > 1.0e-3 || state_error > 1.0e-3 || checkpoint_error > 1.0e-3 {
        return Err(io::Error::other(format!(
            "impulse prefill mismatch: output={output_error:e}, state={state_error:e}, checkpoint={checkpoint_error:e}"
        ))
        .into());
    }
    println!(
        "GDN prefill passed on {:?}: N={}, B={}, checkpointed={}, output={:e}, state={:e}, checkpoint={:e}",
        prefill.backend(),
        plan.total_tokens(),
        plan.batch(),
        checkpointed,
        output_error,
        state_error,
        checkpoint_error,
    );
    Ok(())
}

fn bf16_tensor<S: candle::shape::ShapeWithOneHole>(
    values: Vec<f32>,
    shape: S,
    device: &Device,
) -> candle::Result<Tensor> {
    Tensor::from_vec(values, shape, &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(device)
}

fn bf16_values<S: candle::shape::ShapeWithOneHole>(
    values: Vec<f32>,
    shape: S,
) -> candle::Result<Vec<f32>> {
    f32_values(&Tensor::from_vec(values, shape, &Device::Cpu)?.to_dtype(DType::BF16)?)
}

fn f32_values(tensor: &Tensor) -> candle::Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}
