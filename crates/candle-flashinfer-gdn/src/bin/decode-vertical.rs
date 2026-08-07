//! Nonzero numerical and CUDA Graph acceptance test for the first decode slice.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use candle::cuda_backend::cudarc::driver::sys::{
    CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
};
use candle::cuda_backend::{CudaStorage, CudaStorageSlice};
use candle::{CpuStorage, DType, Device, InplaceOp1, Layout, Storage, Tensor};
use candle_flashinfer_gdn::{PretransposeDecodeInputs, PretransposeDecodePlan};
use flashinfer_gdn::{PretransposeDecodeCompiler, PretransposeDecodeSpecialization};

const BATCH: usize = 2;
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

    let candle_device = Device::new_cuda_with_stream(0)?;
    let cuda_device = candle_device.as_cuda_device()?.clone();
    // SAFETY: this acceptance test synchronizes explicitly at each host boundary,
    // matching modeld-core's CUDA Graph device setup.
    unsafe {
        cuda_device.disable_event_tracking();
    }
    let (major, minor) = cuda_device.cuda_stream().context().compute_capability()?;
    let specialization =
        PretransposeDecodeSpecialization::new(format!("sm_{major}{minor}a"), H, HV, K, V)?;
    let compiler =
        PretransposeDecodeCompiler::from_managed_python(python.clone(), cache_root.clone())?
            .specialization(specialization)?;
    let plan = PretransposeDecodePlan::prepare(&compiler, &cuda_device, BATCH)?;

    let initial_state = values(BATCH * HV * V * K, 0.013, 0.031);
    let a_log_values = vec![-0.4];
    let a_values = vec![0.2, -0.1];
    let dt_bias_values = vec![0.05];
    let q_values = values(BATCH * H * K, 0.021, 0.017);
    let k_values = values(BATCH * H * K, 0.018, 0.023);
    let v_values = values(BATCH * HV * V, 0.11, 0.019);
    let beta_values = vec![-0.2, 0.3];

    let state = Tensor::from_vec(initial_state.clone(), (BATCH, HV, V, K), &candle_device)?;
    let a_log = Tensor::from_vec(a_log_values.clone(), HV, &candle_device)?;
    let (a, a_quantized) = bf16_tensor(a_values, (BATCH, 1, HV), &candle_device)?;
    let dt_bias = Tensor::from_vec(dt_bias_values.clone(), HV, &candle_device)?;
    let (q, q_quantized) = bf16_tensor(q_values, (BATCH, 1, H, K), &candle_device)?;
    let (k, k_quantized) = bf16_tensor(k_values, (BATCH, 1, H, K), &candle_device)?;
    let (v, v_quantized) = bf16_tensor(v_values, (BATCH, 1, HV, V), &candle_device)?;
    let (beta, beta_quantized) = bf16_tensor(beta_values, (BATCH, 1, HV), &candle_device)?;
    let inputs = PretransposeDecodeInputs {
        state: &state,
        a_log: &a_log,
        a: &a,
        dt_bias: &dt_bias,
        q: &q,
        k: &k,
        v: &v,
        beta: &beta,
        state_indices: None,
        output_state_indices: None,
    };

    // The graph owner performs its ordinary eager warmup/reference passes.
    drop(plan.forward(&inputs)?);
    upload_f32(&state, &initial_state)?;
    let output = plan.forward(&inputs)?;
    candle_device.synchronize()?;

    let (expected_output, expected_state) = reference(
        &initial_state,
        &a_log_values,
        &a_quantized,
        &dt_bias_values,
        &q_quantized,
        &k_quantized,
        &v_quantized,
        &beta_quantized,
        plan.specialization().scale,
        &[0, 1],
        &[0, 1],
    );
    let actual_output = f32_values(&output)?;
    let actual_state = f32_values(&state)?;
    let output_error = max_abs_error(&actual_output, &expected_output);
    let state_error = max_abs_error(&actual_state, &expected_state);
    if output_error > 2.0e-3 || state_error > 2.0e-4 {
        return Err(io::Error::other(format!(
            "numerical mismatch: max output error={output_error:e}, state error={state_error:e}"
        ))
        .into());
    }

    upload_f32(&state, &initial_state)?;
    candle_device.synchronize()?;
    let stream = cuda_device.cuda_stream();
    stream
        .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
        .map_err(|error| io::Error::other(format!("failed to begin capture: {error}")))?;
    let captured_output = plan.forward(&inputs).map_err(|error| {
        io::Error::other(format!(
            "generated TVM launch failed during capture: {error}"
        ))
    })?;
    copy_bf16(&captured_output, &output)?;
    drop(captured_output);
    let graph = stream
        .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|error| io::Error::other(format!("failed to end capture: {error}")))?
        .ok_or_else(|| io::Error::other("capture produced no CUDA graph"))?;
    graph.launch()?;
    candle_device.synchronize()?;
    let first_replay = f32_values(&output)?;
    let first_replay_error = max_abs_error(&first_replay, &actual_output);
    if first_replay_error != 0.0 {
        return Err(io::Error::other(format!(
            "first graph replay differs from eager output by {first_replay_error:e}"
        ))
        .into());
    }

    upload_f32(&state, &initial_state)?;
    upload_f32(&a_log, &[0.4])?;
    graph.launch()?;
    candle_device.synchronize()?;
    let changed_replay = f32_values(&output)?;
    let changed_delta = max_abs_error(&changed_replay, &first_replay);
    if changed_delta < 1.0e-5 {
        return Err(io::Error::other("graph replay did not observe changed input contents").into());
    }

    upload_f32(&state, &initial_state)?;
    graph.launch()?;
    candle_device.synchronize()?;
    let repeated_replay = f32_values(&output)?;
    let repeated_error = max_abs_error(&changed_replay, &repeated_replay);
    if repeated_error != 0.0 {
        return Err(io::Error::other(format!(
            "repeated graph replay differs by {repeated_error:e}"
        ))
        .into());
    }

    upload_f32(&a_log, &a_log_values)?;
    let pool_specialization =
        PretransposeDecodeSpecialization::new(format!("sm_{major}{minor}a"), H, HV, K, V)?
            .pool_indexing(true);
    let pool_compiler = PretransposeDecodeCompiler::from_managed_python(python, cache_root)?
        .specialization(pool_specialization)?;
    let pool_plan = PretransposeDecodePlan::prepare(&pool_compiler, &cuda_device, BATCH)?;
    let pool_size = 4;
    let initial_pool = values(pool_size * HV * V * K, 0.014, 0.029);
    let pool = Tensor::from_vec(initial_pool.clone(), (pool_size, HV, V, K), &candle_device)?;
    let read_indices = Tensor::from_vec(vec![0_i32, 1], BATCH, &candle_device)?;
    let write_indices = Tensor::from_vec(vec![2_i32, 3], BATCH, &candle_device)?;
    let pool_inputs = PretransposeDecodeInputs {
        state: &pool,
        a_log: &a_log,
        a: &a,
        dt_bias: &dt_bias,
        q: &q,
        k: &k,
        v: &v,
        beta: &beta,
        state_indices: Some(&read_indices),
        output_state_indices: Some(&write_indices),
    };
    drop(pool_plan.forward(&pool_inputs)?);
    upload_f32(&pool, &initial_pool)?;
    let pool_output = pool_plan.forward(&pool_inputs)?;
    candle_device.synchronize()?;
    let (expected_pool_output, expected_pool) = reference(
        &initial_pool,
        &a_log_values,
        &a_quantized,
        &dt_bias_values,
        &q_quantized,
        &k_quantized,
        &v_quantized,
        &beta_quantized,
        pool_plan.specialization().scale,
        &[0, 1],
        &[2, 3],
    );
    let actual_pool_output = f32_values(&pool_output)?;
    let actual_pool = f32_values(&pool)?;
    let pool_output_error = max_abs_error(&actual_pool_output, &expected_pool_output);
    let pool_state_error = max_abs_error(&actual_pool, &expected_pool);
    if pool_output_error > 2.0e-3 || pool_state_error > 2.0e-4 {
        return Err(io::Error::other(format!(
            "indexed numerical mismatch: max output error={pool_output_error:e}, state error={pool_state_error:e}"
        ))
        .into());
    }

    upload_f32(&pool, &initial_pool)?;
    candle_device.synchronize()?;
    stream
        .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
        .map_err(|error| io::Error::other(format!("failed to begin pool capture: {error}")))?;
    let captured_pool_output = pool_plan.forward(&pool_inputs).map_err(|error| {
        io::Error::other(format!(
            "generated indexed TVM launch failed during capture: {error}"
        ))
    })?;
    copy_bf16(&captured_pool_output, &pool_output)?;
    drop(captured_pool_output);
    let pool_graph = stream
        .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|error| io::Error::other(format!("failed to end pool capture: {error}")))?
        .ok_or_else(|| io::Error::other("pool capture produced no CUDA graph"))?;
    pool_graph.launch()?;
    candle_device.synchronize()?;
    let pool_replay = f32_values(&pool_output)?;
    let pool_replay_error = max_abs_error(&pool_replay, &actual_pool_output);
    if pool_replay_error != 0.0 {
        return Err(io::Error::other(format!(
            "indexed graph replay differs from eager output by {pool_replay_error:e}"
        ))
        .into());
    }

    let second_candle_device = Device::new_cuda_with_stream(0)?;
    let second_cuda_device = second_candle_device.as_cuda_device()?.clone();
    // SAFETY: cross-stream dependencies are synchronized explicitly below.
    unsafe {
        second_cuda_device.disable_event_tracking();
    }
    let second_plan = PretransposeDecodePlan::prepare(&compiler, &second_cuda_device, BATCH)?;
    let second_inputs = PretransposeDecodeInputs {
        state: &state,
        a_log: &a_log,
        a: &a,
        dt_bias: &dt_bias,
        q: &q,
        k: &k,
        v: &v,
        beta: &beta,
        state_indices: None,
        output_state_indices: None,
    };
    let second_output = second_plan.forward(&second_inputs)?;
    second_candle_device.synchronize()?;
    upload_f32(&state, &initial_state)?;
    candle_device.synchronize()?;
    let second_stream = second_cuda_device.cuda_stream();
    second_stream
        .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
        .map_err(|error| io::Error::other(format!("failed to begin second capture: {error}")))?;
    let captured_second_output = second_plan.forward(&second_inputs)?;
    copy_bf16(&captured_second_output, &second_output)?;
    drop(captured_second_output);
    let second_graph = second_stream
        .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|error| io::Error::other(format!("failed to end second capture: {error}")))?
        .ok_or_else(|| io::Error::other("second-stream capture produced no CUDA graph"))?;
    second_graph.launch()?;
    second_candle_device.synchronize()?;
    let second_stream_output = f32_values(&second_output)?;
    let second_stream_error = max_abs_error(&second_stream_output, &actual_output);
    if second_stream_error != 0.0 {
        return Err(io::Error::other(format!(
            "second-stream graph output differs by {second_stream_error:e}"
        ))
        .into());
    }

    println!(
        "decode vertical slice passed: direct output error={output_error:e}, state error={state_error:e}, changed replay delta={changed_delta:e}; indexed output error={pool_output_error:e}, state error={pool_state_error:e}; second-stream replay error={second_stream_error:e}"
    );
    Ok(())
}

fn usage() -> io::Error {
    io::Error::other("usage: decode-vertical <compiler-python> <cache-root>")
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
    initial_state: &[f32],
    a_log: &[f32],
    a: &[f32],
    dt_bias: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    scale: f32,
    read_indices: &[usize],
    write_indices: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0; BATCH * HV * V];
    for batch in 0..BATCH {
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
                let source_offset = ((read_indices[batch] * HV + value_head) * V + value_index) * K;
                let destination_offset =
                    ((write_indices[batch] * HV + value_head) * V + value_index) * K;
                let mut state_key = 0.0;
                for key_index in 0..K {
                    let state_value = initial_state[source_offset + key_index] * decay;
                    state[destination_offset + key_index] = state_value;
                    state_key += state_value * (k[k_offset + key_index] / k_norm);
                }
                let delta = (v[gate_index * V + value_index] - state_key) * update_gate;
                let mut state_query = 0.0;
                for key_index in 0..K {
                    let state_value = &mut state[destination_offset + key_index];
                    *state_value += (k[k_offset + key_index] / k_norm) * delta;
                    state_query += *state_value * (q[q_offset + key_index] / q_norm) * scale;
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

struct UploadF32<'a>(&'a [f32]);

impl InplaceOp1 for UploadF32<'_> {
    fn name(&self) -> &'static str {
        "decode-vertical-upload-f32"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> candle::Result<()> {
        Err(candle::Error::Msg(
            "decode acceptance upload expects CUDA storage".into(),
        ))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> candle::Result<()> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(candle::Error::Msg(
                "test upload requires a compact tensor".into(),
            ));
        }
        let device = storage.device.clone();
        let destination = storage.as_cuda_slice_mut::<f32>()?;
        if destination.len() != self.0.len() {
            return Err(candle::Error::Msg("test upload length mismatch".into()));
        }
        device.memcpy_htod(self.0, destination)
    }
}

fn upload_f32(tensor: &Tensor, values: &[f32]) -> candle::Result<()> {
    tensor.inplace_op1(&UploadF32(values))
}

struct CopyBf16 {
    source: Tensor,
}

impl InplaceOp1 for CopyBf16 {
    fn name(&self) -> &'static str {
        "decode-vertical-copy-bf16"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> candle::Result<()> {
        Err(candle::Error::Msg(
            "decode acceptance copy expects CUDA storage".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        destination_storage: &mut CudaStorage,
        destination_layout: &Layout,
    ) -> candle::Result<()> {
        let (source_storage, source_layout) = self.source.storage_and_layout();
        if source_layout.dims() != destination_layout.dims()
            || !source_layout.is_contiguous()
            || !destination_layout.is_contiguous()
        {
            return Err(candle::Error::Msg(
                "test copy requires equal, contiguous layouts".into(),
            ));
        }
        let Storage::Cuda(source_storage) = &*source_storage else {
            return Err(candle::Error::Msg(
                "test copy source must use CUDA storage".into(),
            ));
        };
        let count = source_layout.shape().elem_count();
        let source_start = source_layout.start_offset();
        let destination_start = destination_layout.start_offset();
        let device = destination_storage.device.clone();
        match (&source_storage.slice, &mut destination_storage.slice) {
            (CudaStorageSlice::BF16(source), CudaStorageSlice::BF16(destination)) => {
                let source = source.slice(source_start..source_start + count);
                let mut destination =
                    destination.slice_mut(destination_start..destination_start + count);
                device.memcpy_dtod(&source, &mut destination)
            }
            _ => Err(candle::Error::Msg(
                "test copy requires BF16 source and destination".into(),
            )),
        }
    }
}

fn copy_bf16(source: &Tensor, destination: &Tensor) -> candle::Result<()> {
    destination.inplace_op1(&CopyBf16 {
        source: source.clone(),
    })
}
