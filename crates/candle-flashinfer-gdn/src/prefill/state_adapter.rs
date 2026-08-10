use std::sync::OnceLock;

use candle::cuda_backend::CudaDevice;
use candle::cuda_backend::cudarc::driver::{
    CudaFunction, CudaStream as DriverStream, LaunchConfig, PushKernelArg,
};
use candle::cuda_backend::cudarc::nvrtc;
use candle::{DType, Device, Result, Tensor};

use crate::message;

const MODULE_NAME: &str = "flashinfer_gdn_candle_prefill_state_v1";
const GATHER_NAME: &str = "flashinfer_gdn_gather_bf16_to_f32";
const SCATTER_NAME: &str = "flashinfer_gdn_scatter_f32_to_bf16";
const CHECKPOINT_CAST_NAME: &str = "flashinfer_gdn_checkpoint_f32_to_bf16";
const CU_NAME: &str = "flashinfer_gdn_i32_to_i64";
const THREADS: u32 = 256;

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void flashinfer_gdn_gather_bf16_to_f32(
    const unsigned short* pool,
    const int* indices,
    float* compact,
    unsigned long long elements_per_state,
    unsigned long long total_elements,
    int pool_size) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_elements) return;
  const unsigned long long batch = linear / elements_per_state;
  const unsigned long long inner = linear - batch * elements_per_state;
  const int pool_index = indices[batch];
  if (pool_index >= 0 && pool_index < pool_size) {
    const unsigned int bits =
        ((unsigned int)pool[(unsigned long long)pool_index * elements_per_state + inner]) << 16;
    compact[linear] = __uint_as_float(bits);
  } else {
    compact[linear] = 0.0f;
  }
}

extern "C" __global__ void flashinfer_gdn_scatter_f32_to_bf16(
    const float* compact,
    const int* indices,
    unsigned short* pool,
    unsigned long long elements_per_state,
    unsigned long long total_elements,
    int pool_size) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_elements) return;
  const unsigned long long batch = linear / elements_per_state;
  const unsigned long long inner = linear - batch * elements_per_state;
  const int pool_index = indices[batch];
  if (pool_index >= 0 && pool_index < pool_size) {
    const unsigned int bits = __float_as_uint(compact[linear]);
    unsigned short result;
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
      result = (unsigned short)((bits >> 16) | 1u);
    } else {
      const unsigned int rounding = 0x7fffu + ((bits >> 16) & 1u);
      result = (unsigned short)((bits + rounding) >> 16);
    }
    pool[(unsigned long long)pool_index * elements_per_state + inner] = result;
  }
}

extern "C" __global__ void flashinfer_gdn_i32_to_i64(
    const int* input, long long* output, unsigned long long count) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear < count) output[linear] = (long long)input[linear];
}

extern "C" __global__ void flashinfer_gdn_checkpoint_f32_to_bf16(
    const float* input, unsigned short* output, unsigned long long count) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const unsigned int bits = __float_as_uint(input[linear]);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
    output[linear] = (unsigned short)((bits >> 16) | 1u);
  } else {
    const unsigned int rounding = 0x7fffu + ((bits >> 16) & 1u);
    output[linear] = (unsigned short)((bits + rounding) >> 16);
  }
}
"#;

static ADAPTER_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

/// Plan-owned buffers and fused conversion kernels for architectures whose
/// upstream prefill implementation only accepts compact float32 state.
#[derive(Debug)]
pub(super) struct StateAdapter {
    compact: Tensor,
    checkpoint_compact: Option<Tensor>,
    cu_i64: Tensor,
    checkpoint_cu_i64: Tensor,
    gather: CudaFunction,
    scatter: CudaFunction,
    convert_cu: CudaFunction,
    cast_checkpoints: CudaFunction,
    elements_per_state: u64,
    total_elements: u64,
    state_grid: u32,
    cu_count: u64,
    cu_grid: u32,
    checkpoint_elements: u64,
    checkpoint_grid: u32,
}

impl StateAdapter {
    pub(super) fn new(
        device: &CudaDevice,
        batch: usize,
        value_heads: usize,
        value_dim: usize,
        key_dim: usize,
        checkpoint_count: usize,
    ) -> Result<Self> {
        let elements_per_state = [value_heads, value_dim, key_dim].into_iter().try_fold(
            1_usize,
            |product, dimension| {
                product
                    .checked_mul(dimension)
                    .ok_or_else(|| message("prefill state size overflows usize"))
            },
        )?;
        let total_elements = elements_per_state
            .checked_mul(batch)
            .ok_or_else(|| message("prefill compact state size overflows usize"))?;
        let candle_device = Device::Cuda(device.clone());
        let compact = Tensor::zeros(
            (batch, value_heads, value_dim, key_dim),
            DType::F32,
            &candle_device,
        )?;
        let checkpoint_elements = elements_per_state
            .checked_mul(checkpoint_count)
            .ok_or_else(|| message("prefill checkpoint scratch size overflows usize"))?;
        let checkpoint_compact = if checkpoint_count > 0 {
            Some(Tensor::zeros(
                (checkpoint_count, value_heads, value_dim, key_dim),
                DType::F32,
                &candle_device,
            )?)
        } else {
            None
        };
        let cu_i64 = Tensor::zeros(batch + 1, DType::I64, &candle_device)?;
        let checkpoint_cu_i64 = Tensor::zeros(batch + 1, DType::I64, &candle_device)?;
        let ptx = adapter_ptx()?;
        let gather = device
            .get_or_load_custom_func(GATHER_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();
        let scatter = device
            .get_or_load_custom_func(SCATTER_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();
        let convert_cu = device
            .get_or_load_custom_func(CU_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();
        let cast_checkpoints = device
            .get_or_load_custom_func(CHECKPOINT_CAST_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();
        let total_elements = u64::try_from(total_elements)
            .map_err(|_| message("prefill compact state size does not fit u64"))?;
        let elements_per_state = u64::try_from(elements_per_state)
            .map_err(|_| message("prefill state row size does not fit u64"))?;
        let cu_count = u64::try_from(batch + 1)
            .map_err(|_| message("prefill cu_seqlens size does not fit u64"))?;
        let checkpoint_elements = u64::try_from(checkpoint_elements)
            .map_err(|_| message("prefill checkpoint scratch size does not fit u64"))?;
        Ok(Self {
            compact,
            checkpoint_compact,
            cu_i64,
            checkpoint_cu_i64,
            gather,
            scatter,
            convert_cu,
            cast_checkpoints,
            elements_per_state,
            total_elements,
            state_grid: grid_size(total_elements)?,
            cu_count,
            cu_grid: grid_size(cu_count)?,
            checkpoint_elements,
            checkpoint_grid: if checkpoint_elements > 0 {
                grid_size(checkpoint_elements)?
            } else {
                0
            },
        })
    }

    pub(super) const fn compact(&self) -> &Tensor {
        &self.compact
    }

    pub(super) const fn cu_i64(&self) -> &Tensor {
        &self.cu_i64
    }

    pub(super) const fn checkpoint_cu_i64(&self) -> &Tensor {
        &self.checkpoint_cu_i64
    }

    pub(super) fn checkpoint_compact(&self) -> Option<&Tensor> {
        self.checkpoint_compact.as_ref()
    }

    /// # Safety
    ///
    /// Addresses must refer to live allocations with the plan's fixed shapes.
    pub(super) unsafe fn gather(
        &self,
        stream: &DriverStream,
        pool: usize,
        indices: usize,
        compact: usize,
        pool_size: i32,
    ) -> Result<()> {
        let mut launch = stream.launch_builder(&self.gather);
        let pool = address(pool)?;
        let indices = address(indices)?;
        let compact = address(compact)?;
        launch
            .arg(&pool)
            .arg(&indices)
            .arg(&compact)
            .arg(&self.elements_per_state)
            .arg(&self.total_elements)
            .arg(&pool_size);
        // SAFETY: upheld by the caller.
        unsafe { launch.launch(config(self.state_grid)) }
            .map(|_| ())
            .map_err(|error| message(format!("BF16-to-F32 state gather failed: {error}")))
    }

    /// # Safety
    ///
    /// Addresses must refer to live allocations with the plan's fixed shapes.
    pub(super) unsafe fn scatter(
        &self,
        stream: &DriverStream,
        compact: usize,
        indices: usize,
        pool: usize,
        pool_size: i32,
    ) -> Result<()> {
        let mut launch = stream.launch_builder(&self.scatter);
        let compact = address(compact)?;
        let indices = address(indices)?;
        let pool = address(pool)?;
        launch
            .arg(&compact)
            .arg(&indices)
            .arg(&pool)
            .arg(&self.elements_per_state)
            .arg(&self.total_elements)
            .arg(&pool_size);
        // SAFETY: upheld by the caller.
        unsafe { launch.launch(config(self.state_grid)) }
            .map(|_| ())
            .map_err(|error| message(format!("F32-to-BF16 state scatter failed: {error}")))
    }

    /// # Safety
    ///
    /// Addresses must refer to `[B+1]` int32 and int64 allocations.
    pub(super) unsafe fn convert_cu_seqlens(
        &self,
        stream: &DriverStream,
        input: usize,
        output: usize,
    ) -> Result<()> {
        let mut launch = stream.launch_builder(&self.convert_cu);
        let input = address(input)?;
        let output = address(output)?;
        launch.arg(&input).arg(&output).arg(&self.cu_count);
        // SAFETY: upheld by the caller.
        unsafe { launch.launch(config(self.cu_grid)) }
            .map(|_| ())
            .map_err(|error| {
                message(format!(
                    "int32-to-int64 cu_seqlens conversion failed: {error}"
                ))
            })
    }

    /// # Safety
    ///
    /// Addresses must refer to the plan-sized float32 scratch input and BF16
    /// compact checkpoint output.
    pub(super) unsafe fn cast_checkpoints(
        &self,
        stream: &DriverStream,
        input: usize,
        output: usize,
    ) -> Result<()> {
        if self.checkpoint_elements == 0 {
            return Err(message("checkpoint conversion requires checkpoint scratch"));
        }
        let mut launch = stream.launch_builder(&self.cast_checkpoints);
        let input = address(input)?;
        let output = address(output)?;
        launch
            .arg(&input)
            .arg(&output)
            .arg(&self.checkpoint_elements);
        // SAFETY: upheld by the caller.
        unsafe { launch.launch(config(self.checkpoint_grid)) }
            .map(|_| ())
            .map_err(|error| message(format!("F32-to-BF16 checkpoint cast failed: {error}")))
    }
}

fn address(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| message("CUDA address does not fit u64"))
}

fn grid_size(elements: u64) -> Result<u32> {
    u32::try_from(elements.div_ceil(u64::from(THREADS)))
        .map_err(|_| message("prefill adapter launch grid exceeds CUDA's x dimension"))
}

const fn config(grid: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn adapter_ptx() -> Result<&'static str> {
    let result = ADAPTER_PTX.get_or_init(|| {
        nvrtc::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    match result {
        Ok(ptx) => Ok(ptx),
        Err(error) => Err(message(format!(
            "failed to compile prefill state conversion kernels: {error}"
        ))),
    }
}
