use std::sync::OnceLock;

use candle::cuda_backend::CudaDevice;
use candle::cuda_backend::cudarc::driver::{
    CudaFunction, CudaStream as DriverStream, LaunchConfig, PushKernelArg,
};
use candle::cuda_backend::cudarc::nvrtc;
use candle::{DType, Device, Result, Tensor};

use crate::message;

const MODULE_NAME: &str = "flashinfer_gdn_candle_state_pool_v1";
const GATHER_NAME: &str = "flashinfer_gdn_gather_state_pool_u32";
const SCATTER_NAME: &str = "flashinfer_gdn_scatter_state_pool_u32";
const THREADS: u32 = 256;

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void flashinfer_gdn_gather_state_pool_u32(
    const unsigned int* pool,
    const int* indices,
    unsigned int* compact,
    unsigned long long words_per_state,
    unsigned long long total_words) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_words) return;
  const unsigned long long batch = linear / words_per_state;
  const unsigned long long inner = linear - batch * words_per_state;
  const int pool_index = indices[batch];
  if (pool_index >= 0) {
    compact[linear] = pool[(unsigned long long)pool_index * words_per_state + inner];
  } else {
    compact[linear] = 0;
  }
}

extern "C" __global__ void flashinfer_gdn_scatter_state_pool_u32(
    const unsigned int* compact,
    const int* indices,
    unsigned int* pool,
    unsigned long long words_per_state,
    unsigned long long total_words) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total_words) return;
  const unsigned long long batch = linear / words_per_state;
  const unsigned long long inner = linear - batch * words_per_state;
  const int pool_index = indices[batch];
  if (pool_index >= 0) {
    pool[(unsigned long long)pool_index * words_per_state + inner] = compact[linear];
  }
}
"#;

static POOL_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

pub(crate) fn validate_indexed_state_pool(
    state: &Tensor,
    state_indices: &Tensor,
    dtype: DType,
    state_tail: &[usize],
    batch: usize,
) -> Result<()> {
    let dimensions = state.dims();
    if dimensions.len() != state_tail.len() + 1
        || dimensions[0] == 0
        || dimensions[1..] != *state_tail
    {
        let expected_tail = state_tail
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        return Err(message(format!(
            "state must have shape [P,{expected_tail}] with P>0, found {dimensions:?}"
        )));
    }
    if state.dtype() != dtype || !state.is_contiguous() {
        return Err(message(format!(
            "state must be a contiguous {dtype:?} state pool, found {:?} with strides {:?}",
            state.dtype(),
            state.stride()
        )));
    }
    validate_state_indices(state_indices, batch, "state_indices")
}

pub(crate) fn validate_state_indices(
    indices: &Tensor,
    batch: usize,
    name: &'static str,
) -> Result<()> {
    if indices.dtype() != DType::I32 || indices.dims() != [batch] || !indices.is_contiguous() {
        return Err(message(format!(
            "{name} must be contiguous int32 [{batch}], found {:?} {:?}",
            indices.dtype(),
            indices.dims()
        )));
    }
    Ok(())
}

/// Graph-stable compact state storage used when an upstream kernel only accepts
/// one state per batch item.
///
/// The current BF16 kernels index pools natively. This workspace is retained so
/// future architecture-specific backends can preserve the same public pool-plus-
/// indices contract without adding launch-time allocation.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct IndexedStateWorkspace {
    compact: Tensor,
    gather: CudaFunction,
    scatter: CudaFunction,
    words_per_state: u64,
    total_words: u64,
    grid_blocks: u32,
}

#[allow(dead_code)]
impl IndexedStateWorkspace {
    pub(crate) fn new(
        device: &CudaDevice,
        batch: usize,
        state_tail: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let elements_per_state = state_tail.iter().try_fold(1_usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| message("state workspace element count overflows usize"))
        })?;
        let bytes_per_state = elements_per_state
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| message("state workspace byte count overflows usize"))?;
        if !bytes_per_state.is_multiple_of(size_of::<u32>()) {
            return Err(message(format!(
                "state workspace rows must contain a multiple of four bytes, found {bytes_per_state}"
            )));
        }
        let words_per_state = u64::try_from(bytes_per_state / size_of::<u32>())
            .map_err(|_| message("state workspace row size does not fit u64"))?;
        let total_words = words_per_state
            .checked_mul(
                u64::try_from(batch)
                    .map_err(|_| message("state workspace batch does not fit u64"))?,
            )
            .ok_or_else(|| message("state workspace word count overflows u64"))?;
        let blocks = total_words.div_ceil(u64::from(THREADS));
        let grid_blocks = u32::try_from(blocks)
            .map_err(|_| message("state workspace launch grid exceeds CUDA's x dimension"))?;

        let mut shape = Vec::with_capacity(state_tail.len() + 1);
        shape.push(batch);
        shape.extend_from_slice(state_tail);
        let compact = Tensor::zeros(shape, dtype, &Device::Cuda(device.clone()))?;

        let ptx = pool_ptx()?;
        let gather = device
            .get_or_load_custom_func(GATHER_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();
        let scatter = device
            .get_or_load_custom_func(SCATTER_NAME, MODULE_NAME, ptx)?
            .into_cuda_function();

        Ok(Self {
            compact,
            gather,
            scatter,
            words_per_state,
            total_words,
            grid_blocks,
        })
    }

    pub(crate) const fn compact(&self) -> &Tensor {
        &self.compact
    }

    /// Enqueues a pool-to-compact gather on the plan's stream.
    ///
    /// # Safety
    ///
    /// The addresses must refer to live, non-overlapping CUDA allocations matching
    /// the workspace shape and an int32 index array of length `batch`.
    pub(crate) unsafe fn gather(
        &self,
        stream: &DriverStream,
        pool: usize,
        indices: usize,
        compact: usize,
    ) -> Result<()> {
        // SAFETY: upheld by the caller and the fixed workspace metadata.
        unsafe { self.launch(stream, &self.gather, pool, indices, compact) }
    }

    /// Enqueues a compact-to-pool scatter on the plan's stream.
    ///
    /// # Safety
    ///
    /// The addresses must refer to live, non-overlapping CUDA allocations matching
    /// the workspace shape and an int32 index array of length `batch`.
    pub(crate) unsafe fn scatter(
        &self,
        stream: &DriverStream,
        compact: usize,
        indices: usize,
        pool: usize,
    ) -> Result<()> {
        // SAFETY: upheld by the caller and the fixed workspace metadata.
        unsafe { self.launch(stream, &self.scatter, compact, indices, pool) }
    }

    unsafe fn launch(
        &self,
        stream: &DriverStream,
        function: &CudaFunction,
        source: usize,
        indices: usize,
        destination: usize,
    ) -> Result<()> {
        for (name, address) in [
            ("state copy source", source),
            ("state_indices", indices),
            ("state copy destination", destination),
        ] {
            if !address.is_multiple_of(align_of::<u32>()) {
                return Err(message(format!(
                    "{name} must be 4-byte aligned, found {address:#x}"
                )));
            }
        }
        let source = u64::try_from(source).map_err(|_| message("CUDA address does not fit u64"))?;
        let indices =
            u64::try_from(indices).map_err(|_| message("CUDA address does not fit u64"))?;
        let destination =
            u64::try_from(destination).map_err(|_| message("CUDA address does not fit u64"))?;
        let config = LaunchConfig {
            grid_dim: (self.grid_blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&source)
            .arg(&indices)
            .arg(&destination)
            .arg(&self.words_per_state)
            .arg(&self.total_words);
        // SAFETY: function arguments and allocation lifetimes are guaranteed by
        // the caller; all operations are ordered on `stream`.
        unsafe { launch.launch(config) }
            .map_err(|error| message(format!("state pool copy launch failed: {error}")))?;
        Ok(())
    }
}

fn pool_ptx() -> Result<&'static str> {
    let result = POOL_PTX.get_or_init(|| {
        nvrtc::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
    });
    match result {
        Ok(ptx) => Ok(ptx),
        Err(error) => Err(message(format!(
            "failed to compile Candle state-pool gather/scatter kernels: {error}"
        ))),
    }
}
