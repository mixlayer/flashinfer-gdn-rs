#![deny(unsafe_op_in_unsafe_fn)]
//! Candle CUDA integration for FlashInfer BF16-state GDN decode.
//!
//! Preparation performs all compilation, loading, and auxiliary allocation before
//! CUDA Graph capture. Eager execution and capture use each plan's `forward`
//! method; graph owners are responsible for their normal warmup, fixed-address,
//! and replay lifecycle.

use candle::Result;
use candle::cuda_backend::CudaDevice;
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute;

pub mod decode;
mod raw_tensor;
mod state_pool;

pub use decode::{DecodeInputs, DecodePlan, GdnDecode, GdnDecodeConfig};
pub use flashinfer_gdn::GdnHandle;

fn device_architecture(device: &CudaDevice) -> Result<String> {
    let (major, minor) = device
        .cuda_stream()
        .context()
        .compute_capability()
        .map_err(|error| message(format!("failed to query CUDA compute capability: {error}")))?;
    if major < 9 {
        return Err(message(format!(
            "FlashInfer GDN requires compute capability 9.0 or newer, found {major}.{minor}"
        )));
    }
    Ok(format!("sm_{major}{minor}a"))
}

fn device_multiprocessor_count(device: &CudaDevice) -> Result<usize> {
    let count = device
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|error| message(format!("failed to query CUDA SM count: {error}")))?;
    usize::try_from(count).map_err(|_| message(format!("invalid CUDA SM count {count}")))
}

fn core_error(error: flashinfer_gdn::Error) -> candle::Error {
    message(error)
}

fn message(message: impl ToString) -> candle::Error {
    candle::Error::Msg(message.to_string())
}
