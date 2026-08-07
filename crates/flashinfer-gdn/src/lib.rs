#![deny(unsafe_op_in_unsafe_fn)]
//! Framework-independent safe APIs for FlashInfer GDN kernels.
//!
//! A framework adapter creates [`CudaTensor`] descriptors while it holds the
//! framework's storage guards. Plans validate all cross-tensor contracts before
//! crossing the generated TVM FFI boundary and keep compiled modules alive for
//! eager launches and CUDA Graph executions.

mod decode;
mod error;
mod nontranspose_decode;
mod tensor;

pub use decode::{PretransposeDecodeCall, PretransposeDecodePlan, validate_pretranspose_decode};
pub use error::{Error, Result};
pub use flashinfer_gdn_sys::{
    DtBiasDType, InputDType, NontransposeDecodeBatchClass, NontransposeDecodeCompiler,
    NontransposeDecodeSpecialization, PretransposeDecodeCompiler, PretransposeDecodeSpecialization,
};
pub use nontranspose_decode::{
    NontransposeDecodeCall, NontransposeDecodePlan, validate_nontranspose_decode,
};
pub use tensor::{CudaStream, CudaTensor, DType};
