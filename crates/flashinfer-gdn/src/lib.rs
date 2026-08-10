#![deny(unsafe_op_in_unsafe_fn)]
//! Framework-independent safe APIs for FlashInfer BF16-state GDN decode.
//!
//! A framework adapter creates [`CudaTensor`] descriptors while it holds the
//! framework's storage guards. Plans validate all cross-tensor contracts before
//! crossing the generated TVM FFI boundary and keep compiled modules alive for
//! eager launches and CUDA Graph executions.

mod bf16_state_decode;
mod bf16_state_mtp;
mod error;
mod tensor;
mod validation;

pub use bf16_state_decode::{Bf16StateDecodeCall, Bf16StateDecodePlan, validate_bf16_state_decode};
pub use bf16_state_mtp::{Bf16StateMtpCall, Bf16StateMtpPlan, validate_bf16_state_mtp};
pub use error::{Error, Result};
pub use flashinfer_gdn_sys::{
    Bf16StateDecodeCompiler, Bf16StateDecodeKernelVariant, Bf16StateDecodeSpecialization,
    Bf16StateMtpCompiler, Bf16StateMtpKernelVariant, Bf16StateMtpSpecialization, DtBiasDType,
    InputDType,
};
pub use tensor::{CudaStream, CudaTensor, DType};
