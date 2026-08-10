#![deny(unsafe_op_in_unsafe_fn)]
//! Framework-independent safe APIs for FlashInfer GDN decode and prefill.
//!
//! A framework adapter creates [`CudaTensor`] descriptors while it holds the
//! framework's storage guards. Plans validate all cross-tensor contracts before
//! crossing the generated TVM FFI boundary and keep compiled modules alive for
//! eager launches and CUDA Graph executions.

mod bf16_state_decode;
mod bf16_state_mtp;
mod error;
mod prefill;
mod tensor;
mod validation;

pub use bf16_state_decode::{Bf16StateDecodeCall, Bf16StateDecodePlan, validate_bf16_state_decode};
pub use bf16_state_mtp::{Bf16StateMtpCall, Bf16StateMtpPlan, validate_bf16_state_mtp};
pub use error::{Error, Result};
pub use flashinfer_gdn_sys::{
    Bf16StateDecodeKernelVariant, Bf16StateDecodeSpecialization, Bf16StateMtpKernelVariant,
    Bf16StateMtpSpecialization, DtBiasDType, GdnHandle, InputDType, PrefillBackend,
    PrefillSpecialization,
};
pub use prefill::{
    PrefillCompactCall, PrefillSm90Plan, PrefillSm100Call, PrefillSm100Plan, PrefillSm120Plan,
    validate_prefill_compact, validate_prefill_sm100,
};
pub use tensor::{CudaStream, CudaTensor, DType};
