//! Unified Candle dispatch for BF16-state GDN decode.

use candle::cuda_backend::CudaDevice;
use candle::{Result, Tensor};
use flashinfer_gdn::{Bf16StateDecodeCompiler, Bf16StateMtpCompiler};

mod bf16;
mod bf16_mtp;

/// Candle tensors shared by single-token and MTP BF16-state decode.
///
/// The token dimension is one for the single-token backend and must match the
/// compile-time MTP specialization otherwise. `checkpoint_indices` is required
/// only for MTP and is ignored by the single-token backend.
pub struct DecodeInputs<'a> {
    /// Main V-major state pool `[P,HV,V,K]`, BF16 and mutated in place.
    pub state: &'a Tensor,
    /// Log-decay parameter `[HV]`, float32.
    pub a_log: &'a Tensor,
    /// Input-dependent decay `[B,T,HV]`, BF16.
    pub a: &'a Tensor,
    /// Decay bias `[HV]`, float32.
    pub dt_bias: &'a Tensor,
    /// Query `[B,T,H,K]`, BF16.
    pub q: &'a Tensor,
    /// Key `[B,T,H,K]`, BF16.
    pub k: &'a Tensor,
    /// Value `[B,T,HV,V]`, BF16.
    pub v: &'a Tensor,
    /// Update gate `[B,T,HV]`, BF16.
    pub beta: &'a Tensor,
    /// Int32 `[B]` main-pool slots containing each request's input state.
    pub state_indices: &'a Tensor,
    /// Int32 `[B,T]` main-pool slots receiving `h_1` through `h_T` for MTP.
    pub checkpoint_indices: Option<&'a Tensor>,
}

/// Compiler selection used to prepare one decode plan.
#[derive(Debug, Clone, Copy)]
pub enum DecodeCompiler<'a> {
    /// The dedicated single-token BF16-state kernel.
    Bf16(&'a Bf16StateDecodeCompiler),
    /// The checkpointed multi-token BF16-state kernel.
    Bf16Mtp(&'a Bf16StateMtpCompiler),
}

impl<'a> From<&'a Bf16StateDecodeCompiler> for DecodeCompiler<'a> {
    fn from(compiler: &'a Bf16StateDecodeCompiler) -> Self {
        Self::Bf16(compiler)
    }
}

impl<'a> From<&'a Bf16StateMtpCompiler> for DecodeCompiler<'a> {
    fn from(compiler: &'a Bf16StateMtpCompiler) -> Self {
        Self::Bf16Mtp(compiler)
    }
}

#[derive(Debug)]
enum Backend {
    Bf16(bf16::Plan),
    Bf16Mtp(bf16_mtp::Plan),
}

/// Prepared Candle decode plan for either supported BF16-state kernel.
#[derive(Debug)]
pub struct DecodePlan {
    backend: Backend,
}

impl DecodePlan {
    /// Compiles or loads the selected backend for a fixed batch size.
    pub fn prepare<'a>(
        compiler: impl Into<DecodeCompiler<'a>>,
        device: &CudaDevice,
        batch: usize,
    ) -> Result<Self> {
        let backend = match compiler.into() {
            DecodeCompiler::Bf16(compiler) => {
                Backend::Bf16(bf16::Plan::prepare(compiler, device, batch)?)
            }
            DecodeCompiler::Bf16Mtp(compiler) => {
                Backend::Bf16Mtp(bf16_mtp::Plan::prepare(compiler, device, batch)?)
            }
        };
        Ok(Self { backend })
    }

    /// Runs the prepared kernel and returns `[B,T,HV,V]` BF16 output.
    pub fn forward(&self, inputs: &DecodeInputs<'_>) -> Result<Tensor> {
        match &self.backend {
            Backend::Bf16(plan) => plan.forward(inputs),
            Backend::Bf16Mtp(plan) => {
                let checkpoint_indices = inputs
                    .checkpoint_indices
                    .ok_or_else(|| crate::message("BF16 MTP decode requires checkpoint_indices"))?;
                plan.forward(inputs, checkpoint_indices)
            }
        }
    }

    /// Fixed runtime batch size.
    #[must_use]
    pub fn batch(&self) -> usize {
        match &self.backend {
            Backend::Bf16(plan) => plan.batch(),
            Backend::Bf16Mtp(plan) => plan.batch(),
        }
    }

    /// Compile-time token count for this plan.
    #[must_use]
    pub fn tokens(&self) -> usize {
        match &self.backend {
            Backend::Bf16(_) => 1,
            Backend::Bf16Mtp(plan) => plan.specialization().t,
        }
    }

    /// Query scale compiled into this plan.
    #[must_use]
    pub fn scale(&self) -> f32 {
        match &self.backend {
            Backend::Bf16(plan) => plan.specialization().scale,
            Backend::Bf16Mtp(plan) => plan.specialization().scale,
        }
    }

    /// Candle CUDA device and stream used for launches.
    #[must_use]
    pub fn device(&self) -> &CudaDevice {
        match &self.backend {
            Backend::Bf16(plan) => plan.device(),
            Backend::Bf16Mtp(plan) => plan.device(),
        }
    }
}
