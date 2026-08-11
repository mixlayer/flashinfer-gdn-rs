//! Unified Candle dispatch for BF16-state GDN decode.

use std::sync::Mutex;

use candle::cuda_backend::CudaDevice;
use candle::{Device, Result, Tensor};
use flashinfer_gdn::{
    Bf16StateDecodePlan as CoreDecodePlan, Bf16StateDecodeSpecialization,
    Bf16StateMtpPlan as CoreMtpPlan, Bf16StateMtpSpecialization, GdnHandle,
};

use crate::{core_error, device_architecture, device_multiprocessor_count, message};

mod bf16;
mod bf16_mtp;

/// Candle tensors shared by single-token and MTP BF16-state decode.
///
/// A token dimension of one selects the single-token backend; larger token
/// dimensions select MTP. `checkpoint_indices` is required only for MTP and is
/// ignored by the single-token backend.
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

/// Model dimensions shared by every BF16-state decode launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDecodeConfig {
    /// Query/key head count `H`.
    pub query_heads: usize,
    /// Value/state head count `HV`.
    pub value_heads: usize,
    /// Query/key head dimension `K`.
    pub key_dim: usize,
    /// Value head dimension `V`.
    pub value_dim: usize,
}

impl GdnDecodeConfig {
    /// Constructs the model-static decode configuration.
    #[must_use]
    pub const fn new(
        query_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        Self {
            query_heads,
            value_heads,
            key_dim,
            value_dim,
        }
    }

    fn validate(self) -> Result<()> {
        if self.query_heads == 0
            || self.value_heads == 0
            || self.value_heads < self.query_heads
            || !self.value_heads.is_multiple_of(self.query_heads)
        {
            return Err(message(format!(
                "value_heads ({}) must be a positive multiple of query_heads ({})",
                self.value_heads, self.query_heads
            )));
        }
        if self.key_dim != 128 || self.value_dim != 128 {
            return Err(message(format!(
                "FlashInfer BF16-state decode requires key_dim=value_dim=128; found {} and {}",
                self.key_dim, self.value_dim
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeShape {
    batch: usize,
    tokens: usize,
}

impl DecodeShape {
    fn from_tensor(x: &Tensor) -> Result<Self> {
        let dimensions = x.dims();
        if dimensions.len() < 2 {
            return Err(message(format!(
                "decode input x must have leading [B,T] dimensions, found {dimensions:?}"
            )));
        }
        let shape = Self {
            batch: dimensions[0],
            tokens: dimensions[1],
        };
        if shape.batch == 0 || shape.tokens == 0 {
            return Err(message(format!(
                "decode requires positive batch and token dimensions, found B={}, T={}",
                shape.batch, shape.tokens
            )));
        }
        Ok(shape)
    }
}

#[derive(Debug, Default)]
struct KernelCache {
    bf16: Vec<(Bf16StateDecodeSpecialization, CoreDecodePlan)>,
    bf16_mtp: Vec<(Bf16StateMtpSpecialization, CoreMtpPlan)>,
}

/// Model-owned BF16 GDN decode runtime.
///
/// Device properties and loaded kernel modules persist here across forwards.
/// [`GdnDecode::prepare`] selects a runtime shape and allocates fresh plan-owned
/// auxiliary tensors.
#[derive(Debug)]
pub struct GdnDecode {
    handle: GdnHandle,
    device: CudaDevice,
    device_id: i32,
    gpu_arch: String,
    num_sms: usize,
    config: GdnDecodeConfig,
    kernels: Mutex<KernelCache>,
}

impl GdnDecode {
    /// Creates a model-owned decode runtime on `device`.
    ///
    /// GPU architecture, SM count, and device identity are queried once here.
    pub fn new(handle: &GdnHandle, device: &Device, config: GdnDecodeConfig) -> Result<Self> {
        config.validate()?;
        let device = device.as_cuda_device()?.clone();
        let ordinal = device.cuda_stream().context().ordinal();
        let device_id = i32::try_from(ordinal)
            .map_err(|_| message(format!("CUDA ordinal does not fit i32: {ordinal}")))?;
        let gpu_arch = device_architecture(&device)?;
        let num_sms = device_multiprocessor_count(&device)?;
        Ok(Self {
            handle: handle.clone(),
            device,
            device_id,
            gpu_arch,
            num_sms,
            config,
            kernels: Mutex::new(KernelCache::default()),
        })
    }

    /// Selects and loads the kernel for the leading `[B,T]` dimensions of `x`,
    /// then allocates a fresh plan.
    ///
    /// Loaded kernel modules are reused across plans with the same effective
    /// specialization. Batch-sized dummy tensors belong exclusively to the
    /// returned plan.
    pub fn prepare(&self, x: &Tensor) -> Result<DecodePlan> {
        let expected_device = Device::Cuda(self.device.clone());
        if !expected_device.same_device(x.device()) {
            return Err(message(format!(
                "decode input x must be on the CUDA device used to create GdnDecode; found {:?}, expected {:?}",
                x.device().location(),
                expected_device.location()
            )));
        }
        let shape = DecodeShape::from_tensor(x)?;
        let backend = if shape.tokens == 1 {
            let specialization = Bf16StateDecodeSpecialization::new(
                self.gpu_arch.clone(),
                self.config.query_heads,
                self.config.value_heads,
                self.config.key_dim,
                self.config.value_dim,
                shape.batch,
                self.num_sms,
            )
            .map_err(message)?;
            let core = self.load_bf16(&specialization)?;
            Backend::Bf16(bf16::Plan::new(core, &self.device, shape.batch)?)
        } else {
            let specialization = Bf16StateMtpSpecialization::new(
                self.gpu_arch.clone(),
                self.config.query_heads,
                self.config.value_heads,
                self.config.key_dim,
                self.config.value_dim,
                shape.tokens,
                shape.batch,
            )
            .map_err(message)?;
            let core = self.load_bf16_mtp(&specialization)?;
            Backend::Bf16Mtp(bf16_mtp::Plan::new(core, &self.device, shape.batch)?)
        };
        Ok(DecodePlan { backend })
    }

    /// Model-static GDN dimensions.
    #[must_use]
    pub const fn config(&self) -> GdnDecodeConfig {
        self.config
    }

    /// Candle CUDA device and stream used by every prepared plan.
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn load_bf16(&self, specialization: &Bf16StateDecodeSpecialization) -> Result<CoreDecodePlan> {
        let mut kernels = self
            .kernels
            .lock()
            .map_err(|_| message("GDN decode kernel cache lock was poisoned"))?;
        if let Some((_, plan)) = kernels
            .bf16
            .iter()
            .find(|(cached, _)| cached == specialization)
        {
            return Ok(plan.clone());
        }
        let plan = CoreDecodePlan::prepare(&self.handle, specialization, self.device_id)
            .map_err(core_error)?;
        kernels.bf16.push((specialization.clone(), plan.clone()));
        Ok(plan)
    }

    fn load_bf16_mtp(&self, specialization: &Bf16StateMtpSpecialization) -> Result<CoreMtpPlan> {
        let mut kernels = self
            .kernels
            .lock()
            .map_err(|_| message("GDN decode kernel cache lock was poisoned"))?;
        if let Some((_, plan)) = kernels
            .bf16_mtp
            .iter()
            .find(|(cached, _)| cached == specialization)
        {
            return Ok(plan.clone());
        }
        let plan = CoreMtpPlan::prepare(&self.handle, specialization, self.device_id)
            .map_err(core_error)?;
        kernels
            .bf16_mtp
            .push((specialization.clone(), plan.clone()));
        Ok(plan)
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
    /// Runs the prepared kernel and returns `[B,T,HV,V]` BF16 output.
    ///
    /// The returned tensor aliases plan-owned graph-stable storage. Calls on one
    /// plan must be ordered on its CUDA stream and must not execute concurrently.
    pub fn forward(&self, inputs: &DecodeInputs<'_>) -> Result<Tensor> {
        validate_input_devices(inputs, self.device(), self.tokens())?;
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

fn validate_input_devices(
    inputs: &DecodeInputs<'_>,
    plan_device: &CudaDevice,
    tokens: usize,
) -> Result<()> {
    let expected = Device::Cuda(plan_device.clone());
    if !expected.same_device(inputs.q.device()) {
        return Err(message(format!(
            "q must be on the CUDA device used to create GdnDecode; found {:?}, expected {:?}",
            inputs.q.device().location(),
            expected.location()
        )));
    }
    for (name, tensor) in [
        ("state", inputs.state),
        ("a_log", inputs.a_log),
        ("a", inputs.a),
        ("dt_bias", inputs.dt_bias),
        ("k", inputs.k),
        ("v", inputs.v),
        ("beta", inputs.beta),
        ("state_indices", inputs.state_indices),
    ] {
        if !expected.same_device(tensor.device()) {
            return Err(message(format!(
                "{name} must be on the CUDA device used to create GdnDecode; found {:?}, expected {:?}",
                tensor.device().location(),
                expected.location()
            )));
        }
    }
    if tokens > 1 {
        let checkpoint_indices = inputs
            .checkpoint_indices
            .ok_or_else(|| message("BF16 MTP decode requires checkpoint_indices"))?;
        if !expected.same_device(checkpoint_indices.device()) {
            return Err(message(format!(
                "checkpoint_indices must be on the CUDA device used to create GdnDecode; found {:?}, expected {:?}",
                checkpoint_indices.device().location(),
                expected.location()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DecodeShape, GdnDecodeConfig};
    use candle::{Device, Tensor};

    #[test]
    fn validates_model_static_dimensions() {
        GdnDecodeConfig::new(8, 16, 128, 128).validate().unwrap();
        assert!(GdnDecodeConfig::new(8, 12, 128, 128).validate().is_err());
        assert!(GdnDecodeConfig::new(8, 16, 64, 128).validate().is_err());
    }

    #[test]
    fn validates_runtime_shape() {
        let x = Tensor::new(&[[1_u32, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();
        assert_eq!(
            DecodeShape::from_tensor(&x).unwrap(),
            DecodeShape {
                batch: 2,
                tokens: 3
            }
        );
        let rank_one = Tensor::new(&[1_u32, 2], &Device::Cpu).unwrap();
        assert!(DecodeShape::from_tensor(&rank_one).is_err());
        let empty = Tensor::zeros((0, 1), candle::DType::F32, &Device::Cpu).unwrap();
        assert!(DecodeShape::from_tensor(&empty).is_err());
    }
}
