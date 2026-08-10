//! Unified Candle dispatch for non-context-parallel GDN prefill.

use std::sync::Mutex;

use candle::cuda_backend::CudaDevice;
use candle::{DType, Device, Result, Tensor};
use flashinfer_gdn::{
    GdnHandle, PrefillBackend, PrefillSm90Plan as CoreSm90Plan, PrefillSm100Plan as CoreSm100Plan,
    PrefillSm120Plan as CoreSm120Plan, PrefillSpecialization,
};

use crate::{core_error, device_architecture, device_multiprocessor_count, message};

mod compact;
mod sm100;
mod state_adapter;

/// Candle inputs shared by all architecture-specific prefill implementations.
pub struct PrefillInputs<'a> {
    /// Main V-major state pool `[P,HV,V,K]`, BF16 and updated in place.
    pub state: &'a Tensor,
    /// Pool slots `[B]`, int32. Entries must be valid and unique.
    pub state_indices: &'a Tensor,
    /// Query `[N,H,K]`, BF16.
    pub q: &'a Tensor,
    /// Key `[N,H,K]`, BF16.
    pub k: &'a Tensor,
    /// Value `[N,HV,V]`, BF16.
    pub v: &'a Tensor,
    /// Multiplicative forget gate `[N,HV]`, float32.
    pub alpha: &'a Tensor,
    /// Update gate `[N,HV]`, float32.
    pub beta: &'a Tensor,
    /// Cumulative sequence lengths `[B+1]`, int32.
    pub cu_seqlens: &'a Tensor,
    /// Compact checkpoint output `[C,HV,V,K]`, BF16 and mutated in place.
    /// Required exactly when the plan came from [`GdnPrefill::prepare_checkpointed`].
    pub state_checkpoints: Option<&'a Tensor>,
    /// Cumulative checkpoint-row offsets `[B+1]`, int32, beginning at zero and
    /// ending at `C`.
    /// Required exactly when the plan came from [`GdnPrefill::prepare_checkpointed`].
    pub checkpoint_cu_starts: Option<&'a Tensor>,
}

/// Model-static dimensions for GDN prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillConfig {
    /// Query/key head count `H`.
    pub query_heads: usize,
    /// Value/state head count `HV`.
    pub value_heads: usize,
    /// Query/key dimension `K`.
    pub key_dim: usize,
    /// Value dimension `V`.
    pub value_dim: usize,
}

impl GdnPrefillConfig {
    /// Constructs a model-static prefill configuration.
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
                "FlashInfer GDN prefill requires key_dim=value_dim=128; found {} and {}",
                self.key_dim, self.value_dim
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefillShape {
    total_tokens: usize,
    batch: usize,
    checkpoint_count: usize,
}

impl PrefillShape {
    fn from_tensors(x: &Tensor, cu_seqlens: &Tensor, checkpoint_count: usize) -> Result<Self> {
        let total_tokens = *x
            .dims()
            .first()
            .ok_or_else(|| message("prefill input x must have a leading token dimension"))?;
        if total_tokens == 0 {
            return Err(message("prefill total token count must be positive"));
        }
        if cu_seqlens.dtype() != DType::I32
            || cu_seqlens.rank() != 1
            || cu_seqlens.dims()[0] < 2
            || !cu_seqlens.is_contiguous()
        {
            return Err(message(format!(
                "cu_seqlens must be contiguous int32 [B+1] with B>0, found {:?} {:?}",
                cu_seqlens.dtype(),
                cu_seqlens.dims()
            )));
        }
        Ok(Self {
            total_tokens,
            batch: cu_seqlens.dims()[0] - 1,
            checkpoint_count,
        })
    }
}

#[derive(Debug, Default)]
struct KernelCache {
    sm90: Vec<(PrefillSpecialization, CoreSm90Plan)>,
    sm100: Vec<(PrefillSpecialization, CoreSm100Plan)>,
    sm120: Vec<(PrefillSpecialization, CoreSm120Plan)>,
}

/// Model-owned GDN prefill runtime.
///
/// Device properties and loaded JIT modules persist here. Each call to
/// [`GdnPrefill::prepare`] creates graph-stable auxiliary storage for one
/// `[N,B]` runtime shape.
#[derive(Debug)]
pub struct GdnPrefill {
    handle: GdnHandle,
    device: CudaDevice,
    device_id: i32,
    gpu_arch: String,
    num_sms: usize,
    backend: PrefillBackend,
    config: GdnPrefillConfig,
    kernels: Mutex<KernelCache>,
}

impl GdnPrefill {
    /// Creates a model-owned prefill runtime and selects the device backend.
    pub fn new(handle: &GdnHandle, device: &Device, config: GdnPrefillConfig) -> Result<Self> {
        config.validate()?;
        let device = device.as_cuda_device()?.clone();
        let ordinal = device.cuda_stream().context().ordinal();
        let device_id = i32::try_from(ordinal)
            .map_err(|_| message(format!("CUDA ordinal does not fit i32: {ordinal}")))?;
        let gpu_arch = device_architecture(&device)?;
        let backend = backend_for_architecture(&gpu_arch)?;
        let num_sms = device_multiprocessor_count(&device)?;
        Ok(Self {
            handle: handle.clone(),
            device,
            device_id,
            gpu_arch,
            num_sms,
            backend,
            config,
            kernels: Mutex::new(KernelCache::default()),
        })
    }

    /// Selects/loads the architecture kernel and allocates a fresh plan for the
    /// leading token dimension of `x` and batch encoded by `cu_seqlens`.
    pub fn prepare(&self, x: &Tensor, cu_seqlens: &Tensor) -> Result<PrefillPlan> {
        self.prepare_inner(x, cu_seqlens, 0, 0)
    }

    /// Selects/loads a checkpoint-emitting kernel and allocates scratch for the
    /// caller-provided total number of compact checkpoint rows.
    pub fn prepare_checkpointed(
        &self,
        x: &Tensor,
        cu_seqlens: &Tensor,
        checkpoint_every_n_tokens: usize,
        checkpoint_count: usize,
    ) -> Result<PrefillPlan> {
        validate_checkpoint_interval(checkpoint_every_n_tokens)?;
        if checkpoint_count == 0 {
            return Err(message("checkpoint_count must be positive"));
        }
        self.prepare_inner(x, cu_seqlens, checkpoint_every_n_tokens, checkpoint_count)
    }

    fn prepare_inner(
        &self,
        x: &Tensor,
        cu_seqlens: &Tensor,
        checkpoint_every_n_tokens: usize,
        checkpoint_count: usize,
    ) -> Result<PrefillPlan> {
        let expected = Device::Cuda(self.device.clone());
        for (name, tensor) in [("x", x), ("cu_seqlens", cu_seqlens)] {
            if !expected.same_device(tensor.device()) {
                return Err(message(format!(
                    "{name} must be on the CUDA device used to create GdnPrefill"
                )));
            }
        }
        if x.dtype() != DType::BF16 {
            return Err(message(format!(
                "prefill input x must be BF16, found {:?}",
                x.dtype()
            )));
        }
        let shape = PrefillShape::from_tensors(x, cu_seqlens, checkpoint_count)?;
        let specialization = PrefillSpecialization::new(
            self.backend,
            self.gpu_arch.clone(),
            self.config.query_heads,
            self.config.value_heads,
            self.config.key_dim,
            self.config.value_dim,
            self.num_sms,
        )
        .map(|spec| spec.checkpoint_every_n_tokens(checkpoint_every_n_tokens))
        .map_err(message)?;
        let backend = match self.backend {
            PrefillBackend::Sm90 => Backend::Compact(compact::Plan::new_sm90(
                self.load_sm90(&specialization)?,
                &self.device,
                shape,
            )?),
            PrefillBackend::Sm100 => Backend::Sm100(sm100::Plan::new(
                self.load_sm100(&specialization)?,
                &self.device,
                shape,
            )?),
            PrefillBackend::Sm120 => Backend::Compact(compact::Plan::new_sm120(
                self.load_sm120(&specialization)?,
                &self.device,
                shape,
            )?),
        };
        Ok(PrefillPlan { backend })
    }

    /// Model-static dimensions.
    #[must_use]
    pub const fn config(&self) -> GdnPrefillConfig {
        self.config
    }

    /// Selected upstream architecture backend.
    #[must_use]
    pub const fn backend(&self) -> PrefillBackend {
        self.backend
    }

    fn load_sm90(&self, spec: &PrefillSpecialization) -> Result<CoreSm90Plan> {
        let mut cache = self
            .kernels
            .lock()
            .map_err(|_| message("prefill kernel cache lock was poisoned"))?;
        if let Some((_, plan)) = cache.sm90.iter().find(|(cached, _)| cached == spec) {
            return Ok(plan.clone());
        }
        let plan = CoreSm90Plan::prepare(&self.handle, spec, self.device_id).map_err(core_error)?;
        cache.sm90.push((spec.clone(), plan.clone()));
        Ok(plan)
    }

    fn load_sm100(&self, spec: &PrefillSpecialization) -> Result<CoreSm100Plan> {
        let mut cache = self
            .kernels
            .lock()
            .map_err(|_| message("prefill kernel cache lock was poisoned"))?;
        if let Some((_, plan)) = cache.sm100.iter().find(|(cached, _)| cached == spec) {
            return Ok(plan.clone());
        }
        let plan =
            CoreSm100Plan::prepare(&self.handle, spec, self.device_id).map_err(core_error)?;
        cache.sm100.push((spec.clone(), plan.clone()));
        Ok(plan)
    }

    fn load_sm120(&self, spec: &PrefillSpecialization) -> Result<CoreSm120Plan> {
        let mut cache = self
            .kernels
            .lock()
            .map_err(|_| message("prefill kernel cache lock was poisoned"))?;
        if let Some((_, plan)) = cache.sm120.iter().find(|(cached, _)| cached == spec) {
            return Ok(plan.clone());
        }
        let plan =
            CoreSm120Plan::prepare(&self.handle, spec, self.device_id).map_err(core_error)?;
        cache.sm120.push((spec.clone(), plan.clone()));
        Ok(plan)
    }
}

#[derive(Debug)]
enum Backend {
    Compact(compact::Plan),
    Sm100(sm100::Plan),
}

/// Prepared prefill plan for one fixed total-token and batch shape.
#[derive(Debug)]
pub struct PrefillPlan {
    backend: Backend,
}

impl PrefillPlan {
    /// Runs prefill, updates selected pool slots, and returns `[N,HV,V]` BF16.
    pub fn forward(&self, inputs: &PrefillInputs<'_>) -> Result<Tensor> {
        validate_input_devices(inputs, self.device())?;
        match &self.backend {
            Backend::Compact(plan) => plan.forward(inputs),
            Backend::Sm100(plan) => plan.forward(inputs),
        }
    }

    /// Fixed total token count.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        match &self.backend {
            Backend::Compact(plan) => plan.shape().total_tokens,
            Backend::Sm100(plan) => plan.shape().total_tokens,
        }
    }

    /// Fixed sequence count.
    #[must_use]
    pub fn batch(&self) -> usize {
        match &self.backend {
            Backend::Compact(plan) => plan.shape().batch,
            Backend::Sm100(plan) => plan.shape().batch,
        }
    }

    /// Fixed number of compact checkpoint rows, or zero when disabled.
    #[must_use]
    pub fn checkpoint_count(&self) -> usize {
        match &self.backend {
            Backend::Compact(plan) => plan.shape().checkpoint_count,
            Backend::Sm100(plan) => plan.shape().checkpoint_count,
        }
    }

    /// Candle CUDA device and stream used by the plan.
    #[must_use]
    pub fn device(&self) -> &CudaDevice {
        match &self.backend {
            Backend::Compact(plan) => plan.device(),
            Backend::Sm100(plan) => plan.device(),
        }
    }
}

fn backend_for_architecture(architecture: &str) -> Result<PrefillBackend> {
    let digits: String = architecture
        .trim_start_matches("sm_")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let numeric = digits.parse::<u32>().map_err(|error| {
        message(format!(
            "failed to parse CUDA architecture {architecture}: {error}"
        ))
    })?;
    match numeric {
        90 => Ok(PrefillBackend::Sm90),
        100 | 103 => Ok(PrefillBackend::Sm100),
        120 | 121 => Ok(PrefillBackend::Sm120),
        _ => Err(message(format!(
            "GDN prefill does not support CUDA architecture {architecture}"
        ))),
    }
}

fn validate_checkpoint_interval(interval: usize) -> Result<()> {
    if interval == 0 || !interval.is_multiple_of(64) {
        return Err(message(format!(
            "checkpoint interval must be a positive multiple of 64, found {interval}"
        )));
    }
    Ok(())
}

fn validate_input_devices(inputs: &PrefillInputs<'_>, device: &CudaDevice) -> Result<()> {
    let expected = Device::Cuda(device.clone());
    for (name, tensor) in [
        ("state", inputs.state),
        ("state_indices", inputs.state_indices),
        ("q", inputs.q),
        ("k", inputs.k),
        ("v", inputs.v),
        ("alpha", inputs.alpha),
        ("beta", inputs.beta),
        ("cu_seqlens", inputs.cu_seqlens),
    ] {
        if !expected.same_device(tensor.device()) {
            return Err(message(format!(
                "{name} must be on the CUDA device used to create GdnPrefill"
            )));
        }
    }
    for (name, tensor) in [
        ("state_checkpoints", inputs.state_checkpoints),
        ("checkpoint_cu_starts", inputs.checkpoint_cu_starts),
    ] {
        if let Some(tensor) = tensor
            && !expected.same_device(tensor.device())
        {
            return Err(message(format!(
                "{name} must be on the CUDA device used to create GdnPrefill"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_checkpoint_inputs(
    inputs: &PrefillInputs<'_>,
    specialization: &PrefillSpecialization,
    batch: usize,
    checkpoint_count: usize,
) -> Result<()> {
    if !specialization.checkpoints_enabled() {
        if inputs.state_checkpoints.is_some() || inputs.checkpoint_cu_starts.is_some() {
            return Err(message(
                "checkpoint tensors require a checkpoint-enabled GdnPrefillConfig",
            ));
        }
        return Ok(());
    }

    let checkpoints = inputs
        .state_checkpoints
        .ok_or_else(|| message("checkpoint-enabled prefill requires state_checkpoints"))?;
    if checkpoints.dtype() != DType::BF16
        || checkpoints.rank() != 4
        || checkpoints.dims()[0] != checkpoint_count
        || checkpoints.dims()[1..] != [specialization.hv, specialization.v, specialization.k]
        || !checkpoints.is_contiguous()
    {
        return Err(message(format!(
            "state_checkpoints must be contiguous BF16 [{checkpoint_count},{},{},{}], found {:?} {:?}",
            specialization.hv,
            specialization.v,
            specialization.k,
            checkpoints.dtype(),
            checkpoints.dims()
        )));
    }

    let checkpoint_cu_starts = inputs
        .checkpoint_cu_starts
        .ok_or_else(|| message("checkpoint-enabled prefill requires checkpoint_cu_starts"))?;
    if checkpoint_cu_starts.dtype() != DType::I32
        || checkpoint_cu_starts.dims() != [batch + 1]
        || !checkpoint_cu_starts.is_contiguous()
    {
        return Err(message(format!(
            "checkpoint_cu_starts must be contiguous int32 [{}], found {:?} {:?}",
            batch + 1,
            checkpoint_cu_starts.dtype(),
            checkpoint_cu_starts.dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_dispatch_is_explicit() {
        assert_eq!(
            backend_for_architecture("sm_90a").unwrap(),
            PrefillBackend::Sm90
        );
        assert_eq!(
            backend_for_architecture("sm_103a").unwrap(),
            PrefillBackend::Sm100
        );
        assert_eq!(
            backend_for_architecture("sm_121a").unwrap(),
            PrefillBackend::Sm120
        );
        assert!(backend_for_architecture("sm_89a").is_err());
    }

    #[test]
    fn checkpoint_interval_matches_upstream_chunking() {
        validate_checkpoint_interval(64).unwrap();
        assert!(validate_checkpoint_interval(0).is_err());
        assert!(validate_checkpoint_interval(96).is_err());
    }
}
