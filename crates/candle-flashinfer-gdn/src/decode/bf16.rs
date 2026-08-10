use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType as CandleDType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{
    Bf16StateDecodeCall, Bf16StateDecodeCompiler as CoreCompiler, Bf16StateDecodePlan as CorePlan,
    Bf16StateDecodeSpecialization, CudaStream,
};

use super::DecodeInputs;
use crate::{
    RawTensor, base_address, core_error, cuda_storage, descriptor, descriptor_parts,
    device_architecture, device_multiprocessor_count, ensure_ordinal, immutable_address, message,
    mutable_address, state_pool::validate_indexed_state_pool, storage_dtype,
};

/// Prepared Candle adapter for BF16-state T=1 decode and a fixed batch size.
#[derive(Debug)]
pub(super) struct Plan {
    core: CorePlan,
    device: CudaDevice,
    batch: usize,
    accepted_steps: Tensor,
    ssm_state_indices: Tensor,
}

impl Plan {
    /// Compiles or loads the upstream-selected specialization and allocates auxiliaries.
    pub fn prepare(compiler: &CoreCompiler, device: &CudaDevice, batch: usize) -> Result<Self> {
        if batch == 0 {
            return Err(message("decode batch size must be positive"));
        }
        let ordinal = device.cuda_stream().context().ordinal();
        let device_id = i32::try_from(ordinal)
            .map_err(|_| message(format!("CUDA ordinal does not fit i32: {ordinal}")))?;
        let expected_arch = device_architecture(device)?;
        let specialization = compiler.selected_specialization();
        if specialization.gpu_arch != expected_arch {
            return Err(message(format!(
                "compiler specialization targets {}, but CUDA device {ordinal} requires {expected_arch}",
                specialization.gpu_arch
            )));
        }
        let num_sms = device_multiprocessor_count(device)?;
        if !specialization
            .matches_runtime(batch, num_sms)
            .map_err(message)?
        {
            return Err(message(format!(
                "BF16-state specialization {:?}/tile_v={} does not match upstream dispatch for batch {batch}, HV={}, and {num_sms} SMs",
                specialization.variant, specialization.tile_v, specialization.hv
            )));
        }

        let core = CorePlan::prepare(compiler, device_id).map_err(core_error)?;
        let candle_device = Device::Cuda(device.clone());
        let accepted_steps = Tensor::zeros(batch, CandleDType::I32, &candle_device)?;
        let ssm_state_indices = Tensor::zeros((batch, 1), CandleDType::I32, &candle_device)?;
        Ok(Self {
            core,
            device: device.clone(),
            batch,
            accepted_steps,
            ssm_state_indices,
        })
    }

    /// The compile-time specialization selected by this plan.
    #[must_use]
    pub fn specialization(&self) -> &Bf16StateDecodeSpecialization {
        self.core.specialization()
    }

    /// Fixed runtime batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Candle CUDA device and stream used for launches.
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn empty_output(&self) -> Result<Tensor> {
        let spec = self.specialization();
        Tensor::zeros(
            (self.batch, 1, spec.hv, spec.v),
            CandleDType::BF16,
            &Device::Cuda(self.device.clone()),
        )
    }

    /// Runs BF16-state GDN decode and returns the BF16 output.
    pub fn forward(&self, inputs: &DecodeInputs<'_>) -> Result<Tensor> {
        let spec = self.specialization();
        validate_indexed_state_pool(
            inputs.state,
            inputs.state_indices,
            CandleDType::BF16,
            &[spec.hv, spec.v, spec.k],
            self.batch,
        )?;
        let output = self.empty_output()?;
        self.validate_mutable_aliases(inputs, &output)?;
        inputs.state.inplace_op1(&StateLaunch {
            plan: self,
            inputs,
            output: &output,
        })?;
        Ok(output)
    }

    fn validate_mutable_aliases(&self, inputs: &DecodeInputs<'_>, output: &Tensor) -> Result<()> {
        let stream = self.device.cuda_stream();
        let state = base_address(inputs.state, &stream, "state")?;
        let output_address = base_address(output, &stream, "output")?;
        if state == output_address {
            return Err(message("output and state must not share CUDA storage"));
        }
        for (name, tensor) in [
            ("a_log", inputs.a_log),
            ("a", inputs.a),
            ("dt_bias", inputs.dt_bias),
            ("q", inputs.q),
            ("k", inputs.k),
            ("v", inputs.v),
            ("beta", inputs.beta),
            ("state_indices", inputs.state_indices),
            ("accepted_steps", &self.accepted_steps),
            ("ssm_state_indices", &self.ssm_state_indices),
        ] {
            let address = base_address(tensor, &stream, name)?;
            if address == state {
                return Err(message(format!(
                    "{name} and state must not share CUDA storage"
                )));
            }
            if address == output_address {
                return Err(message(format!(
                    "{name} and output must not share CUDA storage"
                )));
            }
        }
        Ok(())
    }
}

struct StateLaunch<'a> {
    plan: &'a Plan,
    inputs: &'a DecodeInputs<'a>,
    output: &'a Tensor,
}

impl InplaceOp1 for StateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-bf16-state-decode-state"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        ensure_ordinal(storage, self.plan.core.device_id(), "state")?;
        let state_dtype = storage_dtype(storage)?;
        let stream = self.plan.device.cuda_stream();
        let (state_ptr, _state_use) = mutable_address(storage, &stream)?;
        self.output.inplace_op1(&OutputLaunch {
            parent: self,
            state: RawTensor::new(
                state_ptr,
                state_dtype,
                layout,
                self.plan.core.device_id(),
                "state",
            )?,
        })
    }
}

struct OutputLaunch<'a, 'b> {
    parent: &'a StateLaunch<'b>,
    state: RawTensor,
}

impl InplaceOp1 for OutputLaunch<'_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-bf16-state-decode-output"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN decode requires CUDA storage"))
    }

    #[allow(clippy::too_many_lines)]
    fn cuda_fwd(&self, output_storage: &mut CudaStorage, output_layout: &Layout) -> Result<()> {
        let plan = self.parent.plan;
        ensure_ordinal(output_storage, plan.core.device_id(), "output")?;
        let output_dtype = storage_dtype(output_storage)?;
        let stream = plan.device.cuda_stream();
        let (output_ptr, _output_use) = mutable_address(output_storage, &stream)?;

        let (a_log_storage, a_log_layout) = self.parent.inputs.a_log.storage_and_layout();
        let (a_storage, a_layout) = self.parent.inputs.a.storage_and_layout();
        let (dt_bias_storage, dt_bias_layout) = self.parent.inputs.dt_bias.storage_and_layout();
        let (q_storage, q_layout) = self.parent.inputs.q.storage_and_layout();
        let (k_storage, k_layout) = self.parent.inputs.k.storage_and_layout();
        let (v_storage, v_layout) = self.parent.inputs.v.storage_and_layout();
        let (beta_storage, beta_layout) = self.parent.inputs.beta.storage_and_layout();
        let (state_indices_storage, state_indices_layout) =
            self.parent.inputs.state_indices.storage_and_layout();
        let (accepted_storage, accepted_layout) = plan.accepted_steps.storage_and_layout();
        let (scatter_storage, scatter_layout) = plan.ssm_state_indices.storage_and_layout();

        let a_log_storage = cuda_storage(&a_log_storage, "a_log")?;
        let a_storage = cuda_storage(&a_storage, "a")?;
        let dt_bias_storage = cuda_storage(&dt_bias_storage, "dt_bias")?;
        let q_storage = cuda_storage(&q_storage, "q")?;
        let k_storage = cuda_storage(&k_storage, "k")?;
        let v_storage = cuda_storage(&v_storage, "v")?;
        let beta_storage = cuda_storage(&beta_storage, "beta")?;
        let state_indices_storage = cuda_storage(&state_indices_storage, "state_indices")?;
        let accepted_storage = cuda_storage(&accepted_storage, "accepted_steps")?;
        let scatter_storage = cuda_storage(&scatter_storage, "ssm_state_indices")?;
        for (name, storage) in [
            ("a_log", a_log_storage),
            ("a", a_storage),
            ("dt_bias", dt_bias_storage),
            ("q", q_storage),
            ("k", k_storage),
            ("v", v_storage),
            ("beta", beta_storage),
            ("state_indices", state_indices_storage),
            ("accepted_steps", accepted_storage),
            ("ssm_state_indices", scatter_storage),
        ] {
            ensure_ordinal(storage, plan.core.device_id(), name)?;
        }

        let (a_log_ptr, _a_log_use) = immutable_address(a_log_storage, &stream)?;
        let (a_ptr, _a_use) = immutable_address(a_storage, &stream)?;
        let (dt_bias_ptr, _dt_bias_use) = immutable_address(dt_bias_storage, &stream)?;
        let (q_ptr, _q_use) = immutable_address(q_storage, &stream)?;
        let (k_ptr, _k_use) = immutable_address(k_storage, &stream)?;
        let (v_ptr, _v_use) = immutable_address(v_storage, &stream)?;
        let (beta_ptr, _beta_use) = immutable_address(beta_storage, &stream)?;
        let (state_indices_ptr, _state_indices_use) =
            immutable_address(state_indices_storage, &stream)?;
        let (accepted_ptr, _accepted_use) = immutable_address(accepted_storage, &stream)?;
        let (scatter_ptr, _scatter_use) = immutable_address(scatter_storage, &stream)?;

        let mut state = self.state.descriptor()?;
        let a_log = descriptor(a_log_ptr, a_log_storage, a_log_layout, "a_log")?;
        let a = descriptor(a_ptr, a_storage, a_layout, "a")?;
        let dt_bias = descriptor(dt_bias_ptr, dt_bias_storage, dt_bias_layout, "dt_bias")?;
        let q = descriptor(q_ptr, q_storage, q_layout, "q")?;
        let k = descriptor(k_ptr, k_storage, k_layout, "k")?;
        let v = descriptor(v_ptr, v_storage, v_layout, "v")?;
        let beta = descriptor(beta_ptr, beta_storage, beta_layout, "beta")?;
        let mut output = descriptor_parts(
            output_ptr,
            output_dtype,
            output_layout,
            plan.core.device_id(),
            "output",
        )?;
        let state_indices = descriptor(
            state_indices_ptr,
            state_indices_storage,
            state_indices_layout,
            "state_indices",
        )?;
        let accepted_steps = descriptor(
            accepted_ptr,
            accepted_storage,
            accepted_layout,
            "accepted_steps",
        )?;
        let ssm_state_indices = descriptor(
            scatter_ptr,
            scatter_storage,
            scatter_layout,
            "ssm_state_indices",
        )?;
        let mut call = Bf16StateDecodeCall {
            state: &mut state,
            a_log: &a_log,
            a: &a,
            dt_bias: &dt_bias,
            q: &q,
            k: &k,
            v: &v,
            beta: &beta,
            output: &mut output,
            state_indices: &state_indices,
            accepted_steps: &accepted_steps,
            ssm_state_indices: &ssm_state_indices,
        };
        // SAFETY: the stream is owned by the plan's Candle device and remains live.
        let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
        plan.core.launch(&mut call, cuda_stream).map_err(core_error)
    }
}
