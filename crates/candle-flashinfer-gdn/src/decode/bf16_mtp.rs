use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType as CandleDType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{
    Bf16StateMtpCall, Bf16StateMtpPlan as CorePlan, Bf16StateMtpSpecialization, CudaStream,
};

use super::DecodeInputs;
use crate::{
    core_error, message,
    raw_tensor::{
        RawTensor, base_address, cuda_storage, descriptor, descriptor_parts, ensure_ordinal,
        immutable_address, mutable_address, storage_dtype,
    },
    state_pool::validate_indexed_state_pool,
};

/// Prepared Candle adapter for BF16-state MTP and a fixed batch size.
#[derive(Debug)]
pub(super) struct Plan {
    core: CorePlan,
    device: CudaDevice,
    batch: usize,
    accepted_steps: Tensor,
    output: Tensor,
}

impl Plan {
    /// Binds a persistent loaded kernel and allocates a fresh plan auxiliary.
    pub fn new(core: CorePlan, device: &CudaDevice, batch: usize) -> Result<Self> {
        if batch == 0 {
            return Err(message("MTP batch size must be positive"));
        }
        let candle_device = Device::Cuda(device.clone());
        let accepted_steps = Tensor::zeros(batch, CandleDType::I32, &candle_device)?;
        let spec = core.specialization();
        let output = Tensor::zeros(
            (batch, spec.t, spec.hv, spec.v),
            CandleDType::BF16,
            &candle_device,
        )?;
        Ok(Self {
            core,
            device: device.clone(),
            batch,
            accepted_steps,
            output,
        })
    }

    /// The compile-time specialization selected by this plan.
    #[must_use]
    pub fn specialization(&self) -> &Bf16StateMtpSpecialization {
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

    /// Runs BF16-state MTP into plan-owned graph-stable `[B,T,HV,V]` output.
    pub fn forward(
        &self,
        inputs: &DecodeInputs<'_>,
        checkpoint_indices: &Tensor,
    ) -> Result<Tensor> {
        let spec = self.specialization();
        validate_indexed_state_pool(
            inputs.state,
            inputs.state_indices,
            CandleDType::BF16,
            &[spec.hv, spec.v, spec.k],
            self.batch,
        )?;
        self.validate_mutable_aliases(inputs, checkpoint_indices, &self.output)?;
        inputs.state.inplace_op1(&StateLaunch {
            plan: self,
            inputs,
            checkpoint_indices,
            output: &self.output,
        })?;
        Ok(self.output.clone())
    }

    fn validate_mutable_aliases(
        &self,
        inputs: &DecodeInputs<'_>,
        checkpoint_indices: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
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
            ("checkpoint_indices", checkpoint_indices),
            ("accepted_steps", &self.accepted_steps),
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
    checkpoint_indices: &'a Tensor,
    output: &'a Tensor,
}

impl InplaceOp1 for StateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-bf16-state-mtp-state"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN MTP requires CUDA storage"))
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
        "flashinfer-gdn-bf16-state-mtp-output"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN MTP requires CUDA storage"))
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
        let (checkpoint_indices_storage, checkpoint_indices_layout) =
            self.parent.checkpoint_indices.storage_and_layout();
        let (accepted_storage, accepted_layout) = plan.accepted_steps.storage_and_layout();

        let a_log_storage = cuda_storage(&a_log_storage, "a_log")?;
        let a_storage = cuda_storage(&a_storage, "a")?;
        let dt_bias_storage = cuda_storage(&dt_bias_storage, "dt_bias")?;
        let q_storage = cuda_storage(&q_storage, "q")?;
        let k_storage = cuda_storage(&k_storage, "k")?;
        let v_storage = cuda_storage(&v_storage, "v")?;
        let beta_storage = cuda_storage(&beta_storage, "beta")?;
        let state_indices_storage = cuda_storage(&state_indices_storage, "state_indices")?;
        let checkpoint_indices_storage =
            cuda_storage(&checkpoint_indices_storage, "checkpoint_indices")?;
        let accepted_storage = cuda_storage(&accepted_storage, "accepted_steps")?;
        for (name, storage) in [
            ("a_log", a_log_storage),
            ("a", a_storage),
            ("dt_bias", dt_bias_storage),
            ("q", q_storage),
            ("k", k_storage),
            ("v", v_storage),
            ("beta", beta_storage),
            ("state_indices", state_indices_storage),
            ("checkpoint_indices", checkpoint_indices_storage),
            ("accepted_steps", accepted_storage),
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
        let (checkpoint_indices_ptr, _checkpoint_indices_use) =
            immutable_address(checkpoint_indices_storage, &stream)?;
        let (accepted_ptr, _accepted_use) = immutable_address(accepted_storage, &stream)?;

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
        let checkpoint_indices = descriptor(
            checkpoint_indices_ptr,
            checkpoint_indices_storage,
            checkpoint_indices_layout,
            "checkpoint_indices",
        )?;
        let mut call = Bf16StateMtpCall {
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
            checkpoint_indices: &checkpoint_indices,
        };
        // SAFETY: the stream is owned by the plan's Candle device and remains live.
        let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
        plan.core.launch(&mut call, cuda_stream).map_err(core_error)
    }
}
