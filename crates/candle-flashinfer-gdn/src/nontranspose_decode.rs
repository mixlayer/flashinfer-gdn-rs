use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType as CandleDType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{
    CudaStream, NontransposeDecodeCall, NontransposeDecodeCompiler as CoreCompiler,
    NontransposeDecodePlan as CorePlan, NontransposeDecodeSpecialization,
};

use crate::{
    RawTensor, base_address, core_error, cuda_storage, descriptor, descriptor_parts,
    device_architecture, ensure_ordinal, immutable_address, message, mutable_address,
    storage_dtype,
};

/// Candle tensors consumed by one indexed-pool non-transposed decode.
///
/// `state` is K-major `[P,HV,K,V]` float32 storage and is mutated in place even
/// though Candle tensors use an immutable handle.
pub struct NontransposeDecodeInputs<'a> {
    /// Main state pool `[P,HV,K,V]`, float32.
    pub state: &'a Tensor,
    /// Log-decay parameter `[HV]`, float32.
    pub a_log: &'a Tensor,
    /// Input-dependent decay `[B,1,HV]`.
    pub a: &'a Tensor,
    /// Decay bias `[HV]`.
    pub dt_bias: &'a Tensor,
    /// Query `[B,1,H,K]`.
    pub q: &'a Tensor,
    /// Key `[B,1,H,K]`.
    pub k: &'a Tensor,
    /// Value `[B,1,HV,V]`.
    pub v: &'a Tensor,
    /// Update gate `[B,1,HV]`.
    pub beta: &'a Tensor,
    /// Int32 `[B]` state-pool indices.
    pub state_indices: &'a Tensor,
}

/// Prepared Candle adapter for one specialization and fixed decode batch size.
#[derive(Debug)]
pub struct NontransposeDecodePlan {
    core: CorePlan,
    device: CudaDevice,
    batch: usize,
    cu_seqlens: Tensor,
}

impl NontransposeDecodePlan {
    /// Compiles or loads a specialization and allocates fixed auxiliary tensors.
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
        if !specialization.batch_class.matches(batch) {
            return Err(message(format!(
                "batch {batch} requires the {:?} non-transposed kernel, but the compiler selected {:?}",
                flashinfer_gdn::NontransposeDecodeBatchClass::for_batch(batch),
                specialization.batch_class
            )));
        }
        let core = CorePlan::prepare(compiler, device_id).map_err(core_error)?;
        let candle_device = Device::Cuda(device.clone());
        let cu_seqlens = Tensor::zeros(batch + 1, CandleDType::I32, &candle_device)?;
        Ok(Self {
            core,
            device: device.clone(),
            batch,
            cu_seqlens,
        })
    }

    /// The compile-time specialization selected by this plan.
    #[must_use]
    pub fn specialization(&self) -> &NontransposeDecodeSpecialization {
        self.core.specialization()
    }

    /// Fixed runtime batch size for this plan's auxiliary tensors.
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

    /// Runs non-transposed GDN decode and returns the BF16 output.
    pub fn forward(&self, inputs: &NontransposeDecodeInputs<'_>) -> Result<Tensor> {
        let output = self.empty_output()?;
        self.execute(inputs, &output)?;
        Ok(output)
    }

    fn execute(&self, inputs: &NontransposeDecodeInputs<'_>, output: &Tensor) -> Result<()> {
        self.validate_mutable_aliases(inputs, output)?;
        inputs.state.inplace_op1(&StateLaunch {
            plan: self,
            inputs,
            output,
        })
    }

    // This prevents recursively locking an aliased Candle allocation at the FFI boundary.
    fn validate_mutable_aliases(
        &self,
        inputs: &NontransposeDecodeInputs<'_>,
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
            ("cu_seqlens", &self.cu_seqlens),
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
    plan: &'a NontransposeDecodePlan,
    inputs: &'a NontransposeDecodeInputs<'a>,
    output: &'a Tensor,
}

impl InplaceOp1 for StateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-nontranspose-decode-state"
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
        "flashinfer-gdn-nontranspose-decode-output"
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
        let (cu_seqlens_storage, cu_seqlens_layout) = plan.cu_seqlens.storage_and_layout();

        let a_log_storage = cuda_storage(&a_log_storage, "a_log")?;
        let a_storage = cuda_storage(&a_storage, "a")?;
        let dt_bias_storage = cuda_storage(&dt_bias_storage, "dt_bias")?;
        let q_storage = cuda_storage(&q_storage, "q")?;
        let k_storage = cuda_storage(&k_storage, "k")?;
        let v_storage = cuda_storage(&v_storage, "v")?;
        let beta_storage = cuda_storage(&beta_storage, "beta")?;
        let state_indices_storage = cuda_storage(&state_indices_storage, "state_indices")?;
        let cu_seqlens_storage = cuda_storage(&cu_seqlens_storage, "cu_seqlens")?;

        for (name, storage) in [
            ("a_log", a_log_storage),
            ("a", a_storage),
            ("dt_bias", dt_bias_storage),
            ("q", q_storage),
            ("k", k_storage),
            ("v", v_storage),
            ("beta", beta_storage),
            ("state_indices", state_indices_storage),
            ("cu_seqlens", cu_seqlens_storage),
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
        let (cu_seqlens_ptr, _cu_seqlens_use) = immutable_address(cu_seqlens_storage, &stream)?;

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
        let cu_seqlens = descriptor(
            cu_seqlens_ptr,
            cu_seqlens_storage,
            cu_seqlens_layout,
            "cu_seqlens",
        )?;
        let mut call = NontransposeDecodeCall {
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
            cu_seqlens: &cu_seqlens,
        };
        // SAFETY: the driver stream is owned by the plan's Candle device and remains
        // alive for the duration of the launch.
        let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
        plan.core.launch(&mut call, cuda_stream).map_err(core_error)
    }
}
