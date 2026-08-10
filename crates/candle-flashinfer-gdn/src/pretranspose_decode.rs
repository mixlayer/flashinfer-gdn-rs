use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType as CandleDType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{
    CudaStream, PretransposeDecodeCall, PretransposeDecodeCompiler as CoreCompiler,
    PretransposeDecodePlan as CorePlan, PretransposeDecodeSpecialization,
};

use crate::{
    RawTensor, base_address, core_error, cuda_storage, descriptor, descriptor_parts,
    device_architecture, effective_address, ensure_ordinal, immutable_address, message,
    mutable_address,
    state_pool::{IndexedStateWorkspace, validate_indexed_state_pool, validate_state_indices},
    storage_dtype,
};

/// Candle tensors consumed by one pretransposed float-state decode.
///
/// `state` is always an indexed pool and is mutated in place even though Candle
/// tensors use an immutable handle. `output_state_indices` selects a distinct write
/// destination when supplied; otherwise the read indices are reused for writes.
pub struct PretransposeDecodeInputs<'a> {
    /// Main state pool `[P,HV,V,K]`, float32.
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
    /// Required int32 `[B]` state-pool read indices.
    pub state_indices: &'a Tensor,
    /// Optional int32 `[B]` state-pool write indices.
    pub output_state_indices: Option<&'a Tensor>,
}

/// Prepared Candle adapter for one specialization and fixed decode batch size.
#[derive(Debug)]
pub struct PretransposeDecodePlan {
    core: CorePlan,
    device: CudaDevice,
    batch: usize,
    direct_state_indices: Tensor,
    direct_output_state_indices: Tensor,
    cu_seqlens: Tensor,
    state_workspace: Option<IndexedStateWorkspace>,
}

impl PretransposeDecodePlan {
    /// Compiles or loads a specialization and allocates graph-stable auxiliaries.
    ///
    /// Call this before capture. The compiler architecture must exactly match the
    /// CUDA device's compute capability.
    pub fn prepare(compiler: &CoreCompiler, device: &CudaDevice, batch: usize) -> Result<Self> {
        if batch == 0 {
            return Err(message("decode batch size must be positive"));
        }
        let ordinal = device.cuda_stream().context().ordinal();
        let device_id = i32::try_from(ordinal)
            .map_err(|_| message(format!("CUDA ordinal does not fit i32: {ordinal}")))?;
        let expected_arch = device_architecture(device)?;
        let selected_arch = &compiler.selected_specialization().gpu_arch;
        if selected_arch != &expected_arch {
            return Err(message(format!(
                "compiler specialization targets {selected_arch}, but CUDA device {ordinal} requires {expected_arch}"
            )));
        }

        let core = CorePlan::prepare(compiler, device_id).map_err(core_error)?;
        let candle_device = Device::Cuda(device.clone());
        let direct_state_indices = Tensor::zeros(batch, CandleDType::I32, &candle_device)?;
        let direct_output_state_indices = Tensor::zeros(batch, CandleDType::I32, &candle_device)?;
        let cu_seqlens = Tensor::zeros(batch + 1, CandleDType::I32, &candle_device)?;
        let state_workspace = if compiler.selected_specialization().use_pool_indexing {
            None
        } else {
            Some(IndexedStateWorkspace::new(
                device,
                batch,
                &[
                    compiler.selected_specialization().hv,
                    compiler.selected_specialization().v,
                    compiler.selected_specialization().k,
                ],
                CandleDType::F32,
            )?)
        };
        Ok(Self {
            core,
            device: device.clone(),
            batch,
            direct_state_indices,
            direct_output_state_indices,
            cu_seqlens,
            state_workspace,
        })
    }

    /// The compile-time specialization selected by this plan.
    #[must_use]
    pub fn specialization(&self) -> &PretransposeDecodeSpecialization {
        self.core.specialization()
    }

    /// Fixed runtime batch size for this plan's graph-stable auxiliaries.
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

    /// Runs pretransposed GDN decode and returns the BF16 output.
    ///
    /// This is the only execution path for both eager execution and CUDA Graph capture.
    /// A graph owner using Candle must disable its event tracker before capture,
    /// as `modeld-core`'s graph runtime does, and must keep this plan and the
    /// captured tensor allocations alive for the lifetime of the graph.
    pub fn forward(&self, inputs: &PretransposeDecodeInputs<'_>) -> Result<Tensor> {
        self.validate_pool_contract(inputs)?;
        let output = self.empty_output()?;
        self.execute(inputs, &output)?;
        Ok(output)
    }

    fn execute(&self, inputs: &PretransposeDecodeInputs<'_>, output: &Tensor) -> Result<()> {
        let (state_indices, output_state_indices) = self.indices(inputs);
        self.validate_mutable_aliases(inputs, output, state_indices, output_state_indices)?;
        if self.specialization().use_pool_indexing {
            inputs.state.inplace_op1(&StateLaunch {
                plan: self,
                inputs,
                output,
                state_indices,
                output_state_indices,
            })
        } else {
            inputs.state.inplace_op1(&PoolStateLaunch {
                plan: self,
                inputs,
                output,
                output_state_indices,
            })
        }
    }

    fn indices<'a>(&'a self, inputs: &'a PretransposeDecodeInputs<'a>) -> (&'a Tensor, &'a Tensor) {
        (
            inputs.state_indices,
            inputs.output_state_indices.unwrap_or(inputs.state_indices),
        )
    }

    fn validate_pool_contract(&self, inputs: &PretransposeDecodeInputs<'_>) -> Result<()> {
        let spec = self.specialization();
        validate_indexed_state_pool(
            inputs.state,
            inputs.state_indices,
            CandleDType::F32,
            &[spec.hv, spec.v, spec.k],
            self.batch,
        )?;
        validate_state_indices(
            inputs.output_state_indices.unwrap_or(inputs.state_indices),
            self.batch,
            "output_state_indices",
        )?;
        Ok(())
    }

    // This is an FFI safety check rather than CUDA Graph preparation. It runs
    // before taking Candle's mutable storage locks so aliased views return an
    // error instead of recursively locking the same storage.
    fn validate_mutable_aliases(
        &self,
        inputs: &PretransposeDecodeInputs<'_>,
        output: &Tensor,
        state_indices: &Tensor,
        output_state_indices: &Tensor,
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
            ("state_indices", state_indices),
            ("output_state_indices", output_state_indices),
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
    plan: &'a PretransposeDecodePlan,
    inputs: &'a PretransposeDecodeInputs<'a>,
    output: &'a Tensor,
    state_indices: &'a Tensor,
    output_state_indices: &'a Tensor,
}

impl InplaceOp1 for StateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-pretranspose-decode-state"
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
            plan: self.plan,
            inputs: self.inputs,
            state: RawTensor::new(
                state_ptr,
                state_dtype,
                layout,
                self.plan.core.device_id(),
                "state",
            )?,
            state_indices: self.state_indices,
            output_state_indices: self.output_state_indices,
        })
    }
}

struct PoolStateLaunch<'a> {
    plan: &'a PretransposeDecodePlan,
    inputs: &'a PretransposeDecodeInputs<'a>,
    output: &'a Tensor,
    output_state_indices: &'a Tensor,
}

impl InplaceOp1 for PoolStateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-pretranspose-decode-state-pool"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        ensure_ordinal(storage, self.plan.core.device_id(), "state")?;
        let state_dtype = storage_dtype(storage)?;
        let stream = self.plan.device.cuda_stream();
        let (state_ptr, _state_use) = mutable_address(storage, &stream)?;
        let workspace = self.plan.state_workspace.as_ref().ok_or_else(|| {
            message("direct-state pretranspose specialization has no state workspace")
        })?;
        workspace.compact().inplace_op1(&CompactStateLaunch {
            parent: self,
            pool: RawTensor::new(
                state_ptr,
                state_dtype,
                layout,
                self.plan.core.device_id(),
                "state",
            )?,
        })
    }
}

struct CompactStateLaunch<'a, 'b> {
    parent: &'a PoolStateLaunch<'b>,
    pool: RawTensor,
}

impl InplaceOp1 for CompactStateLaunch<'_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-pretranspose-decode-compact-state"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN decode requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "compact_state")?;
        let compact_dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (compact_ptr, _compact_use) = mutable_address(storage, &stream)?;

        let (read_indices_storage, read_indices_layout) =
            self.parent.inputs.state_indices.storage_and_layout();
        let (write_indices_storage, write_indices_layout) =
            self.parent.output_state_indices.storage_and_layout();
        let read_indices_storage = cuda_storage(&read_indices_storage, "state_indices")?;
        let write_indices_storage = cuda_storage(&write_indices_storage, "output_state_indices")?;
        ensure_ordinal(read_indices_storage, plan.core.device_id(), "state_indices")?;
        ensure_ordinal(
            write_indices_storage,
            plan.core.device_id(),
            "output_state_indices",
        )?;
        let (read_indices_ptr, _read_indices_use) =
            immutable_address(read_indices_storage, &stream)?;
        let (write_indices_ptr, _write_indices_use) =
            immutable_address(write_indices_storage, &stream)?;

        let pool = self.pool.effective_address("state")?;
        let read_indices = effective_address(
            read_indices_ptr,
            storage_dtype(read_indices_storage)?,
            read_indices_layout,
            "state_indices",
        )?;
        let write_indices = effective_address(
            write_indices_ptr,
            storage_dtype(write_indices_storage)?,
            write_indices_layout,
            "output_state_indices",
        )?;
        let compact = effective_address(compact_ptr, compact_dtype, layout, "compact_state")?;
        let workspace = plan.state_workspace.as_ref().ok_or_else(|| {
            message("direct-state pretranspose specialization has no state workspace")
        })?;

        // SAFETY: the surrounding Candle guards keep the validated pool, index,
        // and compact workspace allocations live on this stream.
        unsafe {
            workspace.gather(&stream, pool, read_indices, compact)?;
        }

        self.parent.output.inplace_op1(&OutputLaunch {
            plan,
            inputs: self.parent.inputs,
            state: RawTensor::new(
                compact_ptr,
                compact_dtype,
                layout,
                plan.core.device_id(),
                "compact_state",
            )?,
            state_indices: &plan.direct_state_indices,
            output_state_indices: &plan.direct_output_state_indices,
        })?;

        // SAFETY: FlashInfer's compact-state update is ordered before this
        // writeback on the same stream and all allocation guards remain live.
        unsafe {
            workspace.scatter(&stream, compact, write_indices, pool)?;
        }
        Ok(())
    }
}

struct OutputLaunch<'a> {
    plan: &'a PretransposeDecodePlan,
    inputs: &'a PretransposeDecodeInputs<'a>,
    state: RawTensor,
    state_indices: &'a Tensor,
    output_state_indices: &'a Tensor,
}

impl InplaceOp1 for OutputLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-pretranspose-decode-output"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN decode requires CUDA storage"))
    }

    #[allow(clippy::too_many_lines)]
    fn cuda_fwd(&self, output_storage: &mut CudaStorage, output_layout: &Layout) -> Result<()> {
        let plan = self.plan;
        ensure_ordinal(output_storage, plan.core.device_id(), "output")?;
        let output_dtype = storage_dtype(output_storage)?;
        let stream = plan.device.cuda_stream();
        let (output_ptr, _output_use) = mutable_address(output_storage, &stream)?;

        let (a_log_storage, a_log_layout) = self.inputs.a_log.storage_and_layout();
        let (a_storage, a_layout) = self.inputs.a.storage_and_layout();
        let (dt_bias_storage, dt_bias_layout) = self.inputs.dt_bias.storage_and_layout();
        let (q_storage, q_layout) = self.inputs.q.storage_and_layout();
        let (k_storage, k_layout) = self.inputs.k.storage_and_layout();
        let (v_storage, v_layout) = self.inputs.v.storage_and_layout();
        let (beta_storage, beta_layout) = self.inputs.beta.storage_and_layout();
        let (state_indices_storage, state_indices_layout) = self.state_indices.storage_and_layout();
        let (output_indices_storage, output_indices_layout) =
            self.output_state_indices.storage_and_layout();
        let (cu_seqlens_storage, cu_seqlens_layout) = plan.cu_seqlens.storage_and_layout();

        let a_log_storage = cuda_storage(&a_log_storage, "a_log")?;
        let a_storage = cuda_storage(&a_storage, "a")?;
        let dt_bias_storage = cuda_storage(&dt_bias_storage, "dt_bias")?;
        let q_storage = cuda_storage(&q_storage, "q")?;
        let k_storage = cuda_storage(&k_storage, "k")?;
        let v_storage = cuda_storage(&v_storage, "v")?;
        let beta_storage = cuda_storage(&beta_storage, "beta")?;
        let state_indices_storage = cuda_storage(&state_indices_storage, "state_indices")?;
        let output_indices_storage = cuda_storage(&output_indices_storage, "output_state_indices")?;
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
            ("output_state_indices", output_indices_storage),
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
        let (output_indices_ptr, _output_indices_use) =
            immutable_address(output_indices_storage, &stream)?;
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
        let output_state_indices = descriptor(
            output_indices_ptr,
            output_indices_storage,
            output_indices_layout,
            "output_state_indices",
        )?;
        let cu_seqlens = descriptor(
            cu_seqlens_ptr,
            cu_seqlens_storage,
            cu_seqlens_layout,
            "cu_seqlens",
        )?;
        let mut call = PretransposeDecodeCall {
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
            output_state_indices: &output_state_indices,
            cu_seqlens: &cu_seqlens,
        };
        // SAFETY: the driver stream is owned by the plan's Candle device and remains
        // alive for the duration of this launch and every recorded graph node.
        let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
        plan.core.launch(&mut call, cuda_stream).map_err(core_error)
    }
}
