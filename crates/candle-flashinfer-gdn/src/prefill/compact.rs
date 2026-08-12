use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{
    CudaStream, PrefillCompactCall, PrefillSm90Plan, PrefillSm120Plan, PrefillSpecialization,
};

use super::state_adapter::StateAdapter;
use super::{PrefillInputs, PrefillShape, validate_checkpoint_inputs};
use crate::{
    core_error, message,
    raw_tensor::{
        RawTensor, base_address, cuda_storage, descriptor, ensure_ordinal, immutable_address,
        mutable_address, storage_dtype,
    },
    state_pool::validate_indexed_state_pool,
};

#[derive(Debug, Clone)]
enum CorePlan {
    Sm90(PrefillSm90Plan),
    Sm120(PrefillSm120Plan),
}

impl CorePlan {
    fn specialization(&self) -> &PrefillSpecialization {
        match self {
            Self::Sm90(plan) => plan.specialization(),
            Self::Sm120(plan) => plan.specialization(),
        }
    }

    fn device_id(&self) -> i32 {
        match self {
            Self::Sm90(plan) => plan.device_id(),
            Self::Sm120(plan) => plan.device_id(),
        }
    }

    fn launch(
        &self,
        call: &mut PrefillCompactCall<'_>,
        stream: CudaStream,
    ) -> flashinfer_gdn::Result<()> {
        match self {
            Self::Sm90(plan) => plan.launch(call, stream),
            Self::Sm120(plan) => plan.launch(call, stream),
        }
    }
}

/// Candle adapter shared by the compact-F32-state SM90 and SM120 backends.
#[derive(Debug)]
pub(super) struct Plan {
    core: CorePlan,
    device: CudaDevice,
    shape: PrefillShape,
    adapter: StateAdapter,
    tensormaps: Tensor,
    output: Tensor,
}

impl Plan {
    pub(super) fn new_sm90(
        core: PrefillSm90Plan,
        device: &CudaDevice,
        shape: PrefillShape,
    ) -> Result<Self> {
        Self::new(CorePlan::Sm90(core), device, shape)
    }

    pub(super) fn new_sm120(
        core: PrefillSm120Plan,
        device: &CudaDevice,
        shape: PrefillShape,
    ) -> Result<Self> {
        Self::new(CorePlan::Sm120(core), device, shape)
    }

    fn new(core: CorePlan, device: &CudaDevice, shape: PrefillShape) -> Result<Self> {
        let spec = core.specialization();
        let adapter = StateAdapter::new(
            device,
            shape.batch,
            spec.hv,
            spec.v,
            spec.k,
            shape.checkpoint_count,
        )?;
        let workspace_size = spec
            .num_sms
            .checked_mul(128)
            .ok_or_else(|| message("compact prefill workspace size overflows usize"))?;
        let tensormaps = Tensor::zeros(workspace_size, DType::U8, &Device::Cuda(device.clone()))?;
        let output = Tensor::zeros(
            (shape.total_tokens, spec.hv, spec.v),
            DType::BF16,
            &Device::Cuda(device.clone()),
        )?;
        Ok(Self {
            core,
            device: device.clone(),
            shape,
            adapter,
            tensormaps,
            output,
        })
    }

    pub(super) const fn shape(&self) -> PrefillShape {
        self.shape
    }

    pub(super) const fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub(super) fn forward(&self, inputs: &PrefillInputs<'_>) -> Result<Tensor> {
        let spec = self.core.specialization();
        validate_indexed_state_pool(
            inputs.state,
            inputs.state_indices,
            DType::BF16,
            &[spec.hv, spec.v, spec.k],
            self.shape.batch,
        )?;
        validate_checkpoint_inputs(inputs, spec, self.shape.batch, self.shape.checkpoint_count)?;
        if inputs.cu_seqlens.dtype() != DType::I32
            || inputs.cu_seqlens.dims() != [self.shape.batch + 1]
            || !inputs.cu_seqlens.is_contiguous()
        {
            return Err(message(format!(
                "cu_seqlens must be contiguous int32 [{}], found {:?} {:?}",
                self.shape.batch + 1,
                inputs.cu_seqlens.dtype(),
                inputs.cu_seqlens.dims()
            )));
        }
        self.validate_aliases(inputs, &self.output)?;
        inputs.state.inplace_op1(&PoolLaunch {
            plan: self,
            inputs,
            output: &self.output,
        })?;
        Ok(self.output.clone())
    }

    fn validate_aliases(&self, inputs: &PrefillInputs<'_>, output: &Tensor) -> Result<()> {
        let stream = self.device.cuda_stream();
        let mut mutable = vec![
            ("state", inputs.state),
            ("output", output),
            ("compact_state", self.adapter.compact()),
            ("cu_i64", self.adapter.cu_i64()),
            ("checkpoint_cu_i64", self.adapter.checkpoint_cu_i64()),
            ("tensormaps", &self.tensormaps),
        ];
        if let Some(checkpoint_compact) = self.adapter.checkpoint_compact() {
            mutable.push(("checkpoint_compact", checkpoint_compact));
        }
        let mut addresses = Vec::with_capacity(mutable.len());
        for (name, tensor) in mutable {
            let address = base_address(tensor, &stream, name)?;
            if addresses.contains(&address) {
                return Err(message(format!(
                    "{name} shares CUDA storage with another mutable prefill tensor"
                )));
            }
            addresses.push(address);
        }
        for (name, tensor) in [
            ("state_indices", inputs.state_indices),
            ("q", inputs.q),
            ("k", inputs.k),
            ("v", inputs.v),
            ("alpha", inputs.alpha),
            ("beta", inputs.beta),
            ("cu_seqlens", inputs.cu_seqlens),
        ] {
            if addresses.contains(&base_address(tensor, &stream, name)?) {
                return Err(message(format!(
                    "{name} must not share CUDA storage with mutable prefill tensors"
                )));
            }
        }
        for (name, tensor) in [
            ("state_checkpoints", inputs.state_checkpoints),
            ("checkpoint_cu_starts", inputs.checkpoint_cu_starts),
        ] {
            if let Some(tensor) = tensor
                && addresses.contains(&base_address(tensor, &stream, name)?)
            {
                return Err(message(format!(
                    "{name} must not share CUDA storage with mutable prefill tensors"
                )));
            }
        }
        if let (Some(checkpoints), Some(checkpoint_cu)) =
            (inputs.state_checkpoints, inputs.checkpoint_cu_starts)
            && base_address(checkpoints, &stream, "state_checkpoints")?
                == base_address(checkpoint_cu, &stream, "checkpoint_cu_starts")?
        {
            return Err(message(
                "state_checkpoints and checkpoint_cu_starts must use distinct CUDA storage",
            ));
        }
        Ok(())
    }
}

struct PoolLaunch<'a> {
    plan: &'a Plan,
    inputs: &'a PrefillInputs<'a>,
    output: &'a Tensor,
}

impl InplaceOp1 for PoolLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-pool"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        ensure_ordinal(storage, self.plan.core.device_id(), "state")?;
        let dtype = storage_dtype(storage)?;
        let stream = self.plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        self.plan.adapter.compact().inplace_op1(&CompactLaunch {
            parent: self,
            pool: RawTensor::new(address, dtype, layout, self.plan.core.device_id(), "state")?,
        })
    }
}

struct CompactLaunch<'a, 'b> {
    parent: &'a PoolLaunch<'b>,
    pool: RawTensor,
}

impl InplaceOp1 for CompactLaunch<'_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-state"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "compact_state")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        self.parent.output.inplace_op1(&OutputLaunch {
            parent: self,
            compact: RawTensor::new(
                address,
                dtype,
                layout,
                plan.core.device_id(),
                "compact_state",
            )?,
        })
    }
}

struct OutputLaunch<'a, 'b, 'c> {
    parent: &'a CompactLaunch<'b, 'c>,
    compact: RawTensor,
}

impl InplaceOp1 for OutputLaunch<'_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-output"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "output")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        plan.tensormaps.inplace_op1(&WorkspaceLaunch {
            parent: self,
            output: RawTensor::new(address, dtype, layout, plan.core.device_id(), "output")?,
        })
    }
}

struct WorkspaceLaunch<'a, 'b, 'c, 'd> {
    parent: &'a OutputLaunch<'b, 'c, 'd>,
    output: RawTensor,
}

impl InplaceOp1 for WorkspaceLaunch<'_, '_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-tensormaps"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "tensormaps")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        plan.adapter.cu_i64().inplace_op1(&CuLaunch {
            parent: self,
            workspace: RawTensor::new(address, dtype, layout, plan.core.device_id(), "tensormaps")?,
        })
    }
}

struct CuLaunch<'a, 'b, 'c, 'd, 'e> {
    parent: &'a WorkspaceLaunch<'b, 'c, 'd, 'e>,
    workspace: RawTensor,
}

impl InplaceOp1 for CuLaunch<'_, '_, '_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-cu-i64"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, cu64_storage: &mut CudaStorage, cu64_layout: &Layout) -> Result<()> {
        let output_launch = self.parent.parent;
        let compact_launch = output_launch.parent;
        let pool_launch = compact_launch.parent;
        let plan = pool_launch.plan;
        ensure_ordinal(cu64_storage, plan.core.device_id(), "cu_i64")?;
        let cu64_dtype = storage_dtype(cu64_storage)?;
        let stream = plan.device.cuda_stream();
        let (cu64_ptr, _cu64_guard) = mutable_address(cu64_storage, &stream)?;

        let cu64 = RawTensor::new(
            cu64_ptr,
            cu64_dtype,
            cu64_layout,
            plan.core.device_id(),
            "cu_i64",
        )?;
        if let Some(checkpoints) = pool_launch.inputs.state_checkpoints {
            checkpoints.inplace_op1(&CheckpointLaunch { parent: self, cu64 })
        } else {
            launch_compact(self, cu64, None, None, None)
        }
    }
}

struct CheckpointLaunch<'a, 'b, 'c, 'd, 'e, 'f> {
    parent: &'a CuLaunch<'b, 'c, 'd, 'e, 'f>,
    cu64: RawTensor,
}

impl InplaceOp1 for CheckpointLaunch<'_, '_, '_, '_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-checkpoints"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.parent.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "state_checkpoints")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        let checkpoint_compact = plan
            .adapter
            .checkpoint_compact()
            .ok_or_else(|| message("checkpoint-enabled plan is missing float32 scratch"))?;
        checkpoint_compact.inplace_op1(&KernelCheckpointLaunch {
            parent: self,
            output_checkpoints: RawTensor::new(
                address,
                dtype,
                layout,
                plan.core.device_id(),
                "state_checkpoints",
            )?,
        })
    }
}

struct KernelCheckpointLaunch<'a, 'b, 'c, 'd, 'e, 'f, 'g> {
    parent: &'a CheckpointLaunch<'b, 'c, 'd, 'e, 'f, 'g>,
    output_checkpoints: RawTensor,
}

impl InplaceOp1 for KernelCheckpointLaunch<'_, '_, '_, '_, '_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-checkpoint-f32"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.parent.parent.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "checkpoint_compact")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        plan.adapter
            .checkpoint_cu_i64()
            .inplace_op1(&CheckpointCuLaunch {
                parent: self,
                checkpoints: RawTensor::new(
                    address,
                    dtype,
                    layout,
                    plan.core.device_id(),
                    "checkpoint_compact",
                )?,
            })
    }
}

struct CheckpointCuLaunch<'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h> {
    parent: &'a KernelCheckpointLaunch<'b, 'c, 'd, 'e, 'f, 'g, 'h>,
    checkpoints: RawTensor,
}

impl InplaceOp1 for CheckpointCuLaunch<'_, '_, '_, '_, '_, '_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-compact-checkpoint-cu-i64"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.parent.parent.parent.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "checkpoint_cu_i64")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        let checkpoint_cu64 = RawTensor::new(
            address,
            dtype,
            layout,
            plan.core.device_id(),
            "checkpoint_cu_i64",
        )?;
        launch_compact(
            self.parent.parent.parent,
            self.parent.parent.cu64.clone(),
            Some(self.checkpoints.clone()),
            Some(self.parent.output_checkpoints.clone()),
            Some(checkpoint_cu64),
        )
    }
}

#[allow(clippy::too_many_lines)]
fn launch_compact(
    launch: &CuLaunch<'_, '_, '_, '_, '_>,
    cu64: RawTensor,
    checkpoints: Option<RawTensor>,
    output_checkpoints: Option<RawTensor>,
    checkpoint_cu64: Option<RawTensor>,
) -> Result<()> {
    let output_launch = launch.parent.parent;
    let compact_launch = output_launch.parent;
    let pool_launch = compact_launch.parent;
    let plan = pool_launch.plan;
    let stream = plan.device.cuda_stream();

    let mut tensors = vec![
        ("q", pool_launch.inputs.q),
        ("k", pool_launch.inputs.k),
        ("v", pool_launch.inputs.v),
        ("alpha", pool_launch.inputs.alpha),
        ("beta", pool_launch.inputs.beta),
        ("cu_seqlens", pool_launch.inputs.cu_seqlens),
        ("state_indices", pool_launch.inputs.state_indices),
    ];
    if let Some(checkpoint_cu_starts) = pool_launch.inputs.checkpoint_cu_starts {
        tensors.push(("checkpoint_cu_starts", checkpoint_cu_starts));
    }
    let mut storages = Vec::with_capacity(tensors.len());
    let mut layouts = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let (storage, layout) = tensor.storage_and_layout();
        ensure_ordinal(cuda_storage(&storage, name)?, plan.core.device_id(), name)?;
        storages.push((name, storage));
        layouts.push(layout);
    }
    let mut addresses = Vec::with_capacity(storages.len());
    let mut guards = Vec::with_capacity(storages.len());
    for (name, storage) in &storages {
        let (address, guard) = immutable_address(cuda_storage(storage, name)?, &stream)?;
        addresses.push(address);
        guards.push(guard);
    }
    let mut descriptors = Vec::with_capacity(storages.len());
    for index in 0..storages.len() {
        descriptors.push(descriptor(
            addresses[index],
            cuda_storage(&storages[index].1, storages[index].0)?,
            layouts[index],
            storages[index].0,
        )?);
    }

    let pool_size = i32::try_from(pool_launch.inputs.state.dims()[0])
        .map_err(|_| message("state pool size does not fit i32"))?;
    // SAFETY: nested Candle guards hold all fixed-shape allocations live.
    unsafe {
        plan.adapter.gather(
            &stream,
            compact_launch.pool.address(),
            addresses[6],
            output_launch.compact.address(),
            pool_size,
        )?;
        plan.adapter
            .convert_cu_seqlens(&stream, addresses[5], cu64.address())?;
        if let Some(checkpoint_cu64) = checkpoint_cu64.as_ref() {
            plan.adapter
                .convert_cu_seqlens(&stream, addresses[7], checkpoint_cu64.address())?;
        }
    }

    let mut state = output_launch.compact.descriptor()?;
    let mut output = launch.parent.output.descriptor()?;
    let mut workspace = launch.workspace.descriptor()?;
    let cu64 = cu64.descriptor()?;
    let mut checkpoint_descriptor = checkpoints
        .as_ref()
        .map(RawTensor::descriptor)
        .transpose()?;
    let checkpoint_cu_descriptor = checkpoint_cu64
        .as_ref()
        .map(RawTensor::descriptor)
        .transpose()?;
    let mut call = PrefillCompactCall {
        q: &descriptors[0],
        k: &descriptors[1],
        v: &descriptors[2],
        alpha: &descriptors[3],
        beta: &descriptors[4],
        state: &mut state,
        output: &mut output,
        cu_seqlens: &cu64,
        state_checkpoints: checkpoint_descriptor.as_mut(),
        checkpoint_cu_starts: checkpoint_cu_descriptor.as_ref(),
        tensormaps: &mut workspace,
    };
    let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
    plan.core
        .launch(&mut call, cuda_stream)
        .map_err(core_error)?;
    if let (Some(checkpoints), Some(output_checkpoints)) =
        (checkpoints.as_ref(), output_checkpoints.as_ref())
    {
        // SAFETY: both plan-sized checkpoint allocations are held mutable by
        // the nested Candle guards and use this same stream.
        unsafe {
            plan.adapter.cast_checkpoints(
                &stream,
                checkpoints.address(),
                output_checkpoints.address(),
            )?;
        }
    }
    // SAFETY: the same nested guards remain live and the kernel/scatter are
    // ordered on the same Candle stream.
    unsafe {
        plan.adapter.scatter(
            &stream,
            output_launch.compact.address(),
            addresses[6],
            compact_launch.pool.address(),
            pool_size,
        )?;
    }
    let _guards = guards;
    Ok(())
}
