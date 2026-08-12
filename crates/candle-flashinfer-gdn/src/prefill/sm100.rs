use candle::cuda_backend::{CudaDevice, CudaStorage};
use candle::{CpuStorage, DType, Device, InplaceOp1, Layout, Result, Tensor};
use flashinfer_gdn::{CudaStream, PrefillSm100Call, PrefillSm100Plan as CorePlan};

use super::{PrefillInputs, PrefillShape, validate_checkpoint_inputs};
use crate::{
    core_error, message,
    raw_tensor::{
        RawTensor, base_address, cuda_storage, descriptor, ensure_ordinal, immutable_address,
        mutable_address, storage_dtype,
    },
    state_pool::validate_indexed_state_pool,
};

/// Candle adapter for the SM100/SM103 native indexed-state backend.
#[derive(Debug)]
pub(super) struct Plan {
    core: CorePlan,
    device: CudaDevice,
    shape: PrefillShape,
    tensormaps: Tensor,
    output: Tensor,
}

impl Plan {
    pub(super) fn new(core: CorePlan, device: &CudaDevice, shape: PrefillShape) -> Result<Self> {
        let spec = core.specialization();
        let workspace_size = spec
            .num_sms
            .checked_mul(4 * 128)
            .ok_or_else(|| message("SM100 prefill workspace size overflows usize"))?;
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
        self.validate_aliases(inputs, &self.output)?;
        inputs.state.inplace_op1(&StateLaunch {
            plan: self,
            inputs,
            output: &self.output,
        })?;
        Ok(self.output.clone())
    }

    fn validate_aliases(&self, inputs: &PrefillInputs<'_>, output: &Tensor) -> Result<()> {
        let stream = self.device.cuda_stream();
        let state = base_address(inputs.state, &stream, "state")?;
        let output = base_address(output, &stream, "output")?;
        let workspace = base_address(&self.tensormaps, &stream, "tensormaps")?;
        if state == output || state == workspace || output == workspace {
            return Err(message(
                "state, output, and prefill TMA workspace must use distinct CUDA storage",
            ));
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
            let address = base_address(tensor, &stream, name)?;
            if [state, output, workspace].contains(&address) {
                return Err(message(format!(
                    "{name} must not share CUDA storage with mutable prefill tensors"
                )));
            }
        }
        for (name, tensor) in [
            ("state_checkpoints", inputs.state_checkpoints),
            ("checkpoint_cu_starts", inputs.checkpoint_cu_starts),
        ] {
            if let Some(tensor) = tensor {
                let address = base_address(tensor, &stream, name)?;
                if [state, output, workspace].contains(&address) {
                    return Err(message(format!(
                        "{name} must not share CUDA storage with mutable prefill tensors"
                    )));
                }
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

struct StateLaunch<'a> {
    plan: &'a Plan,
    inputs: &'a PrefillInputs<'a>,
    output: &'a Tensor,
}

impl InplaceOp1 for StateLaunch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-sm100-state"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        ensure_ordinal(storage, self.plan.core.device_id(), "state")?;
        let dtype = storage_dtype(storage)?;
        let stream = self.plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        self.output.inplace_op1(&OutputLaunch {
            parent: self,
            state: RawTensor::new(address, dtype, layout, self.plan.core.device_id(), "state")?,
        })
    }
}

struct OutputLaunch<'a, 'b> {
    parent: &'a StateLaunch<'b>,
    state: RawTensor,
}

impl InplaceOp1 for OutputLaunch<'_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-sm100-output"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "output")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        let output = RawTensor::new(address, dtype, layout, plan.core.device_id(), "output")?;
        if let Some(checkpoints) = self.parent.inputs.state_checkpoints {
            checkpoints.inplace_op1(&CheckpointLaunch {
                parent: self,
                output,
            })
        } else {
            plan.tensormaps.inplace_op1(&WorkspaceLaunch {
                parent: self,
                output,
                checkpoints: None,
            })
        }
    }
}

struct CheckpointLaunch<'a, 'b, 'c> {
    parent: &'a OutputLaunch<'b, 'c>,
    output: RawTensor,
}

impl InplaceOp1 for CheckpointLaunch<'_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-sm100-checkpoints"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let plan = self.parent.parent.plan;
        ensure_ordinal(storage, plan.core.device_id(), "state_checkpoints")?;
        let dtype = storage_dtype(storage)?;
        let stream = plan.device.cuda_stream();
        let (address, _guard) = mutable_address(storage, &stream)?;
        plan.tensormaps.inplace_op1(&WorkspaceLaunch {
            parent: self.parent,
            output: self.output.clone(),
            checkpoints: Some(RawTensor::new(
                address,
                dtype,
                layout,
                plan.core.device_id(),
                "state_checkpoints",
            )?),
        })
    }
}

struct WorkspaceLaunch<'a, 'b, 'c> {
    parent: &'a OutputLaunch<'b, 'c>,
    output: RawTensor,
    checkpoints: Option<RawTensor>,
}

impl InplaceOp1 for WorkspaceLaunch<'_, '_, '_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-prefill-sm100-tensormaps"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        Err(message("FlashInfer GDN prefill requires CUDA storage"))
    }

    #[allow(clippy::too_many_lines)]
    fn cuda_fwd(
        &self,
        workspace_storage: &mut CudaStorage,
        workspace_layout: &Layout,
    ) -> Result<()> {
        let launch = self.parent.parent;
        let plan = launch.plan;
        ensure_ordinal(workspace_storage, plan.core.device_id(), "tensormaps")?;
        let workspace_dtype = storage_dtype(workspace_storage)?;
        let stream = plan.device.cuda_stream();
        let (workspace_ptr, _workspace_guard) = mutable_address(workspace_storage, &stream)?;

        let mut tensors = vec![
            ("q", launch.inputs.q),
            ("k", launch.inputs.k),
            ("v", launch.inputs.v),
            ("alpha", launch.inputs.alpha),
            ("beta", launch.inputs.beta),
            ("cu_seqlens", launch.inputs.cu_seqlens),
            ("state_indices", launch.inputs.state_indices),
        ];
        if let Some(checkpoint_cu_starts) = launch.inputs.checkpoint_cu_starts {
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

        let mut state = self.parent.state.descriptor()?;
        let mut output = self.output.descriptor()?;
        let mut checkpoint_descriptor = self
            .checkpoints
            .as_ref()
            .map(RawTensor::descriptor)
            .transpose()?;
        let mut workspace = RawTensor::new(
            workspace_ptr,
            workspace_dtype,
            workspace_layout,
            plan.core.device_id(),
            "tensormaps",
        )?
        .descriptor()?;
        let mut call = PrefillSm100Call {
            q: &descriptors[0],
            k: &descriptors[1],
            v: &descriptors[2],
            alpha: &descriptors[3],
            beta: &descriptors[4],
            state: &mut state,
            output: &mut output,
            cu_seqlens: &descriptors[5],
            state_indices: &descriptors[6],
            state_checkpoints: checkpoint_descriptor.as_mut(),
            checkpoint_cu_starts: descriptors.get(7),
            tensormaps: &mut workspace,
        };
        // Keep immutable cudarc guards live across the FFI call.
        let _guards = guards;
        // SAFETY: this stream belongs to the plan's Candle device.
        let cuda_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };
        plan.core.launch(&mut call, cuda_stream).map_err(core_error)
    }
}
