use std::sync::Arc;

use flashinfer_gdn_sys::{
    GdnHandle, PrefillBackend, PrefillSm90Kernel, PrefillSm90Tensors, PrefillSm100Kernel,
    PrefillSm100Tensors, PrefillSm120Kernel, PrefillSpecialization,
};

use crate::tensor::DlTensorOwner;
use crate::validation::{bounds_overlap, check_rank, expect};
use crate::{CudaStream, CudaTensor, DType, Error, Result};

/// Arguments shared by SM90 and SM120/121 prefill.
///
/// `state` is compact sequence order because these upstream kernels do not
/// accept pool indices. Framework adapters may gather/cast a public state pool
/// into it and scatter/cast it back after launch.
pub struct PrefillCompactCall<'a> {
    /// Query `[N,H,K]`, BF16.
    pub q: &'a CudaTensor,
    /// Key `[N,H,K]`, BF16.
    pub k: &'a CudaTensor,
    /// Value `[N,HV,V]`, BF16.
    pub v: &'a CudaTensor,
    /// Multiplicative forget gate `[N,HV]`, float32.
    pub alpha: &'a CudaTensor,
    /// Update gate `[N,HV]`, float32.
    pub beta: &'a CudaTensor,
    /// Compact V-major state `[B,HV,V,K]`, float32, updated in place.
    pub state: &'a mut CudaTensor,
    /// Output `[N,HV,V]`, BF16.
    pub output: &'a mut CudaTensor,
    /// Cumulative lengths `[B+1]`, int64.
    pub cu_seqlens: &'a CudaTensor,
    /// Compact float32 checkpoint output `[C,HV,V,K]` when enabled.
    pub state_checkpoints: Option<&'a mut CudaTensor>,
    /// Per-sequence checkpoint row offsets `[B+1]`, int64 when enabled.
    pub checkpoint_cu_starts: Option<&'a CudaTensor>,
    /// TMA descriptor workspace `[num_sms*128]`, uint8.
    pub tensormaps: &'a mut CudaTensor,
}

/// Arguments for SM100/SM103 prefill with native pool indexing.
pub struct PrefillSm100Call<'a> {
    /// Query `[N,H,K]`, BF16.
    pub q: &'a CudaTensor,
    /// Key `[N,H,K]`, BF16.
    pub k: &'a CudaTensor,
    /// Value `[N,HV,V]`, BF16.
    pub v: &'a CudaTensor,
    /// Multiplicative forget gate `[N,HV]`, float32.
    pub alpha: &'a CudaTensor,
    /// Update gate `[N,HV]`, float32.
    pub beta: &'a CudaTensor,
    /// V-major BF16 state pool `[P,HV,V,K]`, updated in place.
    pub state: &'a mut CudaTensor,
    /// Output `[N,HV,V]`, BF16.
    pub output: &'a mut CudaTensor,
    /// Cumulative lengths `[B+1]`, int32.
    pub cu_seqlens: &'a CudaTensor,
    /// Pool slots `[B]`, int32. Entries must be valid and unique.
    pub state_indices: &'a CudaTensor,
    /// Compact BF16 checkpoint output `[C,HV,V,K]` when enabled.
    pub state_checkpoints: Option<&'a mut CudaTensor>,
    /// Per-sequence checkpoint row offsets `[B+1]`, int32 when enabled.
    pub checkpoint_cu_starts: Option<&'a CudaTensor>,
    /// TMA workspace `[num_sms*4*128]`, uint8.
    pub tensormaps: &'a mut CudaTensor,
}

#[derive(Debug)]
struct CompactInner<K> {
    kernel: K,
    device_id: i32,
}

macro_rules! compact_plan {
    ($plan:ident, $kernel:ty, $backend:expr, $load:ident) => {
        /// Prepared architecture-specific compact-state prefill plan.
        #[derive(Debug, Clone)]
        pub struct $plan {
            inner: Arc<CompactInner<$kernel>>,
        }

        impl $plan {
            /// Compiles or loads the selected specialization.
            pub fn prepare(
                handle: &GdnHandle,
                specialization: &PrefillSpecialization,
                device_id: i32,
            ) -> Result<Self> {
                check_plan_identity(specialization, $backend, device_id)?;
                let kernel = handle.$load(specialization)?;
                Ok(Self {
                    inner: Arc::new(CompactInner { kernel, device_id }),
                })
            }

            /// Compile-time specialization owned by this plan.
            #[must_use]
            pub fn specialization(&self) -> &PrefillSpecialization {
                self.inner.kernel.specialization()
            }

            /// Bound CUDA device ordinal.
            #[must_use]
            pub fn device_id(&self) -> i32 {
                self.inner.device_id
            }

            /// Launches without compilation, allocation, or file I/O.
            pub fn launch(
                &self,
                call: &mut PrefillCompactCall<'_>,
                stream: CudaStream,
            ) -> Result<()> {
                validate_prefill_compact_for_device(
                    self.specialization(),
                    self.inner.device_id,
                    call,
                )?;
                let mut q = DlTensorOwner::from_tensor(call.q)?;
                let mut k = DlTensorOwner::from_tensor(call.k)?;
                let mut v = DlTensorOwner::from_tensor(call.v)?;
                let mut alpha = DlTensorOwner::from_tensor(call.alpha)?;
                let mut beta = DlTensorOwner::from_tensor(call.beta)?;
                let mut initial_state = DlTensorOwner::from_tensor(&*call.state)?;
                let mut output_state = DlTensorOwner::from_tensor(&*call.state)?;
                let mut output = DlTensorOwner::from_tensor(&*call.output)?;
                let mut cu_seqlens = DlTensorOwner::from_tensor(call.cu_seqlens)?;
                let mut state_checkpoints = call
                    .state_checkpoints
                    .as_deref()
                    .map(DlTensorOwner::from_tensor)
                    .transpose()?;
                let mut checkpoint_cu_starts = call
                    .checkpoint_cu_starts
                    .map(DlTensorOwner::from_tensor)
                    .transpose()?;
                let mut tensormaps = DlTensorOwner::from_tensor(&*call.tensormaps)?;
                let mut tensors = PrefillSm90Tensors {
                    q: &mut q.tensor,
                    k: &mut k.tensor,
                    v: &mut v.tensor,
                    alpha: &mut alpha.tensor,
                    beta: &mut beta.tensor,
                    initial_state: &mut initial_state.tensor,
                    output: &mut output.tensor,
                    output_state: &mut output_state.tensor,
                    cu_seqlens: &mut cu_seqlens.tensor,
                    state_checkpoints: state_checkpoints.as_mut().map(|owner| &mut owner.tensor),
                    checkpoint_cu_starts: checkpoint_cu_starts
                        .as_mut()
                        .map(|owner| &mut owner.tensor),
                    tensormaps: &mut tensormaps.tensor,
                };
                // SAFETY: all specialization, shape, dtype, device, layout, and
                // aliasing contracts were checked above.
                unsafe { self.inner.kernel.launch(&mut tensors, stream.as_raw())? };
                Ok(())
            }
        }
    };
}

compact_plan!(
    PrefillSm90Plan,
    PrefillSm90Kernel,
    PrefillBackend::Sm90,
    load_prefill_sm90
);
compact_plan!(
    PrefillSm120Plan,
    PrefillSm120Kernel,
    PrefillBackend::Sm120,
    load_prefill_sm120
);

#[derive(Debug)]
struct Sm100Inner {
    kernel: PrefillSm100Kernel,
    device_id: i32,
}

/// Prepared SM100/SM103 native indexed-BF16-state prefill plan.
#[derive(Debug, Clone)]
pub struct PrefillSm100Plan {
    inner: Arc<Sm100Inner>,
}

impl PrefillSm100Plan {
    /// Compiles or loads the selected specialization.
    pub fn prepare(
        handle: &GdnHandle,
        specialization: &PrefillSpecialization,
        device_id: i32,
    ) -> Result<Self> {
        check_plan_identity(specialization, PrefillBackend::Sm100, device_id)?;
        Ok(Self {
            inner: Arc::new(Sm100Inner {
                kernel: handle.load_prefill_sm100(specialization)?,
                device_id,
            }),
        })
    }

    /// Compile-time specialization owned by this plan.
    #[must_use]
    pub fn specialization(&self) -> &PrefillSpecialization {
        self.inner.kernel.specialization()
    }

    /// Bound CUDA device ordinal.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }

    /// Launches without compilation, allocation, or file I/O.
    pub fn launch(&self, call: &mut PrefillSm100Call<'_>, stream: CudaStream) -> Result<()> {
        validate_prefill_sm100_for_device(self.specialization(), self.inner.device_id, call)?;
        let mut q = DlTensorOwner::from_tensor(call.q)?;
        let mut k = DlTensorOwner::from_tensor(call.k)?;
        let mut v = DlTensorOwner::from_tensor(call.v)?;
        let mut alpha = DlTensorOwner::from_tensor(call.alpha)?;
        let mut beta = DlTensorOwner::from_tensor(call.beta)?;
        let mut initial_state = DlTensorOwner::from_tensor(&*call.state)?;
        let mut output_state = DlTensorOwner::from_tensor(&*call.state)?;
        let mut output = DlTensorOwner::from_tensor(&*call.output)?;
        let mut cu_seqlens = DlTensorOwner::from_tensor(call.cu_seqlens)?;
        let mut state_indices = DlTensorOwner::from_tensor(call.state_indices)?;
        let mut state_checkpoints = call
            .state_checkpoints
            .as_deref()
            .map(DlTensorOwner::from_tensor)
            .transpose()?;
        let mut checkpoint_cu_starts = call
            .checkpoint_cu_starts
            .map(DlTensorOwner::from_tensor)
            .transpose()?;
        let mut tensormaps = DlTensorOwner::from_tensor(&*call.tensormaps)?;
        let mut tensors = PrefillSm100Tensors {
            q: &mut q.tensor,
            k: &mut k.tensor,
            v: &mut v.tensor,
            alpha: &mut alpha.tensor,
            beta: &mut beta.tensor,
            initial_state: &mut initial_state.tensor,
            output: &mut output.tensor,
            output_state: &mut output_state.tensor,
            cu_seqlens: &mut cu_seqlens.tensor,
            state_indices: &mut state_indices.tensor,
            state_checkpoints: state_checkpoints.as_mut().map(|owner| &mut owner.tensor),
            checkpoint_cu_starts: checkpoint_cu_starts.as_mut().map(|owner| &mut owner.tensor),
            tensormaps: &mut tensormaps.tensor,
        };
        // SAFETY: all contracts were checked above.
        unsafe { self.inner.kernel.launch(&mut tensors, stream.as_raw())? };
        Ok(())
    }
}

fn check_plan_identity(
    specialization: &PrefillSpecialization,
    backend: PrefillBackend,
    device_id: i32,
) -> Result<()> {
    specialization.validate()?;
    if specialization.backend != backend {
        return Err(Error::tensor(
            "plan",
            format!("expected {backend:?}, found {:?}", specialization.backend),
        ));
    }
    if device_id < 0 {
        return Err(Error::tensor(
            "plan",
            format!("negative CUDA device id {device_id}"),
        ));
    }
    Ok(())
}

/// Validates an SM90/SM120 compact-state launch.
pub fn validate_prefill_compact(
    specialization: &PrefillSpecialization,
    call: &PrefillCompactCall<'_>,
) -> Result<()> {
    validate_prefill_compact_for_device(specialization, call.q.device_id(), call)
}

fn validate_prefill_compact_for_device(
    specialization: &PrefillSpecialization,
    device_id: i32,
    call: &PrefillCompactCall<'_>,
) -> Result<()> {
    specialization.validate()?;
    if !matches!(
        specialization.backend,
        PrefillBackend::Sm90 | PrefillBackend::Sm120
    ) {
        return Err(Error::tensor(
            "plan",
            "compact prefill requires SM90 or SM120",
        ));
    }
    check_rank(call.q, "q", 3)?;
    check_rank(call.state, "state", 4)?;
    let n = call.q.shape()[0];
    let batch = call.state.shape()[0];
    let h = specialization.h as i64;
    let hv = specialization.hv as i64;
    let k = specialization.k as i64;
    let v_dim = specialization.v as i64;
    expect(call.q, "q", DType::BF16, &[n, h, k], device_id)?;
    expect(call.k, "k", DType::BF16, &[n, h, k], device_id)?;
    expect(call.v, "v", DType::BF16, &[n, hv, v_dim], device_id)?;
    expect(call.alpha, "alpha", DType::F32, &[n, hv], device_id)?;
    expect(call.beta, "beta", DType::F32, &[n, hv], device_id)?;
    expect(
        call.state,
        "state",
        DType::F32,
        &[batch, hv, v_dim, k],
        device_id,
    )?;
    expect(
        call.output,
        "output",
        DType::BF16,
        &[n, hv, v_dim],
        device_id,
    )?;
    expect(
        call.cu_seqlens,
        "cu_seqlens",
        DType::I64,
        &[batch + 1],
        device_id,
    )?;
    expect(
        call.tensormaps,
        "tensormaps",
        DType::U8,
        &[(specialization.num_sms * 128) as i64],
        device_id,
    )?;
    for (name, tensor, alignment) in [
        ("q", call.q, 16_usize),
        ("k", call.k, 16),
        ("v", call.v, 16),
        ("alpha", call.alpha, 16),
        ("beta", call.beta, 16),
        ("cu_seqlens", call.cu_seqlens, 8),
    ] {
        require_alignment(name, tensor, alignment)?;
    }
    let immutable = [
        call.q,
        call.k,
        call.v,
        call.alpha,
        call.beta,
        call.cu_seqlens,
    ];
    validate_common_layouts_and_aliases(immutable, call.state, call.output, call.tensormaps)?;
    validate_checkpoint_contract(
        specialization,
        call.state_checkpoints.as_deref(),
        call.checkpoint_cu_starts,
        DType::F32,
        DType::I64,
        batch,
        hv,
        v_dim,
        k,
        device_id,
        immutable,
        call.state,
        call.output,
        call.tensormaps,
    )
}

/// Validates an SM100/SM103 native indexed-state launch.
pub fn validate_prefill_sm100(
    specialization: &PrefillSpecialization,
    call: &PrefillSm100Call<'_>,
) -> Result<()> {
    validate_prefill_sm100_for_device(specialization, call.q.device_id(), call)
}

fn validate_prefill_sm100_for_device(
    specialization: &PrefillSpecialization,
    device_id: i32,
    call: &PrefillSm100Call<'_>,
) -> Result<()> {
    specialization.validate()?;
    if specialization.backend != PrefillBackend::Sm100 {
        return Err(Error::tensor("plan", "indexed BF16 prefill requires SM100"));
    }
    check_rank(call.q, "q", 3)?;
    check_rank(call.state, "state", 4)?;
    check_rank(call.state_indices, "state_indices", 1)?;
    let n = call.q.shape()[0];
    let pool = call.state.shape()[0];
    let batch = call.state_indices.shape()[0];
    let h = specialization.h as i64;
    let hv = specialization.hv as i64;
    let k = specialization.k as i64;
    let v_dim = specialization.v as i64;
    expect(call.q, "q", DType::BF16, &[n, h, k], device_id)?;
    expect(call.k, "k", DType::BF16, &[n, h, k], device_id)?;
    expect(call.v, "v", DType::BF16, &[n, hv, v_dim], device_id)?;
    expect(call.alpha, "alpha", DType::F32, &[n, hv], device_id)?;
    expect(call.beta, "beta", DType::F32, &[n, hv], device_id)?;
    expect(
        call.state,
        "state",
        DType::BF16,
        &[pool, hv, v_dim, k],
        device_id,
    )?;
    expect(
        call.output,
        "output",
        DType::BF16,
        &[n, hv, v_dim],
        device_id,
    )?;
    expect(
        call.cu_seqlens,
        "cu_seqlens",
        DType::I32,
        &[batch + 1],
        device_id,
    )?;
    expect(
        call.state_indices,
        "state_indices",
        DType::I32,
        &[batch],
        device_id,
    )?;
    expect(
        call.tensormaps,
        "tensormaps",
        DType::U8,
        &[(specialization.num_sms * 4 * 128) as i64],
        device_id,
    )?;
    for (name, tensor, alignment) in [
        ("q", call.q, 16_usize),
        ("k", call.k, 16),
        ("v", call.v, 16),
        ("alpha", call.alpha, 16),
        ("beta", call.beta, 16),
        ("cu_seqlens", call.cu_seqlens, 4),
        ("state_indices", call.state_indices, 4),
    ] {
        require_alignment(name, tensor, alignment)?;
    }
    let immutable = [
        call.q,
        call.k,
        call.v,
        call.alpha,
        call.beta,
        call.cu_seqlens,
        call.state_indices,
    ];
    validate_common_layouts_and_aliases(immutable, call.state, call.output, call.tensormaps)?;
    validate_checkpoint_contract(
        specialization,
        call.state_checkpoints.as_deref(),
        call.checkpoint_cu_starts,
        DType::BF16,
        DType::I32,
        batch,
        hv,
        v_dim,
        k,
        device_id,
        immutable,
        call.state,
        call.output,
        call.tensormaps,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_checkpoint_contract<const N: usize>(
    specialization: &PrefillSpecialization,
    state_checkpoints: Option<&CudaTensor>,
    checkpoint_cu_starts: Option<&CudaTensor>,
    checkpoint_dtype: DType,
    cu_dtype: DType,
    batch: i64,
    hv: i64,
    v_dim: i64,
    k: i64,
    device_id: i32,
    immutable: [&CudaTensor; N],
    state: &CudaTensor,
    output: &CudaTensor,
    tensormaps: &CudaTensor,
) -> Result<()> {
    if !specialization.checkpoints_enabled() {
        if state_checkpoints.is_some() || checkpoint_cu_starts.is_some() {
            return Err(Error::tensor(
                "checkpoints",
                "checkpoint tensors require a checkpoint-enabled specialization",
            ));
        }
        return Ok(());
    }

    let state_checkpoints = state_checkpoints.ok_or_else(|| {
        Error::tensor(
            "state_checkpoints",
            "checkpoint-enabled prefill requires compact checkpoint output",
        )
    })?;
    let checkpoint_cu_starts = checkpoint_cu_starts.ok_or_else(|| {
        Error::tensor(
            "checkpoint_cu_starts",
            "checkpoint-enabled prefill requires checkpoint row offsets",
        )
    })?;
    check_rank(state_checkpoints, "state_checkpoints", 4)?;
    let checkpoint_count = state_checkpoints.shape()[0];
    expect(
        state_checkpoints,
        "state_checkpoints",
        checkpoint_dtype,
        &[checkpoint_count, hv, v_dim, k],
        device_id,
    )?;
    expect(
        checkpoint_cu_starts,
        "checkpoint_cu_starts",
        cu_dtype,
        &[batch + 1],
        device_id,
    )?;
    if !state_checkpoints.is_contiguous() || !checkpoint_cu_starts.is_contiguous() {
        return Err(Error::tensor(
            "checkpoints",
            "checkpoint output and row offsets must be compact row-major",
        ));
    }
    require_alignment("state_checkpoints", state_checkpoints, 16)?;
    require_alignment(
        "checkpoint_cu_starts",
        checkpoint_cu_starts,
        if cu_dtype == DType::I64 { 8 } else { 4 },
    )?;

    let checkpoint_bounds = state_checkpoints.byte_bounds()?;
    let checkpoint_cu_bounds = checkpoint_cu_starts.byte_bounds()?;
    for (name, tensor) in [
        ("state", state),
        ("output", output),
        ("tensormaps", tensormaps),
    ] {
        let bounds = tensor.byte_bounds()?;
        if bounds_overlap(checkpoint_bounds, bounds) || bounds_overlap(checkpoint_cu_bounds, bounds)
        {
            return Err(Error::tensor(
                name,
                "tensor must not overlap checkpoint storage",
            ));
        }
    }
    if bounds_overlap(checkpoint_bounds, checkpoint_cu_bounds) {
        return Err(Error::tensor(
            "checkpoints",
            "checkpoint output and row offsets must not overlap",
        ));
    }
    for tensor in immutable {
        let bounds = tensor.byte_bounds()?;
        if bounds_overlap(checkpoint_bounds, bounds) {
            return Err(Error::tensor(
                "input",
                "prefill inputs must not overlap checkpoint output",
            ));
        }
    }
    Ok(())
}

fn validate_common_layouts_and_aliases<const N: usize>(
    immutable: [&CudaTensor; N],
    state: &CudaTensor,
    output: &CudaTensor,
    tensormaps: &CudaTensor,
) -> Result<()> {
    for (name, tensor) in std::iter::once(("state", state))
        .chain(std::iter::once(("output", output)))
        .chain(std::iter::once(("tensormaps", tensormaps)))
    {
        if !tensor.is_contiguous() {
            return Err(Error::tensor(name, "tensor must be compact row-major"));
        }
    }
    for tensor in immutable {
        if !tensor.is_contiguous() {
            return Err(Error::tensor(
                "input",
                "all prefill inputs must be compact row-major",
            ));
        }
        reject_overlap("input", tensor, state, output, tensormaps)?;
    }
    let state_bounds = state.byte_bounds()?;
    let output_bounds = output.byte_bounds()?;
    let workspace_bounds = tensormaps.byte_bounds()?;
    if bounds_overlap(state_bounds, output_bounds)
        || bounds_overlap(state_bounds, workspace_bounds)
        || bounds_overlap(output_bounds, workspace_bounds)
    {
        return Err(Error::tensor(
            "launch",
            "state, output, and TMA workspace must not overlap",
        ));
    }
    for (name, tensor, alignment) in [
        ("state", state, 16_usize),
        ("output", output, 16),
        ("tensormaps", tensormaps, 128),
    ] {
        if !tensor.effective_address()?.is_multiple_of(alignment) {
            return Err(Error::tensor(
                name,
                format!("CUDA address must be {alignment}-byte aligned"),
            ));
        }
    }
    Ok(())
}

fn reject_overlap(
    name: &'static str,
    tensor: &CudaTensor,
    state: &CudaTensor,
    output: &CudaTensor,
    tensormaps: &CudaTensor,
) -> Result<()> {
    let bounds = tensor.byte_bounds()?;
    if bounds_overlap(bounds, state.byte_bounds()?)
        || bounds_overlap(bounds, output.byte_bounds()?)
        || bounds_overlap(bounds, tensormaps.byte_bounds()?)
    {
        return Err(Error::tensor(
            name,
            "input must not overlap state, output, or TMA workspace",
        ));
    }
    Ok(())
}

fn require_alignment(name: &'static str, tensor: &CudaTensor, alignment: usize) -> Result<()> {
    if !tensor.effective_address()?.is_multiple_of(alignment) {
        return Err(Error::tensor(
            name,
            format!("CUDA address must be {alignment}-byte aligned"),
        ));
    }
    Ok(())
}
