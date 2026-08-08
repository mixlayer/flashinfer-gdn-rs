use std::sync::Arc;

use flashinfer_gdn_sys::{
    Bf16StateDecodeCompiler, Bf16StateDecodeKernel, Bf16StateDecodeSpecialization,
    Bf16StateDecodeTensors,
};

use crate::decode::{bounds_overlap, check_rank, expect};
use crate::tensor::DlTensorOwner;
use crate::{CudaStream, CudaTensor, DType, Error, Result};

/// Tensor arguments for one same-slot BF16-state decode launch.
pub struct Bf16StateDecodeCall<'a> {
    /// Main V-major state pool `[P,HV,V,K]`, BF16 and updated in place.
    pub state: &'a mut CudaTensor,
    /// Log-decay parameter `[HV]`, float32.
    pub a_log: &'a CudaTensor,
    /// Input-dependent decay `[B,1,HV]`, BF16.
    pub a: &'a CudaTensor,
    /// Decay bias `[HV]`, float32.
    pub dt_bias: &'a CudaTensor,
    /// Query `[B,1,H,K]`, BF16.
    pub q: &'a CudaTensor,
    /// Key `[B,1,H,K]`, BF16.
    pub k: &'a CudaTensor,
    /// Value `[B,1,HV,V]`, BF16.
    pub v: &'a CudaTensor,
    /// Update gate `[B,1,HV]`, BF16.
    pub beta: &'a CudaTensor,
    /// Output `[B,1,HV,V]`, BF16 and written in place.
    pub output: &'a mut CudaTensor,
    /// Pool indices `[B]`, int32, used for both reads and writes.
    ///
    /// Entries must be in `[0,P)` and unique within a concurrent batch. GPU
    /// index values are not synchronized back to the host for checking.
    pub state_indices: &'a CudaTensor,
    /// Zero-filled `[B]` int32 placeholder for disabled MTP accepted-step support.
    pub accepted_steps: &'a CudaTensor,
    /// Zero-filled `[B,1]` int32 placeholder for disabled per-token scatter.
    pub ssm_state_indices: &'a CudaTensor,
}

#[derive(Debug)]
struct PlanInner {
    kernel: Bf16StateDecodeKernel,
    device_id: i32,
}

/// Prepared, device-bound BF16-state single-token decode plan.
#[derive(Debug, Clone)]
pub struct Bf16StateDecodePlan {
    inner: Arc<PlanInner>,
}

impl Bf16StateDecodePlan {
    /// Compiles or loads the selected specialization.
    pub fn prepare(compiler: &Bf16StateDecodeCompiler, device_id: i32) -> Result<Self> {
        if device_id < 0 {
            return Err(Error::tensor(
                "plan",
                format!("negative CUDA device id {device_id}"),
            ));
        }
        let kernel = compiler.load()?;
        Self::from_kernel(kernel, device_id)
    }

    /// Wraps an already loaded sys-layer kernel.
    pub fn from_kernel(kernel: Bf16StateDecodeKernel, device_id: i32) -> Result<Self> {
        if device_id < 0 {
            return Err(Error::tensor(
                "plan",
                format!("negative CUDA device id {device_id}"),
            ));
        }
        Ok(Self {
            inner: Arc::new(PlanInner { kernel, device_id }),
        })
    }

    /// Compile-time specialization owned by this plan.
    #[must_use]
    pub fn specialization(&self) -> &Bf16StateDecodeSpecialization {
        self.inner.kernel.specialization()
    }

    /// CUDA device ordinal to which this plan is bound.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }

    /// Launches the prepared kernel without compilation, allocation, or file I/O.
    pub fn launch(&self, call: &mut Bf16StateDecodeCall<'_>, stream: CudaStream) -> Result<()> {
        validate_bf16_state_decode_for_device(self.specialization(), self.inner.device_id, call)?;

        let specialization = self.specialization();
        let mut intermediate_shape = [1_i64, 1, 1, specialization.k as i64];
        let mut intermediate_strides = [
            specialization.k as i64,
            specialization.k as i64,
            specialization.k as i64,
            1,
        ];
        let mut state = DlTensorOwner::from_tensor(call.state)?;
        let mut intermediate = DlTensorOwner::from_tensor(&*call.state)?;
        intermediate.tensor.shape = intermediate_shape.as_mut_ptr();
        intermediate.tensor.strides = intermediate_strides.as_mut_ptr();
        let mut a_log = DlTensorOwner::from_tensor(call.a_log)?;
        let mut a = DlTensorOwner::from_tensor(call.a)?;
        let mut dt_bias = DlTensorOwner::from_tensor(call.dt_bias)?;
        let mut q = DlTensorOwner::from_tensor(call.q)?;
        let mut k = DlTensorOwner::from_tensor(call.k)?;
        let mut v = DlTensorOwner::from_tensor(call.v)?;
        let mut beta = DlTensorOwner::from_tensor(call.beta)?;
        let mut output = DlTensorOwner::from_tensor(call.output)?;
        let mut state_indices = DlTensorOwner::from_tensor(call.state_indices)?;
        let mut output_state_indices = DlTensorOwner::from_tensor(call.state_indices)?;
        let mut accepted_steps = DlTensorOwner::from_tensor(call.accepted_steps)?;
        let mut ssm_state_indices = DlTensorOwner::from_tensor(call.ssm_state_indices)?;

        let mut tensors = Bf16StateDecodeTensors {
            state: &mut state.tensor,
            intermediate: &mut intermediate.tensor,
            a_log: &mut a_log.tensor,
            a: &mut a.tensor,
            dt_bias: &mut dt_bias.tensor,
            q: &mut q.tensor,
            k: &mut k.tensor,
            v: &mut v.tensor,
            beta: &mut beta.tensor,
            output: &mut output.tensor,
            state_indices: &mut state_indices.tensor,
            output_state_indices: &mut output_state_indices.tensor,
            accepted_steps: &mut accepted_steps.tensor,
            ssm_state_indices: &mut ssm_state_indices.tensor,
        };
        // SAFETY: validation checked the specialization contract; the intermediate
        // descriptor aliases the state's first elements but is compiled as unused.
        unsafe { self.inner.kernel.launch(&mut tensors, stream.as_raw())? };
        Ok(())
    }
}

/// Validates one launch independently of a loaded plan.
pub fn validate_bf16_state_decode(
    specialization: &Bf16StateDecodeSpecialization,
    call: &Bf16StateDecodeCall<'_>,
) -> Result<()> {
    validate_bf16_state_decode_for_device(specialization, call.q.device_id(), call)
}

fn validate_bf16_state_decode_for_device(
    specialization: &Bf16StateDecodeSpecialization,
    device_id: i32,
    call: &Bf16StateDecodeCall<'_>,
) -> Result<()> {
    specialization.validate()?;
    let [h, hv, k, v] = [
        specialization.h as i64,
        specialization.hv as i64,
        specialization.k as i64,
        specialization.v as i64,
    ];
    check_rank(call.q, "q", 4)?;
    let batch = call.q.shape()[0];
    expect(call.q, "q", DType::BF16, &[batch, 1, h, k], device_id)?;
    expect(call.k, "k", DType::BF16, &[batch, 1, h, k], device_id)?;
    expect(call.v, "v", DType::BF16, &[batch, 1, hv, v], device_id)?;
    expect(call.a, "a", DType::BF16, &[batch, 1, hv], device_id)?;
    expect(call.beta, "beta", DType::BF16, &[batch, 1, hv], device_id)?;
    expect(call.a_log, "a_log", DType::F32, &[hv], device_id)?;
    expect(call.dt_bias, "dt_bias", DType::F32, &[hv], device_id)?;
    expect(
        call.output,
        "output",
        DType::BF16,
        &[batch, 1, hv, v],
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
        call.accepted_steps,
        "accepted_steps",
        DType::I32,
        &[batch],
        device_id,
    )?;
    expect(
        call.ssm_state_indices,
        "ssm_state_indices",
        DType::I32,
        &[batch, 1],
        device_id,
    )?;
    check_rank(call.state, "state", 4)?;
    let pool_size = call.state.shape()[0];
    expect(
        call.state,
        "state",
        DType::BF16,
        &[pool_size, hv, v, k],
        device_id,
    )?;
    if !call.state.is_contiguous() {
        return Err(Error::tensor(
            "state",
            "BF16 state pool must be compact row-major [P,HV,V,K]",
        ));
    }

    for (name, tensor) in [
        ("state", &*call.state),
        ("a_log", call.a_log),
        ("a", call.a),
        ("dt_bias", call.dt_bias),
        ("q", call.q),
        ("k", call.k),
        ("v", call.v),
        ("beta", call.beta),
        ("output", &*call.output),
        ("state_indices", call.state_indices),
        ("accepted_steps", call.accepted_steps),
        ("ssm_state_indices", call.ssm_state_indices),
    ] {
        if tensor.effective_address()? % 32 != 0 {
            return Err(Error::tensor(
                name,
                format!(
                    "effective CUDA address must be 32-byte aligned, found {:#x}",
                    tensor.effective_address()?
                ),
            ));
        }
    }

    let state_bounds = call.state.byte_bounds()?;
    let output_bounds = call.output.byte_bounds()?;
    if bounds_overlap(state_bounds, output_bounds) {
        return Err(Error::tensor("output", "output must not overlap state"));
    }
    for (name, tensor) in [
        ("a_log", call.a_log),
        ("a", call.a),
        ("dt_bias", call.dt_bias),
        ("q", call.q),
        ("k", call.k),
        ("v", call.v),
        ("beta", call.beta),
        ("state_indices", call.state_indices),
        ("accepted_steps", call.accepted_steps),
        ("ssm_state_indices", call.ssm_state_indices),
    ] {
        let bounds = tensor.byte_bounds()?;
        if bounds_overlap(state_bounds, bounds) {
            return Err(Error::tensor(
                "state",
                format!("state must not overlap {name}"),
            ));
        }
        if bounds_overlap(output_bounds, bounds) {
            return Err(Error::tensor(
                "output",
                format!("output must not overlap {name}"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;

    use super::*;

    fn compact_strides(shape: &[i64]) -> Vec<i64> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1_i64;
        for (index, dimension) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride *= dimension;
        }
        strides
    }

    fn tensor(dtype: DType, shape: &[i64], address_slot: usize) -> CudaTensor {
        let address = 0x1000_0000 + address_slot * 0x1000_0000;
        // SAFETY: tests only validate metadata and never dereference these addresses.
        unsafe {
            CudaTensor::from_raw_parts(
                NonNull::<u8>::new(address as *mut u8)
                    .unwrap()
                    .cast()
                    .as_ptr(),
                0,
                0,
                dtype,
                shape.to_vec(),
                compact_strides(shape),
            )
            .unwrap()
        }
    }

    fn validate(state_dtype: DType) -> Result<()> {
        let spec = Bf16StateDecodeSpecialization::new("sm_121a", 1, 1, 128, 128, 2, 20)?;
        let mut state = tensor(state_dtype, &[5, 1, 128, 128], 0);
        let a_log = tensor(DType::F32, &[1], 1);
        let a = tensor(DType::BF16, &[2, 1, 1], 2);
        let dt_bias = tensor(DType::F32, &[1], 3);
        let q = tensor(DType::BF16, &[2, 1, 1, 128], 4);
        let k = tensor(DType::BF16, &[2, 1, 1, 128], 5);
        let v = tensor(DType::BF16, &[2, 1, 1, 128], 6);
        let beta = tensor(DType::BF16, &[2, 1, 1], 7);
        let mut output = tensor(DType::BF16, &[2, 1, 1, 128], 8);
        let indices = tensor(DType::I32, &[2], 9);
        let accepted = tensor(DType::I32, &[2], 10);
        let scatter = tensor(DType::I32, &[2, 1], 11);
        validate_bf16_state_decode(
            &spec,
            &Bf16StateDecodeCall {
                state: &mut state,
                a_log: &a_log,
                a: &a,
                dt_bias: &dt_bias,
                q: &q,
                k: &k,
                v: &v,
                beta: &beta,
                output: &mut output,
                state_indices: &indices,
                accepted_steps: &accepted,
                ssm_state_indices: &scatter,
            },
        )
    }

    #[test]
    fn validates_bf16_state_pool_contract() {
        validate(DType::BF16).unwrap();
    }

    #[test]
    fn rejects_float_state_pool() {
        assert!(matches!(
            validate(DType::F32),
            Err(Error::InvalidTensor { name: "state", .. })
        ));
    }
}
