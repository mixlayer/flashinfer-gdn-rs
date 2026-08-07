use std::sync::Arc;

use flashinfer_gdn_sys::{
    DlDevice, DlDeviceType, DlTensor, DtBiasDType, InputDType, PretransposeDecodeCompiler,
    PretransposeDecodeKernel, PretransposeDecodeSpecialization, PretransposeDecodeTensors,
};

use crate::{CudaStream, CudaTensor, DType, Error, Result};

/// Tensor arguments for one pretransposed float-state decode launch.
pub struct PretransposeDecodeCall<'a> {
    /// Direct state `[B,HV,V,K]` or state pool `[P,HV,V,K]`, updated in place.
    pub state: &'a mut CudaTensor,
    /// Log-decay parameter `[HV]`, float32.
    pub a_log: &'a CudaTensor,
    /// Input-dependent decay `[B,1,HV]`.
    pub a: &'a CudaTensor,
    /// Decay bias `[HV]`.
    pub dt_bias: &'a CudaTensor,
    /// Query `[B,1,H,K]`.
    pub q: &'a CudaTensor,
    /// Key `[B,1,H,K]`.
    pub k: &'a CudaTensor,
    /// Value `[B,1,HV,V]`.
    pub v: &'a CudaTensor,
    /// Update gate `[B,1,HV]`.
    pub beta: &'a CudaTensor,
    /// BF16 output `[B,1,HV,V]`, written in place.
    pub output: &'a mut CudaTensor,
    /// State-pool read indices `[B]`, int32. A zero-filled auxiliary tensor is used in direct mode.
    pub state_indices: &'a CudaTensor,
    /// State-pool write indices `[B]`, int32. May alias `state_indices`.
    pub output_state_indices: &'a CudaTensor,
    /// Zero-filled `[B+1]` int32 auxiliary tensor reserved by the upstream non-varlen ABI.
    pub cu_seqlens: &'a CudaTensor,
}

#[derive(Debug)]
struct PlanInner {
    kernel: PretransposeDecodeKernel,
    device_id: i32,
}

/// Prepared, device-bound pretransposed decode plan.
#[derive(Debug, Clone)]
pub struct PretransposeDecodePlan {
    inner: Arc<PlanInner>,
}

impl PretransposeDecodePlan {
    /// Compiles or loads the selected specialization before any CUDA Graph capture begins.
    pub fn prepare(compiler: &PretransposeDecodeCompiler, device_id: i32) -> Result<Self> {
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
    pub fn from_kernel(kernel: PretransposeDecodeKernel, device_id: i32) -> Result<Self> {
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
    pub fn specialization(&self) -> &PretransposeDecodeSpecialization {
        self.inner.kernel.specialization()
    }

    /// CUDA device ordinal to which this plan is bound.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.inner.device_id
    }

    /// Launches the prepared kernel without compilation, loading, allocation, or file I/O.
    ///
    /// The caller owns any eager warmup and CUDA Graph capture lifecycle. The same
    /// method is used for ordinary execution and stream capture.
    pub fn launch(&self, call: &mut PretransposeDecodeCall<'_>, stream: CudaStream) -> Result<()> {
        validate_pretranspose_decode_for_device(self.specialization(), self.inner.device_id, call)?;

        let batch = call.q.shape()[0];
        let specialization = self.specialization();
        let mut direct_state_shape = [
            checked_mul(batch, specialization.hv as i64, "B*HV")?,
            specialization.v as i64,
            specialization.k as i64,
        ];
        let mut direct_state_strides = [
            checked_mul(specialization.v as i64, specialization.k as i64, "V*K")?,
            specialization.k as i64,
            1,
        ];
        let mut state = DlTensorOwner::from_tensor(call.state)?;
        if !specialization.use_pool_indexing {
            state.tensor.ndim = 3;
            state.tensor.shape = direct_state_shape.as_mut_ptr();
            state.tensor.strides = direct_state_strides.as_mut_ptr();
        }
        let mut a_log = DlTensorOwner::from_tensor(call.a_log)?;
        let mut a = DlTensorOwner::from_tensor(call.a)?;
        let mut dt_bias = DlTensorOwner::from_tensor(call.dt_bias)?;
        let mut q = DlTensorOwner::from_tensor(call.q)?;
        let mut k = DlTensorOwner::from_tensor(call.k)?;
        let mut v = DlTensorOwner::from_tensor(call.v)?;
        let mut beta = DlTensorOwner::from_tensor(call.beta)?;
        let mut output = DlTensorOwner::from_tensor(call.output)?;
        let mut state_indices = DlTensorOwner::from_tensor(call.state_indices)?;
        let mut output_state_indices = DlTensorOwner::from_tensor(call.output_state_indices)?;
        let mut cu_seqlens = DlTensorOwner::from_tensor(call.cu_seqlens)?;

        let mut tensors = PretransposeDecodeTensors {
            state: &mut state.tensor,
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
            cu_seqlens: &mut cu_seqlens.tensor,
        };
        // SAFETY: validation above checked the generated specialization contract;
        // CudaTensor construction guarantees backing storage and stream lifetimes.
        unsafe { self.inner.kernel.launch(&mut tensors, stream.as_raw())? };
        Ok(())
    }
}

/// Validates one launch independently of a loaded plan.
pub fn validate_pretranspose_decode(
    specialization: &PretransposeDecodeSpecialization,
    call: &PretransposeDecodeCall<'_>,
) -> Result<()> {
    validate_pretranspose_decode_for_device(specialization, call.q.device_id(), call)
}

fn validate_pretranspose_decode_for_device(
    specialization: &PretransposeDecodeSpecialization,
    device_id: i32,
    call: &PretransposeDecodeCall<'_>,
) -> Result<()> {
    specialization.validate()?;
    let io_dtype = match specialization.io_dtype {
        InputDType::Float16 => DType::F16,
        InputDType::Bfloat16 => DType::BF16,
    };
    let dt_bias_dtype = match specialization.dt_bias_dtype {
        DtBiasDType::Bfloat16 => DType::BF16,
        DtBiasDType::Float32 => DType::F32,
    };
    let [h, hv, k, v] = [
        specialization.h as i64,
        specialization.hv as i64,
        specialization.k as i64,
        specialization.v as i64,
    ];

    check_rank(call.q, "q", 4)?;
    let batch = call.q.shape()[0];
    expect(call.q, "q", io_dtype, &[batch, 1, h, k], device_id)?;
    expect(call.k, "k", io_dtype, &[batch, 1, h, k], device_id)?;
    expect(call.v, "v", io_dtype, &[batch, 1, hv, v], device_id)?;
    expect(call.a, "a", io_dtype, &[batch, 1, hv], device_id)?;
    expect(call.beta, "beta", io_dtype, &[batch, 1, hv], device_id)?;
    expect(call.a_log, "a_log", DType::F32, &[hv], device_id)?;
    expect(call.dt_bias, "dt_bias", dt_bias_dtype, &[hv], device_id)?;
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
        call.output_state_indices,
        "output_state_indices",
        DType::I32,
        &[batch],
        device_id,
    )?;
    expect(
        call.cu_seqlens,
        "cu_seqlens",
        DType::I32,
        &[batch + 1],
        device_id,
    )?;

    if specialization.use_pool_indexing {
        check_rank(call.state, "state", 4)?;
        let pool_size = call.state.shape()[0];
        expect(
            call.state,
            "state",
            DType::F32,
            &[pool_size, hv, v, k],
            device_id,
        )?;
        if call.state.strides()[3] != 1 {
            return Err(Error::tensor(
                "state",
                format!(
                    "indexed pretransposed state must be K-contiguous, found strides {:?}",
                    call.state.strides()
                ),
            ));
        }
        if call.state.strides()[..3]
            .iter()
            .any(|stride| stride % 4 != 0)
        {
            return Err(Error::tensor(
                "state",
                format!(
                    "indexed pretransposed state outer strides must be multiples of four float elements for 16-byte cp.async alignment, found {:?}",
                    call.state.strides()
                ),
            ));
        }
    } else {
        expect(
            call.state,
            "state",
            DType::F32,
            &[batch, hv, v, k],
            device_id,
        )?;
        if !call.state.is_contiguous() {
            return Err(Error::tensor(
                "state",
                "direct pretransposed state must be compact row-major",
            ));
        }
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
        ("output_state_indices", call.output_state_indices),
        ("cu_seqlens", call.cu_seqlens),
    ] {
        if tensor.effective_address()? % 16 != 0 {
            return Err(Error::tensor(
                name,
                format!(
                    "effective CUDA address must be 16-byte aligned, found {:#x}",
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
        ("output_state_indices", call.output_state_indices),
        ("cu_seqlens", call.cu_seqlens),
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

fn bounds_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

fn check_rank(tensor: &CudaTensor, name: &'static str, rank: usize) -> Result<()> {
    if tensor.shape().len() != rank {
        return Err(Error::tensor(
            name,
            format!("expected rank {rank}, found shape {:?}", tensor.shape()),
        ));
    }
    Ok(())
}

fn expect(
    tensor: &CudaTensor,
    name: &'static str,
    dtype: DType,
    shape: &[i64],
    device_id: i32,
) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(Error::tensor(
            name,
            format!("expected dtype {dtype:?}, found {:?}", tensor.dtype()),
        ));
    }
    if tensor.shape() != shape {
        return Err(Error::tensor(
            name,
            format!("expected shape {shape:?}, found {:?}", tensor.shape()),
        ));
    }
    if tensor.device_id() != device_id {
        return Err(Error::tensor(
            name,
            format!(
                "expected CUDA device {device_id}, found {}",
                tensor.device_id()
            ),
        ));
    }
    if tensor.strides().last() != Some(&1) {
        return Err(Error::tensor(
            name,
            format!(
                "innermost dimension must be contiguous, found strides {:?}",
                tensor.strides()
            ),
        ));
    }
    Ok(())
}

fn checked_mul(left: i64, right: i64, label: &'static str) -> Result<i64> {
    left.checked_mul(right)
        .ok_or_else(|| Error::tensor("state", format!("{label} overflows i64")))
}

#[cfg(test)]
fn compact_strides(shape: &[i64]) -> Result<Vec<i64>> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1_i64;
    for (index, dimension) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride = checked_mul(stride, *dimension, "compact stride")?;
    }
    Ok(strides)
}

struct DlTensorOwner<'a> {
    tensor: DlTensor,
    _source: std::marker::PhantomData<&'a CudaTensor>,
}

impl<'a> DlTensorOwner<'a> {
    fn from_tensor(source: &'a CudaTensor) -> Result<Self> {
        let ndim = i32::try_from(source.shape().len()).map_err(|_| {
            Error::tensor(
                "descriptor",
                format!("tensor rank exceeds i32: {}", source.shape().len()),
            )
        })?;
        let tensor = DlTensor {
            data: source.data(),
            device: DlDevice {
                device_type: DlDeviceType::Cuda,
                device_id: source.device_id(),
            },
            ndim,
            dtype: source.dtype().dlpack(),
            shape: source.shape().as_ptr().cast_mut(),
            strides: source.strides().as_ptr().cast_mut(),
            byte_offset: source.byte_offset(),
        };
        Ok(Self {
            tensor,
            _source: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;

    use super::*;

    fn tensor(dtype: DType, shape: &[i64], address_slot: usize) -> CudaTensor {
        let strides = compact_strides(shape).unwrap();
        tensor_with_strides(dtype, shape, strides, address_slot)
    }

    fn tensor_with_strides(
        dtype: DType,
        shape: &[i64],
        strides: Vec<i64>,
        address_slot: usize,
    ) -> CudaTensor {
        let address = 0x1000_0000 + address_slot * 0x1000_0000;
        // SAFETY: tests only exercise metadata validation and never dereference or launch.
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
                strides,
            )
            .unwrap()
        }
    }

    #[test]
    fn validates_direct_decode_contract() {
        let spec = PretransposeDecodeSpecialization::default();
        let mut state = tensor(DType::F32, &[2, 16, 128, 128], 0);
        let a_log = tensor(DType::F32, &[16], 1);
        let a = tensor(DType::BF16, &[2, 1, 16], 2);
        let dt_bias = tensor(DType::F32, &[16], 3);
        let q = tensor(DType::BF16, &[2, 1, 16, 128], 4);
        let k = tensor(DType::BF16, &[2, 1, 16, 128], 5);
        let v = tensor(DType::BF16, &[2, 1, 16, 128], 6);
        let beta = tensor(DType::BF16, &[2, 1, 16], 7);
        let mut output = tensor(DType::BF16, &[2, 1, 16, 128], 8);
        let indices = tensor(DType::I32, &[2], 9);
        let output_indices = tensor(DType::I32, &[2], 10);
        let cu_seqlens = tensor(DType::I32, &[3], 11);
        let call = PretransposeDecodeCall {
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
            output_state_indices: &output_indices,
            cu_seqlens: &cu_seqlens,
        };
        validate_pretranspose_decode(&spec, &call).unwrap();
    }

    #[test]
    fn rejects_wrong_query_shape() {
        let spec = PretransposeDecodeSpecialization::default();
        let mut state = tensor(DType::F32, &[2, 16, 128, 128], 0);
        let a_log = tensor(DType::F32, &[16], 1);
        let a = tensor(DType::BF16, &[2, 1, 16], 2);
        let dt_bias = tensor(DType::F32, &[16], 3);
        let q = tensor(DType::BF16, &[2, 1, 8, 128], 4);
        let k = tensor(DType::BF16, &[2, 1, 16, 128], 5);
        let v = tensor(DType::BF16, &[2, 1, 16, 128], 6);
        let beta = tensor(DType::BF16, &[2, 1, 16], 7);
        let mut output = tensor(DType::BF16, &[2, 1, 16, 128], 8);
        let indices = tensor(DType::I32, &[2], 9);
        let output_indices = tensor(DType::I32, &[2], 10);
        let cu_seqlens = tensor(DType::I32, &[3], 11);
        let call = PretransposeDecodeCall {
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
            output_state_indices: &output_indices,
            cu_seqlens: &cu_seqlens,
        };
        assert!(matches!(
            validate_pretranspose_decode(&spec, &call),
            Err(Error::InvalidTensor { name: "q", .. })
        ));
    }

    #[test]
    fn rejects_misaligned_indexed_state_stride() {
        let spec = PretransposeDecodeSpecialization::default().pool_indexing(true);
        let mut state = tensor_with_strides(
            DType::F32,
            &[3, 16, 128, 128],
            vec![16 * 128 * 129, 128 * 129, 129, 1],
            0,
        );
        let a_log = tensor(DType::F32, &[16], 1);
        let a = tensor(DType::BF16, &[2, 1, 16], 2);
        let dt_bias = tensor(DType::F32, &[16], 3);
        let q = tensor(DType::BF16, &[2, 1, 16, 128], 4);
        let k = tensor(DType::BF16, &[2, 1, 16, 128], 5);
        let v = tensor(DType::BF16, &[2, 1, 16, 128], 6);
        let beta = tensor(DType::BF16, &[2, 1, 16], 7);
        let mut output = tensor(DType::BF16, &[2, 1, 16, 128], 8);
        let indices = tensor(DType::I32, &[2], 9);
        let output_indices = tensor(DType::I32, &[2], 10);
        let cu_seqlens = tensor(DType::I32, &[3], 11);
        let call = PretransposeDecodeCall {
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
            output_state_indices: &output_indices,
            cu_seqlens: &cu_seqlens,
        };
        assert!(matches!(
            validate_pretranspose_decode(&spec, &call),
            Err(Error::InvalidTensor { name: "state", .. })
        ));
    }
}
