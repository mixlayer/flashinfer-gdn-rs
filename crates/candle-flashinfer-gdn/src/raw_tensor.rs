use std::ffi::c_void;

use candle::cuda_backend::cudarc::driver::{
    CudaStream as DriverStream, DevicePtr, DevicePtrMut, SyncOnDrop,
};
use candle::cuda_backend::{CudaStorage, CudaStorageSlice};
use candle::{DType as CandleDType, Layout, Result, Storage, Tensor};
use flashinfer_gdn::{CudaTensor, DType};

use crate::{core_error, message};

#[derive(Debug, Clone)]
pub(crate) struct RawTensor {
    address: usize,
    dtype: DType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    byte_offset: u64,
    device_id: i32,
}

impl RawTensor {
    pub(crate) fn new(
        address: usize,
        dtype: CandleDType,
        layout: &Layout,
        device_id: i32,
        name: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            address,
            dtype: convert_dtype(dtype)?,
            shape: dimensions(layout.dims(), name)?,
            strides: dimensions(layout.stride(), name)?,
            byte_offset: byte_offset(dtype, layout, name)?,
            device_id,
        })
    }

    pub(crate) fn descriptor(&self) -> Result<CudaTensor> {
        // SAFETY: the caller holds Candle's mutable storage guard and cudarc use
        // guard for the complete nested launch.
        unsafe {
            CudaTensor::from_raw_parts(
                self.address as *mut c_void,
                self.byte_offset,
                self.device_id,
                self.dtype,
                self.shape.clone(),
                self.strides.clone(),
            )
        }
        .map_err(core_error)
    }
}

pub(crate) fn descriptor(
    address: usize,
    storage: &CudaStorage,
    layout: &Layout,
    name: &'static str,
) -> Result<CudaTensor> {
    let ordinal = storage.device.cuda_stream().context().ordinal();
    let device_id = i32::try_from(ordinal)
        .map_err(|_| message(format!("CUDA ordinal does not fit i32: {ordinal}")))?;
    descriptor_parts(address, storage_dtype(storage)?, layout, device_id, name)
}

pub(crate) fn descriptor_parts(
    address: usize,
    dtype: CandleDType,
    layout: &Layout,
    device_id: i32,
    name: &'static str,
) -> Result<CudaTensor> {
    // SAFETY: the surrounding launch retains the Candle storage lock and cudarc
    // access guard until after the kernel has been enqueued.
    unsafe {
        CudaTensor::from_raw_parts(
            address as *mut c_void,
            byte_offset(dtype, layout, name)?,
            device_id,
            convert_dtype(dtype)?,
            dimensions(layout.dims(), name)?,
            dimensions(layout.stride(), name)?,
        )
    }
    .map_err(core_error)
}

pub(crate) fn base_address(
    tensor: &Tensor,
    stream: &DriverStream,
    name: &'static str,
) -> Result<usize> {
    let (storage, _) = tensor.storage_and_layout();
    let storage = cuda_storage(&storage, name)?;
    let (address, _use_guard) = immutable_address(storage, stream)?;
    Ok(address)
}

pub(crate) fn cuda_storage<'a>(
    storage: &'a Storage,
    name: &'static str,
) -> Result<&'a CudaStorage> {
    match storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => Err(message(format!("{name} must use CUDA storage"))),
    }
}

pub(crate) fn immutable_address<'a>(
    storage: &'a CudaStorage,
    stream: &'a DriverStream,
) -> Result<(usize, SyncOnDrop<'a>)> {
    let (address, guard) = match &storage.slice {
        CudaStorageSlice::I32(slice) => slice.device_ptr(stream),
        CudaStorageSlice::BF16(slice) => slice.device_ptr(stream),
        CudaStorageSlice::F16(slice) => slice.device_ptr(stream),
        CudaStorageSlice::F32(slice) => slice.device_ptr(stream),
        _ => {
            return Err(message(format!(
                "unsupported GDN CUDA dtype {:?}",
                storage_dtype(storage)?
            )));
        }
    };
    Ok((address as usize, guard))
}

pub(crate) fn mutable_address<'a>(
    storage: &'a mut CudaStorage,
    stream: &'a DriverStream,
) -> Result<(usize, SyncOnDrop<'a>)> {
    let dtype = storage_dtype(storage)?;
    let (address, guard) = match &mut storage.slice {
        CudaStorageSlice::I32(slice) => slice.device_ptr_mut(stream),
        CudaStorageSlice::BF16(slice) => slice.device_ptr_mut(stream),
        CudaStorageSlice::F16(slice) => slice.device_ptr_mut(stream),
        CudaStorageSlice::F32(slice) => slice.device_ptr_mut(stream),
        _ => return Err(message(format!("unsupported GDN CUDA dtype {dtype:?}"))),
    };
    Ok((address as usize, guard))
}

pub(crate) fn storage_dtype(storage: &CudaStorage) -> Result<CandleDType> {
    Ok(match storage.slice {
        CudaStorageSlice::I32(_) => CandleDType::I32,
        CudaStorageSlice::BF16(_) => CandleDType::BF16,
        CudaStorageSlice::F16(_) => CandleDType::F16,
        CudaStorageSlice::F32(_) => CandleDType::F32,
        _ => return Err(message("unsupported GDN CUDA storage dtype")),
    })
}

pub(crate) fn ensure_ordinal(
    storage: &CudaStorage,
    expected: i32,
    name: &'static str,
) -> Result<()> {
    let actual = storage.device.cuda_stream().context().ordinal();
    if usize::try_from(expected).ok() != Some(actual) {
        return Err(message(format!(
            "{name} is on CUDA device {actual}, but the plan is bound to {expected}"
        )));
    }
    Ok(())
}

fn convert_dtype(dtype: CandleDType) -> Result<DType> {
    Ok(match dtype {
        CandleDType::I32 => DType::I32,
        CandleDType::BF16 => DType::BF16,
        CandleDType::F16 => DType::F16,
        CandleDType::F32 => DType::F32,
        _ => return Err(message(format!("unsupported GDN dtype {dtype:?}"))),
    })
}

fn dimensions(values: &[usize], name: &'static str) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|value| {
            i64::try_from(*value)
                .map_err(|_| message(format!("{name} dimension does not fit i64: {value}")))
        })
        .collect()
}

fn byte_offset(dtype: CandleDType, layout: &Layout, name: &'static str) -> Result<u64> {
    let bytes = layout
        .start_offset()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| message(format!("{name} byte offset overflows usize")))?;
    u64::try_from(bytes).map_err(|_| message(format!("{name} byte offset does not fit u64")))
}
