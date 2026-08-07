use std::ffi::c_void;
use std::ptr::NonNull;

use crate::{Error, Result};

/// Element types currently accepted by the first GDN decode slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE float16.
    F16,
    /// Brain float16.
    BF16,
    /// IEEE float32.
    F32,
    /// Signed int32.
    I32,
}

impl DType {
    /// Bytes occupied by one scalar element.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }

    pub(crate) const fn dlpack(self) -> cutedsl_jit_types::DlDataType {
        use cutedsl_jit_types::{DlDataType, DlDataTypeCode};
        match self {
            Self::F16 => DlDataType::scalar(DlDataTypeCode::Float, 16),
            Self::BF16 => DlDataType::scalar(DlDataTypeCode::Bfloat, 16),
            Self::F32 => DlDataType::scalar(DlDataTypeCode::Float, 32),
            Self::I32 => DlDataType::scalar(DlDataTypeCode::Int, 32),
        }
    }
}

/// Borrowed CUDA tensor metadata used by the framework-independent API.
///
/// The descriptor never owns or frees device memory. Framework adapters should
/// construct it only while holding whatever guards keep the allocation live and
/// correctly synchronized.
#[derive(Debug)]
pub struct CudaTensor {
    data: NonNull<c_void>,
    byte_offset: u64,
    device_id: i32,
    dtype: DType,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl CudaTensor {
    /// Constructs a borrowed CUDA descriptor from raw allocation metadata.
    ///
    /// # Safety
    ///
    /// `data + byte_offset` must address a live CUDA allocation on `device_id`
    /// large enough for every element reachable through `shape` and `strides`.
    /// The allocation and all framework synchronization guards must remain live
    /// until work enqueued by any launch using this descriptor has completed.
    pub unsafe fn from_raw_parts(
        data: *mut c_void,
        byte_offset: u64,
        device_id: i32,
        dtype: DType,
        shape: impl Into<Vec<i64>>,
        strides: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        let data = NonNull::new(data).ok_or_else(|| Error::tensor("descriptor", "null data"))?;
        let shape = shape.into();
        let strides = strides.into();
        if device_id < 0 {
            return Err(Error::tensor(
                "descriptor",
                format!("negative CUDA device id {device_id}"),
            ));
        }
        if shape.is_empty() || shape.len() != strides.len() {
            return Err(Error::tensor(
                "descriptor",
                format!(
                    "shape/stride ranks must be equal and nonzero, found {} and {}",
                    shape.len(),
                    strides.len()
                ),
            ));
        }
        if shape.iter().any(|dimension| *dimension <= 0) {
            return Err(Error::tensor(
                "descriptor",
                format!("all dimensions must be positive, found {shape:?}"),
            ));
        }
        if strides.iter().any(|stride| *stride < 0) {
            return Err(Error::tensor(
                "descriptor",
                format!("negative strides are unsupported, found {strides:?}"),
            ));
        }
        let descriptor = Self {
            data,
            byte_offset,
            device_id,
            dtype,
            shape,
            strides,
        };
        descriptor.max_element_offset()?;
        Ok(descriptor)
    }

    /// Base allocation pointer.
    #[must_use]
    pub fn data(&self) -> *mut c_void {
        self.data.as_ptr()
    }

    /// Byte offset to the first logical element.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// CUDA device ordinal.
    #[must_use]
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Logical dimensions.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Element strides.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Whether this view is compact row-major, ignoring arbitrary strides on size-one modes.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1_i64;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension > 1 && stride != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(dimension) else {
                return false;
            };
            expected = next;
        }
        true
    }

    pub(crate) fn effective_address(&self) -> Result<usize> {
        let offset = usize::try_from(self.byte_offset).map_err(|_| {
            Error::tensor(
                "descriptor",
                format!("byte offset does not fit usize: {}", self.byte_offset),
            )
        })?;
        (self.data.as_ptr() as usize)
            .checked_add(offset)
            .ok_or_else(|| Error::tensor("descriptor", "data pointer plus byte offset overflows"))
    }

    pub(crate) fn byte_bounds(&self) -> Result<(usize, usize)> {
        let start = self.effective_address()?;
        let max_element_offset = usize::try_from(self.max_element_offset()?).map_err(|_| {
            Error::tensor("descriptor", "maximum element offset does not fit usize")
        })?;
        let span = max_element_offset
            .checked_mul(self.dtype.size_in_bytes())
            .and_then(|bytes| bytes.checked_add(self.dtype.size_in_bytes()))
            .ok_or_else(|| Error::tensor("descriptor", "tensor byte span overflows usize"))?;
        let end = start
            .checked_add(span)
            .ok_or_else(|| Error::tensor("descriptor", "tensor address range overflows usize"))?;
        Ok((start, end))
    }

    fn max_element_offset(&self) -> Result<i64> {
        self.shape
            .iter()
            .zip(&self.strides)
            .try_fold(0_i64, |offset, (&dimension, &stride)| {
                (dimension - 1)
                    .checked_mul(stride)
                    .and_then(|delta| offset.checked_add(delta))
                    .ok_or_else(|| {
                        Error::tensor("descriptor", "shape/stride element offset overflows i64")
                    })
            })
    }
}

/// Borrowed CUDA driver stream handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream(*mut c_void);

impl CudaStream {
    /// Legacy/default CUDA stream.
    pub const DEFAULT: Self = Self(std::ptr::null_mut());

    /// Wraps an existing CUDA driver stream.
    ///
    /// # Safety
    ///
    /// `raw` must be null or a live `CUstream` compatible with the tensors' CUDA
    /// context, and it must remain live until all enqueued work has completed.
    #[must_use]
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    /// Raw CUDA driver handle.
    #[must_use]
    pub const fn as_raw(self) -> *mut c_void {
        self.0
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// Keep the raw TVM types private to this crate while avoiding a direct dependency
// from consumers on the generic runtime's implementation modules.
pub(crate) mod cutedsl_jit_types {
    pub use flashinfer_gdn_sys::{DlDataType, DlDataTypeCode};
}
