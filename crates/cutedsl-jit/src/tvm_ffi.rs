//! Dynamic-only subset of the official Apache TVM FFI Rust ABI declarations.
//!
//! These layouts and values are kept aligned with `tvm-ffi-sys` 0.1.0-alpha.0.
//! We cannot depend on that crate directly yet because its build script requires
//! `tvm-ffi-config` and links a process-wide TVM runtime. CuTeDSL artifacts instead
//! select and dynamically load the exact runtime recorded in their manifest.

use std::ffi::c_void;

/// DLPack device type.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlDeviceType {
    /// CUDA device memory.
    Cuda = 2,
}

/// DLPack device descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlDevice {
    /// Device type.
    pub device_type: DlDeviceType,
    /// Device ordinal.
    pub device_id: i32,
}

/// DLPack scalar type code.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlDataTypeCode {
    /// Signed integer.
    Int = 0,
    /// IEEE floating point.
    Float = 2,
    /// Brain floating point.
    Bfloat = 4,
}

/// DLPack scalar data type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlDataType {
    /// [`DlDataTypeCode`] value.
    pub code: u8,
    /// Bits per lane.
    pub bits: u8,
    /// Vector lanes.
    pub lanes: u16,
}

impl DlDataType {
    /// Constructs a scalar DLPack data type.
    #[must_use]
    pub const fn scalar(code: DlDataTypeCode, bits: u8) -> Self {
        Self {
            code: code as u8,
            bits,
            lanes: 1,
        }
    }
}

/// Borrowed DLPack tensor descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct DlTensor {
    /// Base allocation pointer.
    pub data: *mut c_void,
    /// Device containing `data`.
    pub device: DlDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Element data type.
    pub dtype: DlDataType,
    /// Pointer to `ndim` shape values.
    pub shape: *mut i64,
    /// Pointer to `ndim` element-stride values.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first logical element.
    pub byte_offset: u64,
}

/// TVM FFI on-stack type indices used by CuTeDSL entrypoints.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvmFfiTypeIndex {
    /// No value.
    None = 0,
    /// Opaque pointer.
    OpaquePtr = 4,
    /// Borrowed `DLTensor*`.
    DlTensorPtr = 7,
}

/// Eight-byte data union in `TVMFFIAny`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union TvmFfiAnyData {
    /// Signed integer representation.
    pub v_int64: i64,
    /// Opaque pointer representation.
    pub v_ptr: *mut c_void,
    /// Unsigned representation used to initialize the union.
    pub v_uint64: u64,
}

/// Official TVM FFI on-stack argument value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TvmFfiAny {
    /// [`TvmFfiTypeIndex`] value.
    pub type_index: i32,
    /// Must be zero except for small-string values.
    pub zero_padding: u32,
    /// Argument payload.
    pub data: TvmFfiAnyData,
}

impl TvmFfiAny {
    /// Constructs a `None` value suitable for safe-call result storage.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            type_index: TvmFfiTypeIndex::None as i32,
            zero_padding: 0,
            data: TvmFfiAnyData { v_uint64: 0 },
        }
    }

    /// Constructs a borrowed `DLTensor*` argument.
    #[must_use]
    pub fn tensor(tensor: &mut DlTensor) -> Self {
        Self {
            type_index: TvmFfiTypeIndex::DlTensorPtr as i32,
            zero_padding: 0,
            data: TvmFfiAnyData {
                v_ptr: std::ptr::from_mut(tensor).cast(),
            },
        }
    }

    /// Constructs an opaque-pointer argument, including CUDA streams.
    #[must_use]
    pub const fn opaque(pointer: *mut c_void) -> Self {
        Self {
            type_index: TvmFfiTypeIndex::OpaquePtr as i32,
            zero_padding: 0,
            data: TvmFfiAnyData { v_ptr: pointer },
        }
    }
}

impl Default for TvmFfiAny {
    fn default() -> Self {
        Self::none()
    }
}

/// Generated TVM FFI safe-call function signature.
pub type TvmFfiSafeCall = unsafe extern "C" fn(
    handle: *mut c_void,
    arguments: *const TvmFfiAny,
    argument_count: i32,
    result: *mut TvmFfiAny,
) -> i32;

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    #[test]
    fn raw_layouts_match_official_64_bit_headers() {
        assert_eq!(size_of::<DlDevice>(), 8);
        assert_eq!(align_of::<DlDevice>(), 4);
        assert_eq!(size_of::<DlDataType>(), 4);
        assert_eq!(align_of::<DlDataType>(), 2);

        assert_eq!(size_of::<DlTensor>(), 48);
        assert_eq!(align_of::<DlTensor>(), 8);
        assert_eq!(offset_of!(DlTensor, data), 0);
        assert_eq!(offset_of!(DlTensor, device), 8);
        assert_eq!(offset_of!(DlTensor, ndim), 16);
        assert_eq!(offset_of!(DlTensor, dtype), 20);
        assert_eq!(offset_of!(DlTensor, shape), 24);
        assert_eq!(offset_of!(DlTensor, strides), 32);
        assert_eq!(offset_of!(DlTensor, byte_offset), 40);

        assert_eq!(size_of::<TvmFfiAny>(), 16);
        assert_eq!(align_of::<TvmFfiAny>(), 8);
        assert_eq!(offset_of!(TvmFfiAny, type_index), 0);
        assert_eq!(offset_of!(TvmFfiAny, zero_padding), 4);
        assert_eq!(offset_of!(TvmFfiAny, data), 8);
    }

    #[test]
    fn raw_values_match_official_headers() {
        assert_eq!(DlDeviceType::Cuda as u32, 2);
        assert_eq!(DlDataTypeCode::Int as u8, 0);
        assert_eq!(DlDataTypeCode::Float as u8, 2);
        assert_eq!(DlDataTypeCode::Bfloat as u8, 4);
        assert_eq!(TvmFfiTypeIndex::None as i32, 0);
        assert_eq!(TvmFfiTypeIndex::OpaquePtr as i32, 4);
        assert_eq!(TvmFfiTypeIndex::DlTensorPtr as i32, 7);
    }
}
