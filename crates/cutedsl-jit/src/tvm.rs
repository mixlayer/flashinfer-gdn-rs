use std::ffi::{CString, c_void};
use std::fmt;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

use crate::tvm_ffi::{TvmFfiAny, TvmFfiSafeCall};
use crate::{Abi, Artifact, Error, Result, RuntimeLibrary};

type ErrorMoveFromRaised = unsafe extern "C" fn(*mut *mut c_void);
type ObjectDecRef = unsafe extern "C" fn(*mut c_void) -> i32;
type GetVersion = unsafe extern "C" fn(*mut TvmFfiVersion);

const TVM_FFI_ERROR_TYPE_INDEX: i32 = 67;

#[repr(C)]
struct TvmFfiObject {
    combined_ref_count: u64,
    type_index: i32,
    padding: u32,
    deleter: *mut c_void,
}

#[repr(C)]
struct TvmFfiByteArray {
    data: *const u8,
    size: usize,
}

#[repr(C)]
struct TvmFfiErrorCell {
    kind: TvmFfiByteArray,
    message: TvmFfiByteArray,
    backtrace: TvmFfiByteArray,
    update_backtrace: *mut c_void,
    cause_chain: *mut c_void,
    extra_context: *mut c_void,
}

/// Version returned by TVMFFIGetVersion.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TvmFfiVersion {
    /// ABI major version.
    pub major: u32,
    /// ABI minor version.
    pub minor: u32,
    /// ABI patch version.
    pub patch: u32,
}

impl fmt::Display for TvmFfiVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Loaded TVM FFI module whose dependencies and ABI version have been validated.
pub struct TvmModule {
    artifact: Artifact,
    entry: TvmFfiSafeCall,
    error_move_from_raised: ErrorMoveFromRaised,
    object_dec_ref: ObjectDecRef,
    version: TvmFfiVersion,
    // Drop the generated module before its runtime dependencies.
    module: Library,
    runtime_libraries: Vec<Library>,
}

impl fmt::Debug for TvmModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TvmModule")
            .field("artifact", &self.artifact)
            .field("entry_symbol", &self.artifact.manifest().entry_symbol)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl TvmModule {
    /// Loads a validated artifact and resolves its generated safe-call entrypoint.
    pub fn load(artifact: Artifact) -> Result<Self> {
        if artifact.manifest().abi != Abi::TvmFfi {
            return Err(Error::InvalidManifest {
                path: artifact.directory().join("manifest.json"),
                message: format!(
                    "TVM loader cannot open {:?} artifact",
                    artifact.manifest().abi
                ),
            });
        }

        let runtime_libraries = load_runtime_libraries(
            &artifact.manifest().runtime_libraries,
            artifact.manifest().tvm_ffi_runtime_version.as_deref(),
        )?;
        let error_move_from_raised = find_symbol(
            &runtime_libraries,
            b"TVMFFIErrorMoveFromRaised\0",
            "TVMFFIErrorMoveFromRaised",
        )?;
        let object_dec_ref = find_symbol(
            &runtime_libraries,
            b"TVMFFIObjectDecRef\0",
            "TVMFFIObjectDecRef",
        )?;
        let get_version: GetVersion = find_symbol(
            &runtime_libraries,
            b"TVMFFIGetVersion\0",
            "TVMFFIGetVersion",
        )?;
        let mut version = TvmFfiVersion::default();
        // SAFETY: the function was resolved from a digest-validated TVM runtime
        // using the official TVMFFIGetVersion C signature.
        unsafe { get_version(&mut version) };
        if let Some(expected) = &artifact.manifest().tvm_ffi_runtime_version {
            let actual = version.to_string();
            if *expected != actual {
                return Err(Error::TvmVersionMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        let module_path = artifact.file("module")?;
        // SAFETY: the module content and its runtime dependencies were validated
        // against the compiler-produced manifest.
        let module = unsafe { Library::open(Some(&module_path), RTLD_NOW | RTLD_GLOBAL) }.map_err(
            |error| {
                Error::DynamicLoad(format!(
                    "failed to load generated module {}: {error}",
                    module_path.display()
                ))
            },
        )?;
        let entry_name =
            CString::new(artifact.manifest().entry_symbol.as_bytes()).map_err(|_| {
                Error::InvalidManifest {
                    path: artifact.directory().join("manifest.json"),
                    message: "entry_symbol contains an interior NUL".into(),
                }
            })?;
        // SAFETY: CuTeDSL's TVM export contract assigns this signature to every
        // generated TVM safe-call symbol.
        let entry = unsafe {
            *module
                .get::<TvmFfiSafeCall>(entry_name.as_bytes_with_nul())
                .map_err(|error| {
                    Error::DynamicLoad(format!(
                        "failed to resolve {}: {error}",
                        artifact.manifest().entry_symbol
                    ))
                })?
        };

        Ok(Self {
            artifact,
            entry,
            error_move_from_raised,
            object_dec_ref,
            version,
            module,
            runtime_libraries,
        })
    }

    /// Compiler artifact held alive by this module.
    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// TVM C-runtime ABI version.
    pub fn version(&self) -> TvmFfiVersion {
        self.version
    }

    /// Calls the generated entrypoint with typed TVM FFI arguments.
    ///
    /// # Safety
    ///
    /// Every argument payload must satisfy the generated entrypoint contract for
    /// the duration of the call and any asynchronous work it enqueues.
    pub unsafe fn call(&self, arguments: &[TvmFfiAny]) -> Result<()> {
        let argument_count = i32::try_from(arguments.len()).map_err(|_| {
            Error::InvalidInput(format!(
                "TVM FFI argument count exceeds i32: {}",
                arguments.len()
            ))
        })?;
        let mut result = TvmFfiAny::none();
        // SAFETY: the caller upholds each argument's generated ABI contract.
        unsafe {
            self.call_raw(
                std::ptr::null_mut(),
                arguments.as_ptr().cast(),
                argument_count,
                std::ptr::from_mut(&mut result).cast(),
            )
        }
    }

    /// Calls the generated TVM FFI entrypoint and releases any raised error object.
    ///
    /// # Safety
    ///
    /// The resource handle, arguments, and result pointers must follow the official
    /// TVMFFISafeCall contract for this generated entrypoint. In particular,
    /// arguments must point to argument_count valid TVMFFIAny values and result
    /// must point to writable TVMFFIAny storage.
    pub unsafe fn call_raw(
        &self,
        resource_handle: *mut c_void,
        arguments: *const c_void,
        argument_count: i32,
        result: *mut c_void,
    ) -> Result<()> {
        // SAFETY: caller upholds the generated safe-call argument contract.
        let status = unsafe {
            (self.entry)(
                resource_handle,
                arguments.cast(),
                argument_count,
                result.cast(),
            )
        };
        if status == 0 {
            return Ok(());
        }

        let mut error = std::ptr::null_mut();
        // SAFETY: a failed safe call stores its raised object in TVM thread-local
        // state and transfers ownership through TVMFFIErrorMoveFromRaised.
        unsafe { (self.error_move_from_raised)(&mut error) };
        let (kind, message) = if error.is_null() {
            (
                "UnknownError".into(),
                "TVM did not provide an error object".into(),
            )
        } else {
            // SAFETY: TVM transferred one owned error object to this function.
            unsafe { error_details(error) }
        };
        if !error.is_null() {
            // SAFETY: ownership of the raised object was moved into error.
            let _ = unsafe { (self.object_dec_ref)(error) };
        }
        Err(Error::TvmCall {
            status,
            kind,
            message,
        })
    }
}

unsafe fn error_details(error: *mut c_void) -> (String, String) {
    // SAFETY: caller guarantees error points to a live TVM FFI object header.
    let object = unsafe { &*error.cast::<TvmFfiObject>() };
    if object.type_index != TVM_FFI_ERROR_TYPE_INDEX {
        return (
            "UnknownError".into(),
            format!("TVM raised object type index {}", object.type_index),
        );
    }
    // TVMFFIErrorCell immediately follows the common TVMFFIObject header.
    // SAFETY: the checked static type index guarantees this object layout.
    let cell = unsafe {
        &*error
            .cast::<u8>()
            .add(std::mem::size_of::<TvmFfiObject>())
            .cast::<TvmFfiErrorCell>()
    };
    // SAFETY: TVM owns both byte arrays until the error object is released.
    let kind = unsafe { byte_array_to_string(&cell.kind) };
    // SAFETY: same lifetime guarantee as kind.
    let message = unsafe { byte_array_to_string(&cell.message) };
    (kind, message)
}

unsafe fn byte_array_to_string(bytes: &TvmFfiByteArray) -> String {
    if bytes.data.is_null() || bytes.size == 0 {
        return String::new();
    }
    // SAFETY: caller guarantees TVM owns a readable byte array of this length.
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(bytes.data, bytes.size) })
        .into_owned()
}

fn load_runtime_libraries(
    runtimes: &[RuntimeLibrary],
    expected_version: Option<&str>,
) -> Result<Vec<Library>> {
    if runtimes.is_empty() {
        return Err(Error::DynamicLoad(
            "TVM artifact declares no runtime libraries".into(),
        ));
    }
    // Reuse an already-global TVM runtime only when it exactly matches the
    // compiler artifact. This turns an otherwise fatal duplicate-registry C++
    // initializer into a normal version error when runtimes are loaded in the
    // wrong order.
    let process = Library::this();
    let existing_version = unsafe { process.get::<GetVersion>(b"TVMFFIGetVersion\0") }
        .ok()
        .map(|get_version| {
            let mut version = TvmFfiVersion::default();
            // SAFETY: the symbol has the official TVMFFIGetVersion signature.
            unsafe { get_version(&mut version) };
            version
        });
    if let (Some(expected), Some(actual)) = (expected_version, existing_version)
        && expected != actual.to_string()
    {
        return Err(Error::TvmVersionMismatch {
            expected: expected.to_owned(),
            actual: actual.to_string(),
        });
    }

    let mut libraries = Vec::with_capacity(runtimes.len() + 1);
    for runtime in runtimes {
        if existing_version.is_some()
            && runtime.path.file_name().and_then(|name| name.to_str()) == Some("libtvm_ffi.so")
        {
            continue;
        }
        // SAFETY: the artifact validator checked this exact library's content
        // digest. Handles remain alive until after every generated module drops.
        let library = unsafe { Library::open(Some(&runtime.path), RTLD_NOW | RTLD_GLOBAL) }
            .map_err(|error| {
                Error::DynamicLoad(format!(
                    "failed to load runtime library {}: {error}",
                    runtime.path.display()
                ))
            })?;
        libraries.push(library);
    }
    if existing_version.is_some() {
        libraries.push(process);
    }

    Ok(libraries)
}

fn find_symbol<T: Copy>(libraries: &[Library], symbol: &[u8], display_name: &str) -> Result<T> {
    for library in libraries {
        // SAFETY: every requested T is the official C signature for this symbol.
        if let Ok(value) = unsafe { library.get::<T>(symbol) } {
            return Ok(*value);
        }
    }
    Err(Error::DynamicLoad(format!(
        "runtime libraries do not export {display_name}"
    )))
}

impl Drop for TvmModule {
    fn drop(&mut self) {
        // Read these fields so their lifetime relationship remains explicit even
        // if a future refactor changes drop order.
        let _ = &self.module;
        let _ = &self.runtime_libraries;
    }
}
