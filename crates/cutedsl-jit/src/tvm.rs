use std::ffi::{CString, c_void};
use std::fmt;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

use crate::tvm_ffi::{TvmFfiAny, TvmFfiSafeCall};
use crate::{Abi, Artifact, Error, Result};

type ErrorMoveFromRaised = unsafe extern "C" fn(*mut *mut c_void);
type ObjectDecRef = unsafe extern "C" fn(*mut c_void) -> i32;
type GetVersion = unsafe extern "C" fn(*mut TvmFfiVersion);

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

        let mut runtime_libraries = Vec::with_capacity(artifact.manifest().runtime_libraries.len());
        for runtime in &artifact.manifest().runtime_libraries {
            // SAFETY: the artifact validator checked this exact library's content
            // digest. Handles remain alive until after the generated module drops.
            // CuTeDSL and an embedding inference runtime can carry different
            // patch releases of TVM-FFI in the same process. Deep binding keeps
            // each runtime's static registry self-contained instead of binding
            // its constructors to an older RTLD_GLOBAL registry.
            let library = unsafe {
                Library::open(
                    Some(&runtime.path),
                    RTLD_NOW | RTLD_GLOBAL | libc::RTLD_DEEPBIND,
                )
            }
            .map_err(|error| {
                Error::DynamicLoad(format!(
                    "failed to load runtime library {}: {error}",
                    runtime.path.display()
                ))
            })?;
            runtime_libraries.push(library);
        }

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
        let module = unsafe {
            Library::open(
                Some(&module_path),
                RTLD_NOW | RTLD_GLOBAL | libc::RTLD_DEEPBIND,
            )
        }
        .map_err(|error| {
            Error::DynamicLoad(format!(
                "failed to load generated module {}: {error}",
                module_path.display()
            ))
        })?;
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
        if !error.is_null() {
            // SAFETY: ownership of the raised object was moved into error.
            let _ = unsafe { (self.object_dec_ref)(error) };
        }
        Err(Error::TvmCall { status })
    }
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
