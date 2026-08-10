#![deny(unsafe_op_in_unsafe_fn)]
//! Raw FlashInfer BF16-state GDN decode specialization and entrypoint integration.
//!
//! This crate owns the versioned CuTeDSL shims and the unsafe boundary between
//! GDN-specific argument schemas and `cutedsl-jit` modules. It intentionally does
//! not depend on Candle.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use cutedsl_jit::{
    ArtifactCache, Error, PythonEnvironment, Result, default_cache_root, prepare_python_environment,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

mod bf16_state_decode;
mod bf16_state_mtp;

pub use bf16_state_decode::{
    Bf16StateDecodeKernel, Bf16StateDecodeKernelVariant, Bf16StateDecodeSpecialization,
    Bf16StateDecodeTensors,
};
pub use bf16_state_mtp::{
    Bf16StateMtpKernel, Bf16StateMtpKernelVariant, Bf16StateMtpSpecialization, Bf16StateMtpTensors,
};

/// Error type shared with the generic artifact runtime.
pub use cutedsl_jit::Error as JitError;
/// Official DLPack/TVM FFI raw layouts used by the typed sys entrypoint.
pub use cutedsl_jit::{DlDataType, DlDataTypeCode, DlDevice, DlDeviceType, DlTensor};

/// FlashInfer release whose GDN source and shim contracts this crate targets.
pub const FLASHINFER_VERSION: &str = env!("FLASHINFER_GDN_VERSION");

/// Git revision corresponding to [`FLASHINFER_VERSION`].
pub const FLASHINFER_GIT_REV: &str = env!("FLASHINFER_GDN_GIT_REV");

/// Returns the FlashInfer source tree selected by this crate's build script.
#[must_use]
pub fn source_root() -> &'static Path {
    Path::new(env!("FLASHINFER_GDN_SOURCE_ROOT"))
}

/// Shared compiler and artifact-cache context for all GDN kernel families.
///
/// A handle is inexpensive to borrow and should normally be created once for a
/// process, then reused to prepare every required specialization.
#[derive(Debug, Clone)]
pub struct GdnHandle {
    pub(crate) python: PathBuf,
    pub(crate) cache: ArtifactCache,
    pub(crate) toolchain: Value,
    pub(crate) flashinfer_root: PathBuf,
    pub(crate) timeout: Duration,
}

static PYTHON_ENVIRONMENT: OnceLock<PythonEnvironment> = OnceLock::new();

impl GdnHandle {
    /// Creates a handle backed by the process-wide CuTeDSL Python environment.
    ///
    /// The first call locates, validates, or installs the locked environment.
    /// Later calls reuse it without invoking the preparation helper.
    pub fn new() -> Result<Self> {
        Self::with_cache_root(default_cache_root()?)
    }

    /// Creates a handle with an explicit artifact-cache root.
    ///
    /// This does not change the process-wide Python environment cache; set
    /// `CUTEDSL_JIT_CACHE_DIR` when both caches should use an explicit root.
    pub fn with_cache_root(cache_root: impl Into<PathBuf>) -> Result<Self> {
        let environment = python_environment()?;
        Self::from_parts(
            environment.python().to_path_buf(),
            cache_root,
            environment.identity().clone(),
        )
    }

    fn from_parts(
        python: impl Into<PathBuf>,
        cache_root: impl Into<PathBuf>,
        toolchain: Value,
    ) -> Result<Self> {
        Ok(Self {
            python: absolute_path(python.into())?,
            cache: ArtifactCache::new(absolute_path(cache_root.into())?),
            toolchain,
            flashinfer_root: source_root().to_path_buf(),
            timeout: Duration::from_secs(15 * 60),
        })
    }

    /// Overrides the FlashInfer source tree, primarily for source development.
    #[must_use]
    pub fn flashinfer_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.flashinfer_root = root.into();
        self
    }

    /// Overrides the compiler-process timeout for all kernel families.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

fn python_environment() -> Result<&'static PythonEnvironment> {
    if let Some(environment) = PYTHON_ENVIRONMENT.get() {
        return Ok(environment);
    }
    let lock =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("shims/requirements/cu13-aarch64-py312.lock");
    let prepared = prepare_python_environment(lock)?;
    let _ = PYTHON_ENVIRONMENT.set(prepared);
    Ok(PYTHON_ENVIRONMENT
        .get()
        .expect("the Python environment was initialized above"))
}

/// Input element type compiled into a decode specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputDType {
    /// IEEE float16.
    Float16,
    /// Brain float16.
    Bfloat16,
}

/// `dt_bias` element type compiled into a decode specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DtBiasDType {
    /// Brain float16.
    Bfloat16,
    /// IEEE float32.
    Float32,
}

fn valid_gpu_architecture(architecture: &str) -> bool {
    let Some(suffix) = architecture.strip_prefix("sm_") else {
        return false;
    };
    let digit_count = suffix.bytes().take_while(u8::is_ascii_digit).count();
    digit_count >= 2
        && (digit_count == suffix.len()
            || (digit_count + 1 == suffix.len()
                && matches!(suffix.as_bytes()[digit_count], b'a' | b'f')))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = File::create(path).map_err(|error| {
        Error::InvalidInput(format!("failed to create {}: {error}", path.display()))
    })?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|error| {
        Error::InvalidInput(format!("failed to write {}: {error}", path.display()))
    })?;
    file.sync_all()
        .map_err(|error| Error::InvalidInput(format!("failed to sync {}: {error}", path.display())))
}

fn absolute_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }
    std::env::current_dir()
        .map(|current| current.join(path))
        .map_err(|error| {
            Error::InvalidInput(format!("failed to resolve current directory: {error}"))
        })
}

fn host_compiler_identity() -> Result<Value> {
    let program = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    let output = Command::new(&program)
        .arg("--version")
        .output()
        .map_err(|error| {
            Error::InvalidInput(format!(
                "failed to query host C compiler {:?}: {error}",
                program
            ))
        })?;
    if !output.status.success() {
        return Err(Error::InvalidInput(format!(
            "host C compiler {:?} --version exited with {}",
            program, output.status
        )));
    }
    Ok(json!({
        "program": program.to_string_lossy(),
        "version": String::from_utf8_lossy(&output.stdout),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_source_contains_bf16_decode() {
        assert_eq!(FLASHINFER_VERSION, "0.6.16.post2");
        assert!(
            source_root()
                .join("flashinfer/gdn_kernels/gdn_decode_bf16_state.py")
                .is_file()
        );
    }

    #[test]
    fn one_handle_keys_both_decode_families() {
        let handle = GdnHandle::from_parts(
            "/unavailable-python-is-valid-for-key-generation",
            "target/test-cutedsl-cache",
            json!({"environment_digest": "test"}),
        )
        .unwrap();
        let single = handle
            .bf16_state_decode_cache_key(&Bf16StateDecodeSpecialization::default())
            .unwrap();
        let mtp = handle
            .bf16_state_mtp_cache_key(&Bf16StateMtpSpecialization::default())
            .unwrap();

        assert_eq!(single.namespace, "flashinfer-gdn/decode-bf16-state-t1");
        assert_eq!(
            mtp.namespace,
            "flashinfer-gdn/decode-bf16-state-mtp-pool-scatter"
        );
        assert_ne!(single.digest().unwrap(), mtp.digest().unwrap());
    }
}
