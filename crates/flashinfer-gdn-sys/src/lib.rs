#![deny(unsafe_op_in_unsafe_fn)]
//! Raw FlashInfer GDN specialization and entrypoint integration.
//!
//! This crate owns the versioned CuTeDSL shims and the unsafe boundary between
//! GDN-specific argument schemas and `cutedsl-jit` modules. It intentionally does
//! not depend on Candle.

use std::ffi::c_void;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use cutedsl_jit::{
    Abi, Artifact, ArtifactCache, CacheKey, CompilerCommand, Error, Result, TvmFfiAny, TvmModule,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

mod bf16_state_decode;
mod bf16_state_mtp;
mod nontranspose_decode;

pub use bf16_state_decode::{
    Bf16StateDecodeCompiler, Bf16StateDecodeKernel, Bf16StateDecodeKernelVariant,
    Bf16StateDecodeSpecialization, Bf16StateDecodeTensors,
};
pub use bf16_state_mtp::{
    Bf16StateMtpCompiler, Bf16StateMtpKernel, Bf16StateMtpKernelVariant,
    Bf16StateMtpSpecialization, Bf16StateMtpTensors,
};
pub use nontranspose_decode::{
    NontransposeDecodeBatchClass, NontransposeDecodeCompiler, NontransposeDecodeKernel,
    NontransposeDecodeSpecialization, NontransposeDecodeTensors,
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

/// Complete compile-time request for the float-state pretransposed decode kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretransposeDecodeSpecialization {
    /// Request schema version.
    pub schema_version: u32,
    /// Versioned kernel family name.
    pub kernel: String,
    /// Unprefixed symbol passed to CuTeDSL export.
    pub symbol: String,
    /// CuTeDSL GPU architecture, such as `sm_90a` or `sm_121a`.
    pub gpu_arch: String,
    /// Q/K/V/gate input type.
    pub io_dtype: InputDType,
    /// Decay-bias input type.
    pub dt_bias_dtype: DtBiasDType,
    /// Query heads.
    pub h: usize,
    /// Value/state heads.
    pub hv: usize,
    /// Query/key dimension.
    pub k: usize,
    /// Value dimension.
    pub v: usize,
    /// Tokens per call; pretransposed decode requires one.
    pub t: usize,
    /// Query scale.
    pub scale: f32,
    /// Whether Q/K L2 normalization is fused into the kernel.
    pub use_qk_l2norm: bool,
    /// Whether state is selected through read/write pool indices.
    pub use_pool_indexing: bool,
}

impl PretransposeDecodeSpecialization {
    /// Creates the standard BF16-input, float-state specialization.
    pub fn new(
        gpu_arch: impl Into<String>,
        h: usize,
        hv: usize,
        k: usize,
        v: usize,
    ) -> Result<Self> {
        let specialization = Self {
            schema_version: 1,
            kernel: "gdn_decode_pretranspose".into(),
            symbol: "flashinfer_gdn_decode_pretranspose_f32_state".into(),
            gpu_arch: gpu_arch.into(),
            io_dtype: InputDType::Bfloat16,
            dt_bias_dtype: DtBiasDType::Float32,
            h,
            hv,
            k,
            v,
            t: 1,
            scale: (k as f32).sqrt().recip(),
            use_qk_l2norm: true,
            use_pool_indexing: false,
        };
        specialization.validate()?;
        Ok(specialization)
    }

    /// Selects direct-state or indexed state-pool code generation.
    #[must_use]
    pub fn pool_indexing(mut self, enabled: bool) -> Self {
        self.use_pool_indexing = enabled;
        self
    }

    /// Selects the Q/K/V/gate input type.
    #[must_use]
    pub fn input_dtype(mut self, dtype: InputDType) -> Self {
        self.io_dtype = dtype;
        self
    }

    /// Selects the decay-bias type.
    #[must_use]
    pub fn dt_bias_dtype(mut self, dtype: DtBiasDType) -> Self {
        self.dt_bias_dtype = dtype;
        self
    }

    /// Overrides the query scale.
    #[must_use]
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enables or disables fused Q/K L2 normalization.
    #[must_use]
    pub fn qk_l2norm(mut self, enabled: bool) -> Self {
        self.use_qk_l2norm = enabled;
        self
    }

    /// Validates constraints imposed by the pinned upstream kernel.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.kernel != "gdn_decode_pretranspose"
            || self.symbol != "flashinfer_gdn_decode_pretranspose_f32_state"
        {
            return Err(Error::InvalidInput(
                "unsupported pretransposed-decode request identity".into(),
            ));
        }
        if !valid_gpu_architecture(&self.gpu_arch) {
            return Err(Error::InvalidInput(format!(
                "invalid CuTeDSL GPU architecture {:?}",
                self.gpu_arch
            )));
        }
        if self.h == 0 || self.hv == 0 || self.hv < self.h || !self.hv.is_multiple_of(self.h) {
            return Err(Error::InvalidInput(format!(
                "HV ({}) must be a positive multiple of H ({})",
                self.hv, self.h
            )));
        }
        if self.t != 1 {
            return Err(Error::InvalidInput(format!(
                "pretransposed decode requires T=1, found {}",
                self.t
            )));
        }
        if self.k != 128 || self.v < 128 || !self.v.is_multiple_of(64) {
            return Err(Error::InvalidInput(format!(
                "the small-batch pretransposed decode shim requires K=128, V>=128, and V divisible by 64; found K={}, V={}",
                self.k, self.v
            )));
        }
        if [self.h, self.hv, self.k, self.v, self.t]
            .into_iter()
            .any(|dimension| i64::try_from(dimension).is_err())
        {
            return Err(Error::InvalidInput(
                "pretransposed decode dimensions must fit signed 64-bit DLPack shapes".into(),
            ));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "query scale must be positive and finite, found {}",
                self.scale
            )));
        }
        Ok(())
    }
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

impl Default for PretransposeDecodeSpecialization {
    fn default() -> Self {
        Self::new("sm_121a", 16, 16, 128, 128)
            .expect("the built-in pretransposed-decode specialization is valid")
    }
}

/// Compiler adapter for the first pretransposed float-state decode specialization.
///
/// This is intentionally a narrow vertical slice. The generic cache and loader live
/// in cutedsl-jit; this type owns the FlashInfer-specific source paths and worker
/// arguments.
#[derive(Debug, Clone)]
pub struct PretransposeDecodeCompiler {
    python: PathBuf,
    cache: ArtifactCache,
    toolchain: Value,
    flashinfer_root: PathBuf,
    timeout: Duration,
    specialization: PretransposeDecodeSpecialization,
}

impl PretransposeDecodeCompiler {
    /// Constructs an adapter with an explicit, path-independent toolchain identity.
    pub fn new(
        python: impl Into<PathBuf>,
        cache_root: impl Into<PathBuf>,
        toolchain: Value,
    ) -> Result<Self> {
        let python = absolute_path(python.into())?;
        Ok(Self {
            python,
            cache: ArtifactCache::new(absolute_path(cache_root.into())?),
            toolchain,
            flashinfer_root: source_root().to_path_buf(),
            timeout: Duration::from_secs(15 * 60),
            specialization: PretransposeDecodeSpecialization::default(),
        })
    }

    /// Constructs an adapter from an environment created by prepare_environment.py.
    pub fn from_managed_python(
        python: impl Into<PathBuf>,
        cache_root: impl Into<PathBuf>,
    ) -> Result<Self> {
        let (python, toolchain) = managed_python_toolchain(python.into())?;
        Self::new(python, cache_root, toolchain)
    }

    /// Overrides the FlashInfer tree, primarily for source development.
    #[must_use]
    pub fn flashinfer_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.flashinfer_root = root.into();
        self
    }

    /// Overrides the compiler-process timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Selects the exact kernel specialization to prepare.
    pub fn specialization(
        mut self,
        specialization: PretransposeDecodeSpecialization,
    ) -> Result<Self> {
        specialization.validate()?;
        self.specialization = specialization;
        Ok(self)
    }

    /// Selected specialization.
    #[must_use]
    pub fn selected_specialization(&self) -> &PretransposeDecodeSpecialization {
        &self.specialization
    }

    /// Returns the full content-addressed key without compiling.
    pub fn cache_key(&self) -> Result<CacheKey> {
        let paths = compiler_paths();
        self.specialization.validate()?;
        let request = serde_json::to_value(&self.specialization).map_err(|error| {
            Error::InvalidInput(format!(
                "failed to serialize pretransposed-decode request: {error}"
            ))
        })?;
        let toolchain = json!({
            "python_environment": self.toolchain.clone(),
            "host_c_compiler": host_compiler_identity()?,
        });
        CacheKey::new(
            "flashinfer-gdn/decode-pretranspose-f32-state",
            Abi::TvmFfi,
            request,
            toolchain,
        )?
        .with_input_file("compiler-shim", &paths.shim)?
        .with_input_file("compiler-support", &paths.support)?
        .with_input_file("requirements-lock", &paths.requirements_lock)?
        .with_input_file(
            "flashinfer-kernel-source",
            self.flashinfer_root
                .join("flashinfer/gdn_kernels/gdn_decode_pretranspose.py"),
        )
    }

    /// Returns a validated cache hit or compiles and atomically publishes a miss.
    pub fn prepare(&self) -> Result<Artifact> {
        let paths = compiler_paths();
        let key = self.cache_key()?;
        let request = self.specialization.clone();
        self.cache.get_or_build(&key, |directory| {
            let request_path = directory.join("request.json");
            write_json(&request_path, &request)?;
            CompilerCommand::new(&self.python)
                .arg(&paths.shim)
                .arg("--request")
                .arg(&request_path)
                .arg("--flashinfer-root")
                .arg(&self.flashinfer_root)
                .arg("--output-dir")
                .arg(directory)
                .timeout(self.timeout)
                .run(directory)
        })
    }

    /// Explicitly prepares and dynamically loads the selected specialization.
    pub fn load(&self) -> Result<PretransposeDecodeKernel> {
        let artifact = self.prepare()?;
        let module = TvmModule::load(artifact)?;
        Ok(PretransposeDecodeKernel {
            module: Arc::new(module),
            specialization: self.specialization.clone(),
        })
    }
}

#[derive(Debug)]
struct CompilerPaths {
    shim: PathBuf,
    support: PathBuf,
    requirements_lock: PathBuf,
}

fn compiler_paths() -> CompilerPaths {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shims = manifest.join("shims");
    CompilerPaths {
        shim: shims.join("compile_pretranspose_decode.py"),
        support: shims.join("_artifact.py"),
        requirements_lock: shims.join("requirements/cu13-aarch64-py312.lock"),
    }
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

/// The thirteen tensor/stream arguments of the generated pretransposed decode entrypoint.
#[derive(Debug)]
pub struct PretransposeDecodeTensors<'a> {
    /// Direct state `[B*HV,V,K]` or pool state `[P,HV,V,K]`.
    pub state: &'a mut DlTensor,
    /// Log-decay parameter `[HV]`.
    pub a_log: &'a mut DlTensor,
    /// Input-dependent decay `[B,1,HV]`.
    pub a: &'a mut DlTensor,
    /// Decay bias `[HV]`.
    pub dt_bias: &'a mut DlTensor,
    /// Query `[B,1,H,K]`.
    pub q: &'a mut DlTensor,
    /// Key `[B,1,H,K]`.
    pub k: &'a mut DlTensor,
    /// Value `[B,1,HV,V]`.
    pub v: &'a mut DlTensor,
    /// Update gate `[B,1,HV]`.
    pub beta: &'a mut DlTensor,
    /// BF16 output `[B,1,HV,V]`.
    pub output: &'a mut DlTensor,
    /// State-pool read indices `[B]`.
    pub state_indices: &'a mut DlTensor,
    /// State-pool write indices `[B]`.
    pub output_state_indices: &'a mut DlTensor,
    /// Reserved sequence offsets `[B+1]` for the non-varlen decode entrypoint.
    pub cu_seqlens: &'a mut DlTensor,
}

/// Loaded pretransposed float-state decode specialization.
#[derive(Debug, Clone)]
pub struct PretransposeDecodeKernel {
    module: Arc<TvmModule>,
    specialization: PretransposeDecodeSpecialization,
}

impl PretransposeDecodeKernel {
    /// Specialization enforced by this generated entrypoint.
    #[must_use]
    pub fn specialization(&self) -> &PretransposeDecodeSpecialization {
        &self.specialization
    }

    /// Keeps the compiled module alive and exposes its artifact metadata.
    #[must_use]
    pub fn module(&self) -> &Arc<TvmModule> {
        &self.module
    }

    /// Launches the generated TVM FFI entrypoint on `stream`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every descriptor points to live CUDA storage
    /// matching its shape, strides, dtype, device, and mutability contract until
    /// all work enqueued on `stream` has completed.
    pub unsafe fn launch(
        &self,
        tensors: &mut PretransposeDecodeTensors<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        let arguments = [
            TvmFfiAny::tensor(tensors.state),
            TvmFfiAny::tensor(tensors.a_log),
            TvmFfiAny::tensor(tensors.a),
            TvmFfiAny::tensor(tensors.dt_bias),
            TvmFfiAny::tensor(tensors.q),
            TvmFfiAny::tensor(tensors.k),
            TvmFfiAny::tensor(tensors.v),
            TvmFfiAny::tensor(tensors.beta),
            TvmFfiAny::tensor(tensors.output),
            TvmFfiAny::tensor(tensors.state_indices),
            TvmFfiAny::tensor(tensors.output_state_indices),
            TvmFfiAny::tensor(tensors.cu_seqlens),
            TvmFfiAny::opaque(stream),
        ];
        // SAFETY: the caller upholds the descriptor and stream contracts above.
        unsafe { self.module.call(&arguments) }
    }
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

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| {
        Error::InvalidInput(format!(
            "failed to read JSON file {}: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        Error::InvalidInput(format!(
            "failed to parse JSON file {}: {error}",
            path.display()
        ))
    })
}

fn managed_python_toolchain(python: PathBuf) -> Result<(PathBuf, Value)> {
    let python = absolute_path(python)?;
    let environment = python
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| Error::InvalidInput("compiler Python has no environment root".into()))?;
    let marker_path = environment.join("environment.json");
    let marker = read_json(&marker_path)?;
    let marker = marker.as_object().ok_or_else(|| {
        Error::InvalidInput(format!(
            "managed environment marker is not an object: {}",
            marker_path.display()
        ))
    })?;
    let required = |name: &str| {
        marker.get(name).cloned().ok_or_else(|| {
            Error::InvalidInput(format!(
                "managed environment marker {} has no {name:?} field",
                marker_path.display()
            ))
        })
    };
    // Absolute lock and runtime-library paths are deliberately excluded. The
    // environment digest and exact package set identify the compiler without
    // making equivalent environments in different cache roots miss.
    let toolchain = json!({
        "environment_schema_version": required("schema_version")?,
        "environment_digest": required("environment_digest")?,
        "python": required("python")?,
        "packages": required("packages")?,
    });
    Ok((python, toolchain))
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
    fn selected_source_contains_pretranspose_decode() {
        assert_eq!(FLASHINFER_VERSION, "0.6.16.post2");
        assert!(
            source_root()
                .join("flashinfer/gdn_kernels/gdn_decode_pretranspose.py")
                .is_file()
        );
    }

    #[test]
    fn pretranspose_cache_key_covers_all_current_inputs() {
        let compiler = PretransposeDecodeCompiler::new(
            "/unavailable-python-is-valid-for-a-cache-hit",
            "target/test-cutedsl-cache",
            json!({"environment_digest": "test"}),
        )
        .unwrap();
        let key = compiler.cache_key().unwrap();
        assert_eq!(key.abi, Abi::TvmFfi);
        assert_eq!(key.inputs.len(), 4);
        for name in [
            "compiler-shim",
            "compiler-support",
            "requirements-lock",
            "flashinfer-kernel-source",
        ] {
            assert!(key.inputs.contains_key(name), "missing {name}");
        }
        assert_eq!(key.request["gpu_arch"], "sm_121a");
        assert!(key.toolchain.get("host_c_compiler").is_some());
    }

    #[test]
    fn pool_indexing_is_a_distinct_specialization() {
        let direct = PretransposeDecodeSpecialization::default();
        let pool = direct.clone().pool_indexing(true);
        assert_ne!(direct, pool);
        assert!(!direct.use_pool_indexing);
        assert!(pool.use_pool_indexing);
    }

    #[test]
    fn specialization_rejects_incompatible_head_counts() {
        assert!(PretransposeDecodeSpecialization::new("sm_90a", 16, 24, 128, 128).is_err());
    }
}
