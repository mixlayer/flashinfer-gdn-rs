use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cutedsl_jit::{
    Abi, Artifact, ArtifactCache, CacheKey, CompilerCommand, Error, Result, TvmFfiAny, TvmModule,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    DlTensor, DtBiasDType, InputDType, absolute_path, host_compiler_identity, read_json,
    source_root, valid_gpu_architecture, write_json,
};

/// FlashInfer's execution-class boundary for non-transposed decode.
pub const SMALL_BATCH_THRESHOLD: usize = 32;

/// Compile-time kernel selected by FlashInfer for a decode batch size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NontransposeDecodeBatchClass {
    /// Eight 128-thread blocks per state/head, used for batches below 32.
    Small,
    /// One 256-thread block per state/head, used for batches of 32 or more.
    Large,
}

impl NontransposeDecodeBatchClass {
    /// Returns the upstream kernel class for `batch`.
    #[must_use]
    pub const fn for_batch(batch: usize) -> Self {
        if batch < SMALL_BATCH_THRESHOLD {
            Self::Small
        } else {
            Self::Large
        }
    }

    /// Whether this class is the one selected by upstream for `batch`.
    #[must_use]
    pub const fn matches(self, batch: usize) -> bool {
        matches!(
            (self, batch < SMALL_BATCH_THRESHOLD),
            (Self::Small, true) | (Self::Large, false)
        )
    }
}

/// Complete compile-time request for float-state non-transposed decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NontransposeDecodeSpecialization {
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
    /// Tokens per call; decode requires one.
    pub t: usize,
    /// Query scale.
    pub scale: f32,
    /// Whether Q/K L2 normalization is fused into the kernel.
    pub use_qk_l2norm: bool,
    /// Small- or large-batch upstream implementation.
    pub batch_class: NontransposeDecodeBatchClass,
}

impl NontransposeDecodeSpecialization {
    /// Creates the standard BF16-input, float-state specialization for `batch`.
    pub fn new(
        gpu_arch: impl Into<String>,
        h: usize,
        hv: usize,
        k: usize,
        v: usize,
        batch: usize,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(Error::InvalidInput(
                "non-transposed decode batch size must be positive".into(),
            ));
        }
        let specialization = Self {
            schema_version: 1,
            kernel: "gdn_decode_nontranspose".into(),
            symbol: "flashinfer_gdn_decode_nontranspose_f32_state".into(),
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
            batch_class: NontransposeDecodeBatchClass::for_batch(batch),
        };
        specialization.validate()?;
        Ok(specialization)
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
            || self.kernel != "gdn_decode_nontranspose"
            || self.symbol != "flashinfer_gdn_decode_nontranspose_f32_state"
        {
            return Err(Error::InvalidInput(
                "unsupported non-transposed-decode request identity".into(),
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
                "non-transposed decode requires T=1, found {}",
                self.t
            )));
        }
        if self.k != 128 {
            return Err(Error::InvalidInput(format!(
                "non-transposed decode requires K=128, found {}",
                self.k
            )));
        }
        let valid_v = match self.batch_class {
            NontransposeDecodeBatchClass::Small => self.v >= 128 && self.v.is_multiple_of(128),
            NontransposeDecodeBatchClass::Large => self.v >= 32 && self.v.is_multiple_of(32),
        };
        if !valid_v {
            let requirement = match self.batch_class {
                NontransposeDecodeBatchClass::Small => "V>=128 and divisible by 128",
                NontransposeDecodeBatchClass::Large => "V>=32 and divisible by 32",
            };
            return Err(Error::InvalidInput(format!(
                "the {:?} non-transposed decode kernel requires {requirement}; found V={}",
                self.batch_class, self.v
            )));
        }
        if [self.h, self.hv, self.k, self.v, self.t]
            .into_iter()
            .any(|dimension| i64::try_from(dimension).is_err())
        {
            return Err(Error::InvalidInput(
                "non-transposed decode dimensions must fit signed 64-bit DLPack shapes".into(),
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

impl Default for NontransposeDecodeSpecialization {
    fn default() -> Self {
        Self::new("sm_121a", 16, 16, 128, 128, 1)
            .expect("the built-in non-transposed-decode specialization is valid")
    }
}

/// Compiler adapter for float-state non-transposed decode.
#[derive(Debug, Clone)]
pub struct NontransposeDecodeCompiler {
    python: PathBuf,
    cache: ArtifactCache,
    toolchain: Value,
    flashinfer_root: PathBuf,
    timeout: Duration,
    specialization: NontransposeDecodeSpecialization,
}

impl NontransposeDecodeCompiler {
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
            specialization: NontransposeDecodeSpecialization::default(),
        })
    }

    /// Constructs an adapter from an environment created by `prepare_environment.py`.
    pub fn from_managed_python(
        python: impl Into<PathBuf>,
        cache_root: impl Into<PathBuf>,
    ) -> Result<Self> {
        let python = absolute_path(python.into())?;
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
        let toolchain = json!({
            "environment_schema_version": required("schema_version")?,
            "environment_digest": required("environment_digest")?,
            "python": required("python")?,
            "packages": required("packages")?,
        });
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
        specialization: NontransposeDecodeSpecialization,
    ) -> Result<Self> {
        specialization.validate()?;
        self.specialization = specialization;
        Ok(self)
    }

    /// Selected specialization.
    #[must_use]
    pub fn selected_specialization(&self) -> &NontransposeDecodeSpecialization {
        &self.specialization
    }

    /// Returns the full content-addressed key without compiling.
    pub fn cache_key(&self) -> Result<CacheKey> {
        let paths = compiler_paths();
        self.specialization.validate()?;
        let request = serde_json::to_value(&self.specialization).map_err(|error| {
            Error::InvalidInput(format!(
                "failed to serialize non-transposed-decode request: {error}"
            ))
        })?;
        let toolchain = json!({
            "python_environment": self.toolchain.clone(),
            "host_c_compiler": host_compiler_identity()?,
        });
        CacheKey::new(
            "flashinfer-gdn/decode-nontranspose-f32-state",
            Abi::TvmFfi,
            request,
            toolchain,
        )?
        .with_input_file("compiler-shim", &paths.shim)?
        .with_input_file("requirements-lock", &paths.requirements_lock)?
        .with_input_file(
            "flashinfer-kernel-source",
            self.flashinfer_root
                .join("flashinfer/gdn_kernels/gdn_decode_nontranspose.py"),
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
    pub fn load(&self) -> Result<NontransposeDecodeKernel> {
        let artifact = self.prepare()?;
        let module = TvmModule::load(artifact)?;
        Ok(NontransposeDecodeKernel {
            module: Arc::new(module),
            specialization: self.specialization.clone(),
        })
    }
}

#[derive(Debug)]
struct CompilerPaths {
    shim: PathBuf,
    requirements_lock: PathBuf,
}

fn compiler_paths() -> CompilerPaths {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shims = manifest.join("shims");
    CompilerPaths {
        shim: shims.join("compile_nontranspose_decode.py"),
        requirements_lock: shims.join("requirements/cu13-aarch64-py312.lock"),
    }
}

/// The eleven tensor arguments of the generated non-transposed decode entrypoint.
#[derive(Debug)]
pub struct NontransposeDecodeTensors<'a> {
    /// Reserved sequence offsets `[B+1]` for the non-varlen decode entrypoint.
    pub cu_seqlens: &'a mut DlTensor,
    /// Query `[B,1,H,K]`.
    pub q: &'a mut DlTensor,
    /// Key `[B,1,H,K]`.
    pub k: &'a mut DlTensor,
    /// Value `[B,1,HV,V]`.
    pub v: &'a mut DlTensor,
    /// Input-dependent decay `[B,1,HV]`.
    pub a: &'a mut DlTensor,
    /// Update gate `[B,1,HV]`.
    pub beta: &'a mut DlTensor,
    /// Log-decay parameter `[HV]`.
    pub a_log: &'a mut DlTensor,
    /// Decay bias `[HV]`.
    pub dt_bias: &'a mut DlTensor,
    /// Main state pool flattened to `[P*HV,K,V]`.
    pub state: &'a mut DlTensor,
    /// Identity state indices `[B]`.
    pub state_indices: &'a mut DlTensor,
    /// BF16 output `[B,1,HV,V]`.
    pub output: &'a mut DlTensor,
}

/// Loaded non-transposed float-state decode specialization.
#[derive(Debug, Clone)]
pub struct NontransposeDecodeKernel {
    module: Arc<TvmModule>,
    specialization: NontransposeDecodeSpecialization,
}

impl NontransposeDecodeKernel {
    /// Specialization enforced by this generated entrypoint.
    #[must_use]
    pub fn specialization(&self) -> &NontransposeDecodeSpecialization {
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
        tensors: &mut NontransposeDecodeTensors<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        let arguments = [
            TvmFfiAny::tensor(tensors.cu_seqlens),
            TvmFfiAny::tensor(tensors.q),
            TvmFfiAny::tensor(tensors.k),
            TvmFfiAny::tensor(tensors.v),
            TvmFfiAny::tensor(tensors.a),
            TvmFfiAny::tensor(tensors.beta),
            TvmFfiAny::tensor(tensors.a_log),
            TvmFfiAny::tensor(tensors.dt_bias),
            TvmFfiAny::tensor(tensors.state),
            TvmFfiAny::tensor(tensors.state_indices),
            TvmFfiAny::tensor(tensors.output),
            TvmFfiAny::opaque(stream),
        ];
        // SAFETY: the caller upholds the descriptor and stream contracts above.
        unsafe { self.module.call(&arguments) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_boundary_selects_distinct_classes() {
        assert_eq!(
            NontransposeDecodeBatchClass::for_batch(31),
            NontransposeDecodeBatchClass::Small
        );
        assert_eq!(
            NontransposeDecodeBatchClass::for_batch(32),
            NontransposeDecodeBatchClass::Large
        );
        assert!(NontransposeDecodeBatchClass::Small.matches(31));
        assert!(!NontransposeDecodeBatchClass::Small.matches(32));
    }

    #[test]
    fn validates_class_specific_value_tiles() {
        assert!(NontransposeDecodeSpecialization::new("sm_90a", 16, 16, 128, 128, 1).is_ok());
        assert!(NontransposeDecodeSpecialization::new("sm_90a", 16, 16, 128, 64, 32).is_ok());
        assert!(NontransposeDecodeSpecialization::new("sm_90a", 16, 16, 128, 64, 1).is_err());
    }
}
