use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

use cutedsl_jit::{Abi, Artifact, CacheKey, CompilerCommand, Error, Result, TvmFfiAny, TvmModule};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    DlTensor, GdnHandle, InputDType, host_compiler_identity, runtime_asset_root,
    valid_gpu_architecture, write_json,
};

/// Architecture-specific non-context-parallel GDN prefill implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefillBackend {
    /// Hopper SM90 implementation with compact float32 state.
    Sm90,
    /// Blackwell SM100/SM103 implementation with native indexed BF16 state.
    Sm100,
    /// SM120/SM121 implementation with compact float32 state.
    Sm120,
}

impl PrefillBackend {
    fn identity(self) -> (&'static str, &'static str) {
        match self {
            Self::Sm90 => ("gdn_prefill_sm90", "flashinfer_gdn_prefill_sm90"),
            Self::Sm100 => ("gdn_prefill_sm100", "flashinfer_gdn_prefill_sm100"),
            Self::Sm120 => ("gdn_prefill_sm120", "flashinfer_gdn_prefill_sm120"),
        }
    }

    fn namespace(self) -> &'static str {
        match self {
            Self::Sm90 => "flashinfer-gdn/prefill-sm90",
            Self::Sm100 => "flashinfer-gdn/prefill-sm100-indexed-bf16-state",
            Self::Sm120 => "flashinfer-gdn/prefill-sm120",
        }
    }
}

/// Complete compile-time request for a non-CP GDN prefill kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefillSpecialization {
    /// Request schema version.
    pub schema_version: u32,
    /// Versioned kernel family name.
    pub kernel: String,
    /// Unprefixed TVM FFI symbol.
    pub symbol: String,
    /// Architecture-specific backend.
    pub backend: PrefillBackend,
    /// CuTeDSL target, such as `sm_90a`, `sm_100a`, or `sm_121a`.
    pub gpu_arch: String,
    /// Q/K/V/output element type; BF16 in this initial slice.
    pub io_dtype: InputDType,
    /// Kernel-facing state type (`float32` or `bfloat16`).
    pub state_dtype: String,
    /// Query/key heads.
    pub h: usize,
    /// Value/state heads.
    pub hv: usize,
    /// Query/key dimension.
    pub k: usize,
    /// Value dimension.
    pub v: usize,
    /// Query scale.
    pub scale: f32,
    /// Device multiprocessor count compiled into persistent scheduling/workspace sizing.
    pub num_sms: usize,
    /// Whether the generated kernel accepts pool indices directly.
    pub use_state_indices: bool,
    /// Compile-time checkpoint interval. Zero disables checkpoint emission.
    pub checkpoint_every_n_tokens: usize,
}

impl PrefillSpecialization {
    /// Creates an architecture-specific prefill specialization.
    pub fn new(
        backend: PrefillBackend,
        gpu_arch: impl Into<String>,
        h: usize,
        hv: usize,
        k: usize,
        v: usize,
        num_sms: usize,
    ) -> Result<Self> {
        let (kernel, symbol) = backend.identity();
        let specialization = Self {
            schema_version: 2,
            kernel: kernel.into(),
            symbol: symbol.into(),
            backend,
            gpu_arch: gpu_arch.into(),
            io_dtype: InputDType::Bfloat16,
            state_dtype: if backend == PrefillBackend::Sm100 {
                "bfloat16".into()
            } else {
                "float32".into()
            },
            h,
            hv,
            k,
            v,
            scale: (k as f32).sqrt().recip(),
            num_sms,
            use_state_indices: backend == PrefillBackend::Sm100,
            checkpoint_every_n_tokens: 0,
        };
        specialization.validate()?;
        Ok(specialization)
    }

    /// Overrides the query scale.
    #[must_use]
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enables compact state checkpoint emission at the given token interval.
    #[must_use]
    pub fn checkpoint_every_n_tokens(mut self, interval: usize) -> Self {
        self.checkpoint_every_n_tokens = interval;
        self
    }

    /// Whether this specialization emits compact checkpoint rows.
    #[must_use]
    pub const fn checkpoints_enabled(&self) -> bool {
        self.checkpoint_every_n_tokens > 0
    }

    /// Validates the pinned upstream prefill contract.
    pub fn validate(&self) -> Result<()> {
        let identity = self.backend.identity();
        if self.schema_version != 2 || self.kernel != identity.0 || self.symbol != identity.1 {
            return Err(Error::InvalidInput(
                "unsupported GDN prefill request identity".into(),
            ));
        }
        if !valid_gpu_architecture(&self.gpu_arch) {
            return Err(Error::InvalidInput(format!(
                "invalid CuTeDSL GPU architecture {:?}",
                self.gpu_arch
            )));
        }
        let numeric = architecture_number(&self.gpu_arch)?;
        let arch_matches = match self.backend {
            PrefillBackend::Sm90 => numeric == 90,
            PrefillBackend::Sm100 => matches!(numeric, 100 | 103),
            PrefillBackend::Sm120 => matches!(numeric, 120 | 121),
        };
        if !arch_matches {
            return Err(Error::InvalidInput(format!(
                "{:?} prefill does not support {}",
                self.backend, self.gpu_arch
            )));
        }
        if self.io_dtype != InputDType::Bfloat16 {
            return Err(Error::InvalidInput("GDN prefill requires BF16 I/O".into()));
        }
        let expected_state = if self.backend == PrefillBackend::Sm100 {
            "bfloat16"
        } else {
            "float32"
        };
        if self.state_dtype != expected_state
            || self.use_state_indices != (self.backend == PrefillBackend::Sm100)
        {
            return Err(Error::InvalidInput(format!(
                "{:?} prefill requires {expected_state} state and use_state_indices={}",
                self.backend,
                self.backend == PrefillBackend::Sm100
            )));
        }
        if self.h == 0 || self.hv == 0 || self.hv < self.h || !self.hv.is_multiple_of(self.h) {
            return Err(Error::InvalidInput(format!(
                "HV ({}) must be a positive multiple of H ({})",
                self.hv, self.h
            )));
        }
        if self.k != 128 || self.v != 128 {
            return Err(Error::InvalidInput(format!(
                "GDN prefill requires K=V=128; found K={}, V={}",
                self.k, self.v
            )));
        }
        if self.num_sms == 0 {
            return Err(Error::InvalidInput("num_sms must be positive".into()));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "query scale must be positive and finite, found {}",
                self.scale
            )));
        }
        if self.checkpoint_every_n_tokens > 0 && !self.checkpoint_every_n_tokens.is_multiple_of(64)
        {
            return Err(Error::InvalidInput(format!(
                "checkpoint interval must be zero or a multiple of 64, found {}",
                self.checkpoint_every_n_tokens
            )));
        }
        Ok(())
    }
}

fn architecture_number(architecture: &str) -> Result<u32> {
    let digits: String = architecture
        .trim_start_matches("sm_")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().map_err(|error| {
        Error::InvalidInput(format!(
            "failed to parse GPU architecture {architecture:?}: {error}"
        ))
    })
}

impl GdnHandle {
    /// Returns the content-addressed key for an SM90 prefill specialization.
    pub fn prefill_sm90_cache_key(&self, spec: &PrefillSpecialization) -> Result<CacheKey> {
        self.prefill_cache_key(spec, PrefillBackend::Sm90)
    }

    /// Returns the content-addressed key for an SM100/SM103 prefill specialization.
    pub fn prefill_sm100_cache_key(&self, spec: &PrefillSpecialization) -> Result<CacheKey> {
        self.prefill_cache_key(spec, PrefillBackend::Sm100)
    }

    /// Returns the content-addressed key for an SM120/SM121 prefill specialization.
    pub fn prefill_sm120_cache_key(&self, spec: &PrefillSpecialization) -> Result<CacheKey> {
        self.prefill_cache_key(spec, PrefillBackend::Sm120)
    }

    fn prefill_cache_key(
        &self,
        spec: &PrefillSpecialization,
        backend: PrefillBackend,
    ) -> Result<CacheKey> {
        validate_backend(spec, backend)?;
        let paths = compiler_paths();
        let request = serde_json::to_value(spec).map_err(|error| {
            Error::InvalidInput(format!("failed to serialize prefill request: {error}"))
        })?;
        let toolchain = json!({
            "python_environment": self.toolchain.clone(),
            "host_c_compiler": host_compiler_identity()?,
        });
        let mut key = CacheKey::new(backend.namespace(), Abi::TvmFfi, request, toolchain)?
            .with_input_file("compiler-shim", &paths.shim)?
            .with_input_file("compiler-support", &paths.support)?
            .with_input_file("requirements-lock", &paths.requirements_lock)?;
        for (label, relative) in source_inputs(backend) {
            key = key.with_input_file(label, self.flashinfer_root.join(relative))?;
        }
        Ok(key)
    }

    fn prepare_prefill(
        &self,
        spec: &PrefillSpecialization,
        backend: PrefillBackend,
    ) -> Result<Artifact> {
        let paths = compiler_paths();
        let key = self.prefill_cache_key(spec, backend)?;
        let request = spec.clone();
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

    /// Compiles or loads an SM90 prefill module.
    pub fn load_prefill_sm90(&self, spec: &PrefillSpecialization) -> Result<PrefillSm90Kernel> {
        let module = TvmModule::load(self.prepare_prefill(spec, PrefillBackend::Sm90)?)?;
        Ok(PrefillSm90Kernel::new(module, spec.clone()))
    }

    /// Compiles or loads an SM100/SM103 prefill module.
    pub fn load_prefill_sm100(&self, spec: &PrefillSpecialization) -> Result<PrefillSm100Kernel> {
        let module = TvmModule::load(self.prepare_prefill(spec, PrefillBackend::Sm100)?)?;
        Ok(PrefillSm100Kernel::new(module, spec.clone()))
    }

    /// Compiles or loads an SM120/SM121 prefill module.
    pub fn load_prefill_sm120(&self, spec: &PrefillSpecialization) -> Result<PrefillSm120Kernel> {
        let module = TvmModule::load(self.prepare_prefill(spec, PrefillBackend::Sm120)?)?;
        Ok(PrefillSm120Kernel::new(module, spec.clone()))
    }
}

fn validate_backend(spec: &PrefillSpecialization, backend: PrefillBackend) -> Result<()> {
    spec.validate()?;
    if spec.backend != backend {
        return Err(Error::InvalidInput(format!(
            "expected {backend:?} specialization, found {:?}",
            spec.backend
        )));
    }
    Ok(())
}

fn source_inputs(backend: PrefillBackend) -> Vec<(&'static str, &'static str)> {
    const DELTA_COMMON: &[(&str, &str)] = &[
        (
            "flashinfer-alpha",
            "flashinfer/gdn_kernels/delta_rule_dsl/alpha.py",
        ),
        (
            "flashinfer-inverse",
            "flashinfer/gdn_kernels/delta_rule_dsl/collective_inverse_hmma.py",
        ),
        (
            "flashinfer-store",
            "flashinfer/gdn_kernels/delta_rule_dsl/collective_store_tma.py",
        ),
        (
            "flashinfer-helpers",
            "flashinfer/gdn_kernels/delta_rule_dsl/helpers.py",
        ),
        (
            "flashinfer-schedule",
            "flashinfer/gdn_kernels/delta_rule_dsl/schedule.py",
        ),
    ];
    match backend {
        PrefillBackend::Sm90 => std::iter::once((
            "flashinfer-kernel-source",
            "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py",
        ))
        .chain(DELTA_COMMON.iter().copied())
        .collect(),
        PrefillBackend::Sm120 => std::iter::once((
            "flashinfer-kernel-source",
            "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py",
        ))
        .chain(DELTA_COMMON.iter().copied())
        .collect(),
        PrefillBackend::Sm100 => vec![
            (
                "flashinfer-kernel-source",
                "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py",
            ),
            (
                "flashinfer-tile-scheduler",
                "flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py",
            ),
        ],
    }
}

#[derive(Debug)]
struct CompilerPaths {
    shim: PathBuf,
    support: PathBuf,
    requirements_lock: PathBuf,
}

fn compiler_paths() -> CompilerPaths {
    let shims = runtime_asset_root().join("shims");
    CompilerPaths {
        shim: shims.join("compile_prefill.py"),
        support: shims.join("_artifact.py"),
        requirements_lock: shims.join("requirements/cu13-linux-py312.lock"),
    }
}

/// Tensor arguments for SM90 compact-float32-state prefill.
#[derive(Debug)]
pub struct PrefillSm90Tensors<'a> {
    pub q: &'a mut DlTensor,
    pub k: &'a mut DlTensor,
    pub v: &'a mut DlTensor,
    pub alpha: &'a mut DlTensor,
    pub beta: &'a mut DlTensor,
    pub initial_state: &'a mut DlTensor,
    pub output: &'a mut DlTensor,
    pub output_state: &'a mut DlTensor,
    pub cu_seqlens: &'a mut DlTensor,
    /// Compact float32 checkpoint rows, present only for checkpointed modules.
    pub state_checkpoints: Option<&'a mut DlTensor>,
    /// Per-sequence checkpoint row offsets, int64 `[B+1]`.
    pub checkpoint_cu_starts: Option<&'a mut DlTensor>,
    pub tensormaps: &'a mut DlTensor,
}

/// Tensor arguments for SM100/SM103 native indexed-state prefill.
#[derive(Debug)]
pub struct PrefillSm100Tensors<'a> {
    pub q: &'a mut DlTensor,
    pub k: &'a mut DlTensor,
    pub v: &'a mut DlTensor,
    pub alpha: &'a mut DlTensor,
    pub beta: &'a mut DlTensor,
    pub initial_state: &'a mut DlTensor,
    pub output: &'a mut DlTensor,
    pub output_state: &'a mut DlTensor,
    pub cu_seqlens: &'a mut DlTensor,
    pub state_indices: &'a mut DlTensor,
    /// Compact BF16 checkpoint rows, present only for checkpointed modules.
    pub state_checkpoints: Option<&'a mut DlTensor>,
    /// Per-sequence checkpoint row offsets, int32 `[B+1]`.
    pub checkpoint_cu_starts: Option<&'a mut DlTensor>,
    pub tensormaps: &'a mut DlTensor,
}

/// Tensor arguments for SM120/SM121 compact-float32-state prefill.
pub type PrefillSm120Tensors<'a> = PrefillSm90Tensors<'a>;

macro_rules! compact_kernel {
    ($name:ident, $backend:expr) => {
        /// Loaded architecture-specific compact-state prefill module.
        #[derive(Debug, Clone)]
        pub struct $name {
            module: Arc<TvmModule>,
            specialization: PrefillSpecialization,
        }

        impl $name {
            fn new(module: TvmModule, specialization: PrefillSpecialization) -> Self {
                Self {
                    module: Arc::new(module),
                    specialization,
                }
            }

            #[must_use]
            pub fn specialization(&self) -> &PrefillSpecialization {
                &self.specialization
            }

            #[must_use]
            pub fn module(&self) -> &Arc<TvmModule> {
                &self.module
            }

            /// Launches the generated TVM FFI entrypoint.
            ///
            /// # Safety
            ///
            /// Descriptors and stream must remain valid until queued work completes
            /// and must satisfy the specialization contract.
            pub unsafe fn launch(
                &self,
                tensors: &mut PrefillSm90Tensors<'_>,
                stream: *mut c_void,
            ) -> Result<()> {
                debug_assert_eq!(self.specialization.backend, $backend);
                if self.specialization.checkpoints_enabled() {
                    let state_checkpoints =
                        tensors.state_checkpoints.as_deref_mut().ok_or_else(|| {
                            Error::InvalidInput(
                                "checkpointed prefill requires state_checkpoints".into(),
                            )
                        })?;
                    let checkpoint_cu_starts =
                        tensors.checkpoint_cu_starts.as_deref_mut().ok_or_else(|| {
                            Error::InvalidInput(
                                "checkpointed prefill requires checkpoint_cu_starts".into(),
                            )
                        })?;
                    let arguments = [
                        TvmFfiAny::tensor(tensors.q),
                        TvmFfiAny::tensor(tensors.k),
                        TvmFfiAny::tensor(tensors.v),
                        TvmFfiAny::tensor(tensors.alpha),
                        TvmFfiAny::tensor(tensors.beta),
                        TvmFfiAny::tensor(tensors.initial_state),
                        TvmFfiAny::tensor(tensors.output),
                        TvmFfiAny::tensor(tensors.output_state),
                        TvmFfiAny::tensor(tensors.cu_seqlens),
                        TvmFfiAny::tensor(state_checkpoints),
                        TvmFfiAny::tensor(checkpoint_cu_starts),
                        TvmFfiAny::tensor(tensors.tensormaps),
                        TvmFfiAny::opaque(stream),
                    ];
                    // SAFETY: upheld by the caller.
                    unsafe { self.module.call(&arguments) }
                } else {
                    if tensors.state_checkpoints.is_some() || tensors.checkpoint_cu_starts.is_some()
                    {
                        return Err(Error::InvalidInput(
                            "non-checkpointed prefill does not accept checkpoint tensors".into(),
                        ));
                    }
                    let arguments = [
                        TvmFfiAny::tensor(tensors.q),
                        TvmFfiAny::tensor(tensors.k),
                        TvmFfiAny::tensor(tensors.v),
                        TvmFfiAny::tensor(tensors.alpha),
                        TvmFfiAny::tensor(tensors.beta),
                        TvmFfiAny::tensor(tensors.initial_state),
                        TvmFfiAny::tensor(tensors.output),
                        TvmFfiAny::tensor(tensors.output_state),
                        TvmFfiAny::tensor(tensors.cu_seqlens),
                        TvmFfiAny::tensor(tensors.tensormaps),
                        TvmFfiAny::opaque(stream),
                    ];
                    // SAFETY: upheld by the caller.
                    unsafe { self.module.call(&arguments) }
                }
            }
        }
    };
}

compact_kernel!(PrefillSm90Kernel, PrefillBackend::Sm90);
compact_kernel!(PrefillSm120Kernel, PrefillBackend::Sm120);

/// Loaded SM100/SM103 native indexed-BF16-state prefill module.
#[derive(Debug, Clone)]
pub struct PrefillSm100Kernel {
    module: Arc<TvmModule>,
    specialization: PrefillSpecialization,
}

impl PrefillSm100Kernel {
    fn new(module: TvmModule, specialization: PrefillSpecialization) -> Self {
        Self {
            module: Arc::new(module),
            specialization,
        }
    }

    #[must_use]
    pub fn specialization(&self) -> &PrefillSpecialization {
        &self.specialization
    }

    #[must_use]
    pub fn module(&self) -> &Arc<TvmModule> {
        &self.module
    }

    /// Launches the generated TVM FFI entrypoint.
    ///
    /// # Safety
    ///
    /// Descriptors and stream must remain valid until queued work completes
    /// and must satisfy the specialization contract.
    pub unsafe fn launch(
        &self,
        tensors: &mut PrefillSm100Tensors<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        if self.specialization.checkpoints_enabled() {
            let state_checkpoints = tensors.state_checkpoints.as_deref_mut().ok_or_else(|| {
                Error::InvalidInput("checkpointed prefill requires state_checkpoints".into())
            })?;
            let checkpoint_cu_starts =
                tensors.checkpoint_cu_starts.as_deref_mut().ok_or_else(|| {
                    Error::InvalidInput("checkpointed prefill requires checkpoint_cu_starts".into())
                })?;
            let arguments = [
                TvmFfiAny::tensor(tensors.q),
                TvmFfiAny::tensor(tensors.k),
                TvmFfiAny::tensor(tensors.v),
                TvmFfiAny::tensor(tensors.alpha),
                TvmFfiAny::tensor(tensors.beta),
                TvmFfiAny::tensor(tensors.initial_state),
                TvmFfiAny::tensor(tensors.output),
                TvmFfiAny::tensor(tensors.output_state),
                TvmFfiAny::tensor(tensors.cu_seqlens),
                TvmFfiAny::tensor(tensors.state_indices),
                TvmFfiAny::tensor(state_checkpoints),
                TvmFfiAny::tensor(checkpoint_cu_starts),
                TvmFfiAny::tensor(tensors.tensormaps),
                TvmFfiAny::opaque(stream),
            ];
            // SAFETY: upheld by the caller.
            unsafe { self.module.call(&arguments) }
        } else {
            if tensors.state_checkpoints.is_some() || tensors.checkpoint_cu_starts.is_some() {
                return Err(Error::InvalidInput(
                    "non-checkpointed prefill does not accept checkpoint tensors".into(),
                ));
            }
            let arguments = [
                TvmFfiAny::tensor(tensors.q),
                TvmFfiAny::tensor(tensors.k),
                TvmFfiAny::tensor(tensors.v),
                TvmFfiAny::tensor(tensors.alpha),
                TvmFfiAny::tensor(tensors.beta),
                TvmFfiAny::tensor(tensors.initial_state),
                TvmFfiAny::tensor(tensors.output),
                TvmFfiAny::tensor(tensors.output_state),
                TvmFfiAny::tensor(tensors.cu_seqlens),
                TvmFfiAny::tensor(tensors.state_indices),
                TvmFfiAny::tensor(tensors.tensormaps),
                TvmFfiAny::opaque(stream),
            ];
            // SAFETY: upheld by the caller.
            unsafe { self.module.call(&arguments) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_backend_architecture_and_state_contracts() {
        PrefillSpecialization::new(PrefillBackend::Sm90, "sm_90a", 8, 16, 128, 128, 132).unwrap();
        PrefillSpecialization::new(PrefillBackend::Sm100, "sm_100a", 8, 16, 128, 128, 132).unwrap();
        PrefillSpecialization::new(PrefillBackend::Sm120, "sm_121a", 8, 16, 128, 128, 20).unwrap();
        assert!(
            PrefillSpecialization::new(PrefillBackend::Sm100, "sm_121a", 8, 16, 128, 128, 20)
                .is_err()
        );
        PrefillSpecialization::new(PrefillBackend::Sm100, "sm_100a", 8, 16, 128, 128, 132)
            .unwrap()
            .checkpoint_every_n_tokens(64)
            .validate()
            .unwrap();
        assert!(
            PrefillSpecialization::new(PrefillBackend::Sm100, "sm_100a", 8, 16, 128, 128, 132)
                .unwrap()
                .checkpoint_every_n_tokens(65)
                .validate()
                .is_err()
        );
    }
}
