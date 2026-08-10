use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutedsl_jit::{Abi, Artifact, CacheKey, CompilerCommand, Error, Result, TvmFfiAny, TvmModule};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::bf16_state_decode::packed_fma_for_architecture;
use super::{
    DlTensor, DtBiasDType, GdnHandle, InputDType, host_compiler_identity, valid_gpu_architecture,
    write_json,
};

/// Device implementation selected by FlashInfer's BF16-state MTP dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bf16StateMtpKernelVariant {
    /// Higher-occupancy fallback for `B*HV < 128`.
    Ilp4,
    /// General 128-bit vector load/store path for `B*HV >= 128`.
    WideVec,
}

/// Compile-time request for checkpointed BF16-state MTP decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bf16StateMtpSpecialization {
    /// Request schema version.
    pub schema_version: u32,
    /// Versioned kernel family name.
    pub kernel: String,
    /// Unprefixed symbol passed to CuTeDSL export.
    pub symbol: String,
    /// CuTeDSL GPU architecture, such as `sm_90a` or `sm_121a`.
    pub gpu_arch: String,
    /// Q/K/V/gate input type. The pinned kernel requires BF16.
    pub io_dtype: InputDType,
    /// Decay-bias type. The pinned kernel requires float32.
    pub dt_bias_dtype: DtBiasDType,
    /// Query heads.
    pub h: usize,
    /// Value/state heads.
    pub hv: usize,
    /// Query/key dimension; fixed to 128 upstream.
    pub k: usize,
    /// Value dimension; fixed to 128 upstream.
    pub v: usize,
    /// Tokens processed sequentially by one call; at least two.
    pub t: usize,
    /// Query scale.
    pub scale: f32,
    /// Whether Q/K L2 normalization is fused.
    pub use_qk_l2norm: bool,
    /// Upstream implementation selected for this artifact.
    pub variant: Bf16StateMtpKernelVariant,
    /// Compile-time number of V rows handled by one CTA.
    pub tile_v: usize,
    /// Whether SM100+ packed F32x2 arithmetic is generated.
    pub use_packed_fma: bool,
    /// Whether read and write state indices alias. This initial slice requires it.
    pub same_pool: bool,
    /// Whether every post-token state is scattered to a caller-selected pool slot.
    pub per_token_pool_scatter: bool,
    /// Whether compact pool storage is also passed as a flat scatter view.
    pub per_token_pool_scatter_flat: bool,
}

impl Bf16StateMtpSpecialization {
    /// Selects the checkpointed MTP implementation used by upstream.
    pub fn new(
        gpu_arch: impl Into<String>,
        h: usize,
        hv: usize,
        k: usize,
        v: usize,
        t: usize,
        batch: usize,
    ) -> Result<Self> {
        let gpu_arch = gpu_arch.into();
        let (variant, tile_v) = select_dispatch(batch, hv, t)?;
        let specialization = Self {
            schema_version: 1,
            kernel: "gdn_decode_bf16_state_mtp".into(),
            symbol: "flashinfer_gdn_decode_bf16_state_mtp".into(),
            use_packed_fma: packed_fma_for_architecture(&gpu_arch)?,
            gpu_arch,
            io_dtype: InputDType::Bfloat16,
            dt_bias_dtype: DtBiasDType::Float32,
            h,
            hv,
            k,
            v,
            t,
            scale: (k as f32).sqrt().recip(),
            use_qk_l2norm: true,
            variant,
            tile_v,
            same_pool: true,
            per_token_pool_scatter: true,
            per_token_pool_scatter_flat: true,
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

    /// Enables or disables fused Q/K L2 normalization.
    #[must_use]
    pub fn qk_l2norm(mut self, enabled: bool) -> Self {
        self.use_qk_l2norm = enabled;
        self
    }

    /// Returns whether this artifact matches upstream dispatch for `batch`.
    pub fn matches_runtime(&self, batch: usize) -> Result<bool> {
        let (variant, tile_v) = select_dispatch(batch, self.hv, self.t)?;
        Ok(self.variant == variant && self.tile_v == tile_v)
    }

    /// Validates the initial MTP specialization contract.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.kernel != "gdn_decode_bf16_state_mtp"
            || self.symbol != "flashinfer_gdn_decode_bf16_state_mtp"
        {
            return Err(Error::InvalidInput(
                "unsupported BF16-state MTP request identity".into(),
            ));
        }
        if !valid_gpu_architecture(&self.gpu_arch) {
            return Err(Error::InvalidInput(format!(
                "invalid CuTeDSL GPU architecture {:?}",
                self.gpu_arch
            )));
        }
        if self.use_packed_fma != packed_fma_for_architecture(&self.gpu_arch)? {
            return Err(Error::InvalidInput(format!(
                "packed-FMA selection does not match architecture {}",
                self.gpu_arch
            )));
        }
        if self.io_dtype != InputDType::Bfloat16 || self.dt_bias_dtype != DtBiasDType::Float32 {
            return Err(Error::InvalidInput(
                "BF16-state MTP requires BF16 I/O and float32 dt_bias".into(),
            ));
        }
        if self.h == 0 || self.hv == 0 || self.hv < self.h || !self.hv.is_multiple_of(self.h) {
            return Err(Error::InvalidInput(format!(
                "HV ({}) must be a positive multiple of H ({})",
                self.hv, self.h
            )));
        }
        if self.t < 2 || self.k != 128 || self.v != 128 {
            return Err(Error::InvalidInput(format!(
                "BF16-state MTP requires T>=2 and K=V=128; found T={}, K={}, V={}",
                self.t, self.k, self.v
            )));
        }
        let valid_tile = match self.variant {
            Bf16StateMtpKernelVariant::Ilp4 => self.tile_v == 16,
            Bf16StateMtpKernelVariant::WideVec => matches!(self.tile_v, 32 | 64 | 128),
        };
        if !valid_tile || !self.v.is_multiple_of(self.tile_v) {
            return Err(Error::InvalidInput(format!(
                "invalid tile_v {} for {:?}",
                self.tile_v, self.variant
            )));
        }
        if !self.same_pool {
            return Err(Error::InvalidInput(
                "checkpointed BF16-state MTP requires aliased read/final-write indices".into(),
            ));
        }
        if !self.per_token_pool_scatter || !self.per_token_pool_scatter_flat {
            return Err(Error::InvalidInput(
                "BF16-state MTP requires per-token flat pool scatter".into(),
            ));
        }
        if [self.h, self.hv, self.k, self.v, self.t, self.tile_v]
            .into_iter()
            .any(|dimension| i64::try_from(dimension).is_err())
        {
            return Err(Error::InvalidInput(
                "BF16-state MTP dimensions must fit signed 64-bit DLPack shapes".into(),
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

impl Default for Bf16StateMtpSpecialization {
    fn default() -> Self {
        Self::new("sm_121a", 16, 16, 128, 128, 2, 1)
            .expect("the built-in BF16-state MTP specialization is valid")
    }
}

fn select_dispatch(
    batch: usize,
    hv: usize,
    t: usize,
) -> Result<(Bf16StateMtpKernelVariant, usize)> {
    if batch == 0 || hv == 0 || t < 2 {
        return Err(Error::InvalidInput(format!(
            "BF16-state MTP dispatch requires positive batch/HV and T>=2; found {batch}, {hv}, {t}"
        )));
    }
    let work_units = batch
        .checked_mul(hv)
        .ok_or_else(|| Error::InvalidInput("B*HV overflows usize".into()))?;
    if work_units >= 1024 {
        Ok((Bf16StateMtpKernelVariant::WideVec, 128))
    } else if work_units >= 512 {
        Ok((Bf16StateMtpKernelVariant::WideVec, 64))
    } else if work_units >= 128 {
        Ok((Bf16StateMtpKernelVariant::WideVec, 32))
    } else {
        Ok((Bf16StateMtpKernelVariant::Ilp4, 16))
    }
}

impl GdnHandle {
    /// Returns the content-addressed key for one MTP specialization.
    pub fn bf16_state_mtp_cache_key(
        &self,
        specialization: &Bf16StateMtpSpecialization,
    ) -> Result<CacheKey> {
        specialization.validate()?;
        let paths = compiler_paths();
        let request = serde_json::to_value(specialization).map_err(|error| {
            Error::InvalidInput(format!(
                "failed to serialize BF16-state MTP request: {error}"
            ))
        })?;
        let toolchain = json!({
            "python_environment": self.toolchain.clone(),
            "host_c_compiler": host_compiler_identity()?,
        });
        CacheKey::new(
            "flashinfer-gdn/decode-bf16-state-mtp-pool-scatter",
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
                .join("flashinfer/gdn_kernels/gdn_decode_bf16_state.py"),
        )
    }

    /// Prepares one MTP artifact, compiling only on a cache miss.
    pub fn prepare_bf16_state_mtp(
        &self,
        specialization: &Bf16StateMtpSpecialization,
    ) -> Result<Artifact> {
        let paths = compiler_paths();
        let key = self.bf16_state_mtp_cache_key(specialization)?;
        let request = specialization.clone();
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

    /// Prepares and dynamically loads one MTP specialization.
    pub fn load_bf16_state_mtp(
        &self,
        specialization: &Bf16StateMtpSpecialization,
    ) -> Result<Bf16StateMtpKernel> {
        let artifact = self.prepare_bf16_state_mtp(specialization)?;
        let module = TvmModule::load(artifact)?;
        Ok(Bf16StateMtpKernel {
            module: Arc::new(module),
            specialization: specialization.clone(),
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
        shim: shims.join("compile_bf16_state_decode.py"),
        support: shims.join("_artifact.py"),
        requirements_lock: shims.join("requirements/cu13-aarch64-py312.lock"),
    }
}

/// Tensor arguments shared by the ILP4 and wide-vector BF16-state MTP launchers.
#[derive(Debug)]
pub struct Bf16StateMtpTensors<'a> {
    /// BF16 state pool `[P,HV,V,K]`.
    pub state: &'a mut DlTensor,
    /// Flat alias of `state`, `[P*HV,V,K]`, used for compact-pool scatter.
    pub intermediate: &'a mut DlTensor,
    /// Log-decay `[HV]`, float32.
    pub a_log: &'a mut DlTensor,
    /// Input-dependent decay `[B,T,HV]`, BF16.
    pub a: &'a mut DlTensor,
    /// Decay bias `[HV]`, float32.
    pub dt_bias: &'a mut DlTensor,
    /// Query `[B,T,H,K]`, BF16.
    pub q: &'a mut DlTensor,
    /// Key `[B,T,H,K]`, BF16.
    pub k: &'a mut DlTensor,
    /// Value `[B,T,HV,V]`, BF16.
    pub v: &'a mut DlTensor,
    /// Update gate `[B,T,HV]`, BF16.
    pub beta: &'a mut DlTensor,
    /// Output `[B,T,HV,V]`, BF16.
    pub output: &'a mut DlTensor,
    /// Read indices `[B]`, int32.
    pub state_indices: &'a mut DlTensor,
    /// Write indices `[B]`, int32; aliases `state_indices` in this slice.
    pub output_state_indices: &'a mut DlTensor,
    /// Unused accepted-step placeholder `[B]`, int32.
    pub accepted_steps: &'a mut DlTensor,
    /// Caller-selected checkpoint pool slots `[B,T]`, int32.
    pub ssm_state_indices: &'a mut DlTensor,
}

/// Loaded BF16-state MTP specialization.
#[derive(Debug, Clone)]
pub struct Bf16StateMtpKernel {
    module: Arc<TvmModule>,
    specialization: Bf16StateMtpSpecialization,
}

impl Bf16StateMtpKernel {
    /// Specialization enforced by this generated entrypoint.
    #[must_use]
    pub fn specialization(&self) -> &Bf16StateMtpSpecialization {
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
    /// Every descriptor and `stream` must remain valid until all enqueued work
    /// completes and must match the selected specialization.
    pub unsafe fn launch(
        &self,
        tensors: &mut Bf16StateMtpTensors<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        let arguments = [
            TvmFfiAny::tensor(tensors.state),
            TvmFfiAny::tensor(tensors.intermediate),
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
            TvmFfiAny::tensor(tensors.accepted_steps),
            TvmFfiAny::tensor(tensors.ssm_state_indices),
            TvmFfiAny::opaque(stream),
        ];
        // SAFETY: the caller upholds the descriptor and stream contracts.
        unsafe { self.module.call(&arguments) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_upstream_mtp_dispatch_boundaries() {
        let fallback = Bf16StateMtpSpecialization::new("sm_121a", 1, 1, 128, 128, 2, 127).unwrap();
        assert_eq!(fallback.variant, Bf16StateMtpKernelVariant::Ilp4);
        assert_eq!(fallback.tile_v, 16);

        let wide32 = Bf16StateMtpSpecialization::new("sm_121a", 1, 1, 128, 128, 2, 128).unwrap();
        assert_eq!(wide32.variant, Bf16StateMtpKernelVariant::WideVec);
        assert_eq!(wide32.tile_v, 32);

        let wide64 = Bf16StateMtpSpecialization::new("sm_121a", 1, 1, 128, 128, 2, 512).unwrap();
        assert_eq!(wide64.tile_v, 64);

        let wide128 = Bf16StateMtpSpecialization::new("sm_121a", 1, 1, 128, 128, 2, 1024).unwrap();
        assert_eq!(wide128.tile_v, 128);
    }

    #[test]
    fn rejects_single_token_mtp() {
        assert!(Bf16StateMtpSpecialization::new("sm_121a", 1, 1, 128, 128, 1, 1).is_err());
    }
}
