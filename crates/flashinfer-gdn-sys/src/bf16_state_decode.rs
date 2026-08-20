use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

use cutedsl_jit::{Abi, Artifact, CacheKey, CompilerCommand, Error, Result, TvmFfiAny, TvmModule};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    DlTensor, DtBiasDType, GdnHandle, InputDType, host_compiler_identity, runtime_asset_root,
    valid_gpu_architecture, write_json,
};

/// Device implementation selected by FlashInfer's BF16-state T=1 dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bf16StateDecodeKernelVariant {
    /// Higher-occupancy ILP4 fallback for smaller `B*HV` workloads.
    Ilp4,
    /// 128-bit vector load/store path used once `B*HV >= 512`.
    WideVecT1,
}

/// Complete compile-time request for BF16-state, single-token decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bf16StateDecodeSpecialization {
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
    /// Decay-bias type. The pinned T=1 API uses float32.
    pub dt_bias_dtype: DtBiasDType,
    /// Query heads.
    pub h: usize,
    /// Value/state heads.
    pub hv: usize,
    /// Query/key dimension; fixed to 128 upstream.
    pub k: usize,
    /// Value dimension; fixed to 128 upstream.
    pub v: usize,
    /// Tokens per call; fixed to one.
    pub t: usize,
    /// Query scale.
    pub scale: f32,
    /// Whether Q/K L2 normalization is fused.
    pub use_qk_l2norm: bool,
    /// Upstream implementation selected for this artifact.
    pub variant: Bf16StateDecodeKernelVariant,
    /// Compile-time number of V rows handled by one CTA.
    pub tile_v: usize,
    /// Whether SM100+ packed F32x2 arithmetic is generated.
    pub use_packed_fma: bool,
    /// Whether read and write state indices alias. The initial slice requires this.
    pub same_pool: bool,
}

impl Bf16StateDecodeSpecialization {
    /// Creates the standard same-slot BF16-state specialization selected by upstream.
    pub fn new(
        gpu_arch: impl Into<String>,
        h: usize,
        hv: usize,
        k: usize,
        v: usize,
        batch: usize,
        num_sms: usize,
    ) -> Result<Self> {
        let gpu_arch = gpu_arch.into();
        let (variant, tile_v) = select_dispatch(batch, hv, v, num_sms)?;
        let specialization = Self {
            schema_version: 1,
            kernel: "gdn_decode_bf16_state_t1".into(),
            symbol: "flashinfer_gdn_decode_bf16_state_t1".into(),
            use_packed_fma: packed_fma_for_architecture(&gpu_arch)?,
            gpu_arch,
            io_dtype: InputDType::Bfloat16,
            dt_bias_dtype: DtBiasDType::Float32,
            h,
            hv,
            k,
            v,
            t: 1,
            scale: (k as f32).sqrt().recip(),
            use_qk_l2norm: true,
            variant,
            tile_v,
            same_pool: true,
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

    /// Returns whether this artifact matches upstream dispatch on a device.
    pub fn matches_runtime(&self, batch: usize, num_sms: usize) -> Result<bool> {
        let (variant, tile_v) = select_dispatch(batch, self.hv, self.v, num_sms)?;
        Ok(self.variant == variant && self.tile_v == tile_v)
    }

    /// Validates the pinned kernel's compile-time contract.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.kernel != "gdn_decode_bf16_state_t1"
            || self.symbol != "flashinfer_gdn_decode_bf16_state_t1"
        {
            return Err(Error::InvalidInput(
                "unsupported BF16-state decode request identity".into(),
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
                "BF16-state T=1 decode requires BF16 I/O and float32 dt_bias".into(),
            ));
        }
        if self.h == 0 || self.hv == 0 || self.hv < self.h || !self.hv.is_multiple_of(self.h) {
            return Err(Error::InvalidInput(format!(
                "HV ({}) must be a positive multiple of H ({})",
                self.hv, self.h
            )));
        }
        if self.t != 1 || self.k != 128 || self.v != 128 {
            return Err(Error::InvalidInput(format!(
                "BF16-state decode requires T=1 and K=V=128; found T={}, K={}, V={}",
                self.t, self.k, self.v
            )));
        }
        let valid_tile = match self.variant {
            Bf16StateDecodeKernelVariant::Ilp4 => matches!(self.tile_v, 16 | 32 | 64 | 128),
            Bf16StateDecodeKernelVariant::WideVecT1 => matches!(self.tile_v, 64 | 128),
        };
        if !valid_tile || !self.v.is_multiple_of(self.tile_v) {
            return Err(Error::InvalidInput(format!(
                "invalid tile_v {} for {:?}",
                self.tile_v, self.variant
            )));
        }
        if !self.same_pool {
            return Err(Error::InvalidInput(
                "the initial BF16-state decode slice requires same-slot state updates".into(),
            ));
        }
        if [self.h, self.hv, self.k, self.v, self.t, self.tile_v]
            .into_iter()
            .any(|dimension| i64::try_from(dimension).is_err())
        {
            return Err(Error::InvalidInput(
                "BF16-state decode dimensions must fit signed 64-bit DLPack shapes".into(),
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

impl Default for Bf16StateDecodeSpecialization {
    fn default() -> Self {
        Self::new("sm_121a", 16, 16, 128, 128, 1, 1)
            .expect("the built-in BF16-state decode specialization is valid")
    }
}

fn select_dispatch(
    batch: usize,
    hv: usize,
    v: usize,
    num_sms: usize,
) -> Result<(Bf16StateDecodeKernelVariant, usize)> {
    if batch == 0 || hv == 0 || num_sms == 0 {
        return Err(Error::InvalidInput(format!(
            "BF16-state dispatch requires positive batch, HV, and SM count; found {batch}, {hv}, {num_sms}"
        )));
    }
    let work_units = batch
        .checked_mul(hv)
        .ok_or_else(|| Error::InvalidInput("B*HV overflows usize".into()))?;
    if work_units >= 1024 {
        return Ok((Bf16StateDecodeKernelVariant::WideVecT1, 128));
    }
    if work_units >= 512 {
        return Ok((Bf16StateDecodeKernelVariant::WideVecT1, 64));
    }
    if work_units <= 128 {
        return Ok((Bf16StateDecodeKernelVariant::Ilp4, usize::min(16, v)));
    }
    let target_blocks = num_sms
        .checked_mul(4)
        .ok_or_else(|| Error::InvalidInput("4*num_sms overflows usize".into()))?;
    for tile_v in [128, 64, 32] {
        if tile_v <= v && v.is_multiple_of(tile_v) {
            let blocks = work_units.checked_mul(v / tile_v).ok_or_else(|| {
                Error::InvalidInput("BF16-state grid size overflows usize".into())
            })?;
            if blocks >= target_blocks {
                return Ok((Bf16StateDecodeKernelVariant::Ilp4, tile_v));
            }
        }
    }
    Ok((Bf16StateDecodeKernelVariant::Ilp4, 32))
}

pub(super) fn packed_fma_for_architecture(architecture: &str) -> Result<bool> {
    if !valid_gpu_architecture(architecture) {
        return Err(Error::InvalidInput(format!(
            "invalid CuTeDSL GPU architecture {architecture:?}"
        )));
    }
    let digits: String = architecture
        .trim_start_matches("sm_")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let numeric = digits.parse::<u32>().map_err(|error| {
        Error::InvalidInput(format!(
            "failed to parse GPU architecture {architecture:?}: {error}"
        ))
    })?;
    Ok(numeric >= 100)
}

impl GdnHandle {
    /// Returns the content-addressed key for one `T=1` specialization.
    pub fn bf16_state_decode_cache_key(
        &self,
        specialization: &Bf16StateDecodeSpecialization,
    ) -> Result<CacheKey> {
        specialization.validate()?;
        let paths = compiler_paths();
        let request = serde_json::to_value(specialization).map_err(|error| {
            Error::InvalidInput(format!(
                "failed to serialize BF16-state decode request: {error}"
            ))
        })?;
        let toolchain = json!({
            "python_environment": self.toolchain.clone(),
            "host_c_compiler": host_compiler_identity()?,
        });
        CacheKey::new(
            "flashinfer-gdn/decode-bf16-state-t1",
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

    /// Prepares one `T=1` artifact, compiling only on a cache miss.
    pub fn prepare_bf16_state_decode(
        &self,
        specialization: &Bf16StateDecodeSpecialization,
    ) -> Result<Artifact> {
        let paths = compiler_paths();
        let key = self.bf16_state_decode_cache_key(specialization)?;
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

    /// Prepares and dynamically loads one `T=1` specialization.
    pub fn load_bf16_state_decode(
        &self,
        specialization: &Bf16StateDecodeSpecialization,
    ) -> Result<Bf16StateDecodeKernel> {
        let artifact = self.prepare_bf16_state_decode(specialization)?;
        let module = TvmModule::load(artifact)?;
        Ok(Bf16StateDecodeKernel {
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
    let shims = runtime_asset_root().join("shims");
    CompilerPaths {
        shim: shims.join("compile_bf16_state_decode.py"),
        support: shims.join("_artifact.py"),
        requirements_lock: shims.join("requirements/cu13-linux-py312.lock"),
    }
}

/// Superset of tensor arguments used by the two BF16-state T=1 launchers.
#[derive(Debug)]
pub struct Bf16StateDecodeTensors<'a> {
    /// BF16 state pool `[P,HV,V,K]`.
    pub state: &'a mut DlTensor,
    /// Unused intermediate-state placeholder.
    pub intermediate: &'a mut DlTensor,
    /// Log-decay `[HV]`, float32.
    pub a_log: &'a mut DlTensor,
    /// Input-dependent decay `[B,1,HV]`, BF16.
    pub a: &'a mut DlTensor,
    /// Decay bias `[HV]`, float32.
    pub dt_bias: &'a mut DlTensor,
    /// Query `[B,1,H,K]`, BF16.
    pub q: &'a mut DlTensor,
    /// Key `[B,1,H,K]`, BF16.
    pub k: &'a mut DlTensor,
    /// Value `[B,1,HV,V]`, BF16.
    pub v: &'a mut DlTensor,
    /// Update gate `[B,1,HV]`, BF16.
    pub beta: &'a mut DlTensor,
    /// Output `[B,1,HV,V]`, BF16.
    pub output: &'a mut DlTensor,
    /// Read indices `[B]`, int32.
    pub state_indices: &'a mut DlTensor,
    /// Write indices `[B]`, int32; aliases `state_indices` in this slice.
    pub output_state_indices: &'a mut DlTensor,
    /// Unused accepted-step placeholder `[B]`, int32.
    pub accepted_steps: &'a mut DlTensor,
    /// Unused per-token scatter placeholder `[B,1]`, int32.
    pub ssm_state_indices: &'a mut DlTensor,
}

/// Loaded BF16-state single-token decode specialization.
#[derive(Debug, Clone)]
pub struct Bf16StateDecodeKernel {
    module: Arc<TvmModule>,
    specialization: Bf16StateDecodeSpecialization,
}

impl Bf16StateDecodeKernel {
    /// Specialization enforced by this generated entrypoint.
    #[must_use]
    pub fn specialization(&self) -> &Bf16StateDecodeSpecialization {
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
        tensors: &mut Bf16StateDecodeTensors<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        let common = [
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
        ];
        match self.specialization.variant {
            Bf16StateDecodeKernelVariant::WideVecT1 => {
                let arguments = [
                    common[0],
                    common[1],
                    common[2],
                    common[3],
                    common[4],
                    common[5],
                    common[6],
                    common[7],
                    common[8],
                    common[9],
                    common[10],
                    common[11],
                    TvmFfiAny::opaque(stream),
                ];
                // SAFETY: the caller upholds the descriptor and stream contracts.
                unsafe { self.module.call(&arguments) }
            }
            Bf16StateDecodeKernelVariant::Ilp4 => {
                let arguments = [
                    common[0],
                    common[1],
                    common[2],
                    common[3],
                    common[4],
                    common[5],
                    common[6],
                    common[7],
                    common[8],
                    common[9],
                    common[10],
                    common[11],
                    TvmFfiAny::tensor(tensors.accepted_steps),
                    TvmFfiAny::tensor(tensors.ssm_state_indices),
                    TvmFfiAny::opaque(stream),
                ];
                // SAFETY: the caller upholds the descriptor and stream contracts.
                unsafe { self.module.call(&arguments) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_upstream_t1_dispatch_boundaries() {
        let tiny = Bf16StateDecodeSpecialization::new("sm_121a", 1, 64, 128, 128, 1, 132).unwrap();
        assert_eq!(tiny.variant, Bf16StateDecodeKernelVariant::Ilp4);
        assert_eq!(tiny.tile_v, 16);

        let medium =
            Bf16StateDecodeSpecialization::new("sm_121a", 1, 64, 128, 128, 7, 132).unwrap();
        assert_eq!(medium.variant, Bf16StateDecodeKernelVariant::Ilp4);
        assert_eq!(medium.tile_v, 64);

        let wide = Bf16StateDecodeSpecialization::new("sm_121a", 1, 64, 128, 128, 8, 132).unwrap();
        assert_eq!(wide.variant, Bf16StateDecodeKernelVariant::WideVecT1);
        assert_eq!(wide.tile_v, 64);

        let widest =
            Bf16StateDecodeSpecialization::new("sm_121a", 1, 64, 128, 128, 16, 132).unwrap();
        assert_eq!(widest.variant, Bf16StateDecodeKernelVariant::WideVecT1);
        assert_eq!(widest.tile_v, 128);
    }

    #[test]
    fn packed_fma_follows_architecture_generation() {
        assert!(
            Bf16StateDecodeSpecialization::new("sm_90a", 1, 1, 128, 128, 1, 1)
                .is_ok_and(|spec| !spec.use_packed_fma)
        );
        assert!(
            Bf16StateDecodeSpecialization::new("sm_100a", 1, 1, 128, 128, 1, 1)
                .is_ok_and(|spec| spec.use_packed_fma)
        );
    }
}
