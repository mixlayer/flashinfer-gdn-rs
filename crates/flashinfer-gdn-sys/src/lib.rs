#![deny(unsafe_op_in_unsafe_fn)]
//! Raw FlashInfer GDN specialization and entrypoint integration.
//!
//! This crate owns the versioned CuTeDSL shims and the unsafe boundary between
//! GDN-specific argument schemas and `cutedsl-jit` modules. It intentionally does
//! not depend on Candle.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use cutedsl_jit::{Abi, Artifact, ArtifactCache, CacheKey, CompilerCommand, Error, Result};
use serde_json::{Value, json};

/// FlashInfer release whose GDN source and shim contracts this crate targets.
pub const FLASHINFER_VERSION: &str = env!("FLASHINFER_GDN_VERSION");

/// Git revision corresponding to [`FLASHINFER_VERSION`].
pub const FLASHINFER_GIT_REV: &str = env!("FLASHINFER_GDN_GIT_REV");

/// Returns the FlashInfer source tree selected by this crate's build script.
#[must_use]
pub fn source_root() -> &'static Path {
    Path::new(env!("FLASHINFER_GDN_SOURCE_ROOT"))
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
        })
    }

    /// Constructs an adapter from an environment created by prepare_environment.py.
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
        // Absolute lock and runtime-library paths are deliberately excluded. The
        // environment digest and exact package set identify the compiler without
        // making equivalent environments in different cache roots miss.
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

    /// Returns the full content-addressed key without compiling.
    pub fn cache_key(&self) -> Result<CacheKey> {
        let paths = compiler_paths();
        let request = read_json(&paths.request)?;
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
        .with_input_file("request-file", &paths.request)?
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
        let command = CompilerCommand::new(&self.python)
            .arg(&paths.shim)
            .arg("--request")
            .arg(&paths.request)
            .arg("--flashinfer-root")
            .arg(&self.flashinfer_root)
            .arg("--output-dir")
            .output_directory_arg()
            .timeout(self.timeout);
        self.cache.prepare(&key, &command)
    }
}

#[derive(Debug)]
struct CompilerPaths {
    shim: PathBuf,
    request: PathBuf,
    requirements_lock: PathBuf,
}

fn compiler_paths() -> CompilerPaths {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shims = manifest.join("shims");
    CompilerPaths {
        shim: shims.join("compile_pretranspose_decode.py"),
        request: shims.join("requests/pretranspose_decode_sm121_bf16.json"),
        requirements_lock: shims.join("requirements/cu13-aarch64-py312.lock"),
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
            "request-file",
            "requirements-lock",
            "flashinfer-kernel-source",
        ] {
            assert!(key.inputs.contains_key(name), "missing {name}");
        }
        assert_eq!(key.request["gpu_arch"], "sm_121a");
        assert!(key.toolchain.get("host_c_compiler").is_some());
    }
}
