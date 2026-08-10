#![deny(unsafe_op_in_unsafe_fn)]
//! Raw FlashInfer BF16-state GDN decode specialization and entrypoint integration.
//!
//! This crate owns the versioned CuTeDSL shims and the unsafe boundary between
//! GDN-specific argument schemas and `cutedsl-jit` modules. It intentionally does
//! not depend on Candle.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use cutedsl_jit::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

mod bf16_state_decode;
mod bf16_state_mtp;

pub use bf16_state_decode::{
    Bf16StateDecodeCompiler, Bf16StateDecodeKernel, Bf16StateDecodeKernelVariant,
    Bf16StateDecodeSpecialization, Bf16StateDecodeTensors,
};
pub use bf16_state_mtp::{
    Bf16StateMtpCompiler, Bf16StateMtpKernel, Bf16StateMtpKernelVariant,
    Bf16StateMtpSpecialization, Bf16StateMtpTensors,
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
    fn selected_source_contains_bf16_decode() {
        assert_eq!(FLASHINFER_VERSION, "0.6.16.post2");
        assert!(
            source_root()
                .join("flashinfer/gdn_kernels/gdn_decode_bf16_state.py")
                .is_file()
        );
    }
}
