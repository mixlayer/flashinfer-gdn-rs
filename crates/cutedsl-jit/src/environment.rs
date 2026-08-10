use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::{Error, Result};

const PREPARE_ENVIRONMENT: &str = include_str!("../python/prepare_environment.py");

/// Validated CuTeDSL compiler environment selected by the preparation helper.
#[derive(Debug, Clone)]
pub struct PythonEnvironment {
    python: PathBuf,
    identity: Value,
}

impl PythonEnvironment {
    /// Interpreter used for out-of-process compilation.
    #[must_use]
    pub fn python(&self) -> &Path {
        &self.python
    }

    /// Path-independent identity included in artifact cache keys.
    #[must_use]
    pub fn identity(&self) -> &Value {
        &self.identity
    }
}

/// Returns the shared CuTeDSL cache root for environments and artifacts.
///
/// Resolution follows `CUTEDSL_JIT_CACHE_DIR`, then `XDG_CACHE_HOME/cutedsl-jit`,
/// then `HOME/.cache/cutedsl-jit`.
pub fn default_cache_root() -> Result<PathBuf> {
    cache_root_from(
        std::env::var_os("CUTEDSL_JIT_CACHE_DIR"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// Locates, validates, or atomically installs the environment for `lock`.
///
/// `CUTEDSL_JIT_BASE_PYTHON` selects the bootstrap interpreter. The embedded
/// helper additionally honors `CUTEDSL_JIT_PYTHON` and `CUTEDSL_JIT_CACHE_DIR`.
pub fn prepare_python_environment(lock: impl AsRef<Path>) -> Result<PythonEnvironment> {
    let lock = lock.as_ref();
    let bootstrap = std::env::var_os("CUTEDSL_JIT_BASE_PYTHON").unwrap_or_else(|| "python3".into());
    let mut command = Command::new(&bootstrap);
    command
        .arg("-c")
        .arg(PREPARE_ENVIRONMENT)
        .arg("--lock")
        .arg(lock);
    if environment_flag("CUTEDSL_JIT_OFFLINE") {
        command.arg("--offline");
    }
    if let Some(wheelhouse) = std::env::var_os("CUTEDSL_JIT_WHEELHOUSE") {
        command.arg("--wheelhouse").arg(wheelhouse);
    }

    let output = command.output().map_err(|error| {
        Error::InvalidInput(format!(
            "failed to start CuTeDSL environment preparation with {bootstrap:?}: {error}"
        ))
    })?;
    if !output.status.success() {
        return Err(Error::InvalidInput(format!(
            "CuTeDSL environment preparation exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_environment_metadata(&output.stdout)
}

fn parse_environment_metadata(bytes: &[u8]) -> Result<PythonEnvironment> {
    let metadata: Value = serde_json::from_slice(bytes).map_err(|error| {
        Error::InvalidInput(format!(
            "failed to parse CuTeDSL environment metadata: {error}; output was {:?}",
            String::from_utf8_lossy(bytes).trim()
        ))
    })?;
    let field = |name: &str| {
        metadata.get(name).cloned().ok_or_else(|| {
            Error::InvalidInput(format!(
                "CuTeDSL environment metadata has no {name:?} field"
            ))
        })
    };
    let python = field("python")?
        .as_str()
        .map(PathBuf::from)
        .ok_or_else(|| {
            Error::InvalidInput("CuTeDSL environment Python path is not a string".into())
        })?;
    if !python.is_file() {
        return Err(Error::InvalidInput(format!(
            "CuTeDSL environment Python does not exist: {}",
            python.display()
        )));
    }
    let identity = json!({
        "environment_schema_version": field("schema_version")?,
        "managed": field("managed")?,
        "environment_digest": metadata.get("environment_digest").cloned(),
        "python": field("python_version")?,
        "packages": field("packages")?,
    });
    Ok(PythonEnvironment { python, identity })
}

fn environment_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        !value.is_empty()
            && value != "0"
            && !value.eq_ignore_ascii_case("false")
            && !value.eq_ignore_ascii_case("no")
    })
}

fn cache_root_from(
    cutedsl: Option<OsString>,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Result<PathBuf> {
    if let Some(value) = cutedsl.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value));
    }
    if let Some(value) = xdg.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value).join("cutedsl-jit"));
    }
    if let Some(value) = home.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value).join(".cache/cutedsl-jit"));
    }
    Err(Error::InvalidInput(
        "cannot determine the CuTeDSL cache root; set CUTEDSL_JIT_CACHE_DIR, XDG_CACHE_HOME, or HOME"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_path_and_path_independent_identity() {
        let python = std::env::current_exe().unwrap();
        let metadata = serde_json::to_vec(&json!({
            "schema_version": 1,
            "managed": false,
            "python": python,
            "python_version": "3.12.3",
            "packages": {"nvidia-cutlass-dsl": "4.7.0"},
            "runtime_libraries": ["/some/environment/lib/runtime.so"],
        }))
        .unwrap();

        let environment = parse_environment_metadata(&metadata).unwrap();
        assert_eq!(environment.python(), std::env::current_exe().unwrap());
        assert_eq!(environment.identity()["managed"], false);
        assert_eq!(environment.identity()["python"], "3.12.3");
        assert!(environment.identity().get("runtime_libraries").is_none());
    }

    #[test]
    fn cache_root_uses_standard_precedence() {
        assert_eq!(
            cache_root_from(
                Some("/explicit".into()),
                Some("/xdg".into()),
                Some("/home".into())
            )
            .unwrap(),
            PathBuf::from("/explicit")
        );
        assert_eq!(
            cache_root_from(None, Some("/xdg".into()), Some("/home".into())).unwrap(),
            PathBuf::from("/xdg/cutedsl-jit")
        );
        assert_eq!(
            cache_root_from(None, None, Some("/home".into())).unwrap(),
            PathBuf::from("/home/.cache/cutedsl-jit")
        );
    }
}
