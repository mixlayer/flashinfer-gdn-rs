use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::digest::{canonical_json, digest_bytes, digest_file};
use crate::{Error, Result};

/// Current content-addressed cache-key schema.
pub const CACHE_KEY_SCHEMA_VERSION: u32 = 2;

/// Current compiler artifact manifest schema.
pub const ARTIFACT_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Invocation ABI exported by a compiled artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Abi {
    /// TVM FFI generated safe-call wrapper.
    TvmFfi,
    /// CuTeDSL's specialization-specific packed ABI.
    CutePacked,
}

/// Host identity included in every artifact key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostIdentity {
    /// Rust target architecture of the running compiler orchestrator.
    pub architecture: String,
    /// Rust target operating system of the running compiler orchestrator.
    pub operating_system: String,
    /// Rust target environment, such as `gnu` or `musl`.
    pub environment: String,
}

impl HostIdentity {
    fn current() -> Self {
        Self {
            architecture: std::env::consts::ARCH.to_owned(),
            operating_system: std::env::consts::OS.to_owned(),
            environment: target_environment().to_owned(),
        }
    }
}

fn target_environment() -> &'static str {
    if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else if cfg!(target_env = "msvc") {
        "msvc"
    } else {
        ""
    }
}

/// Canonical, content-addressed description of one compiler invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Cache-key format version.
    pub schema_version: u32,
    /// Library-defined kernel family, for diagnostics rather than path selection.
    pub namespace: String,
    /// Exported invocation ABI.
    pub abi: Abi,
    /// Host on which the loadable module is built.
    pub host: HostIdentity,
    /// Library-specific specialization request.
    pub request: Value,
    /// Named source and shim content digests.
    pub inputs: BTreeMap<String, String>,
    /// Exact compiler-environment identity or marker.
    pub toolchain: Value,
}

impl CacheKey {
    /// Starts a cache key for a library-specific specialization.
    pub fn new(
        namespace: impl Into<String>,
        abi: Abi,
        request: Value,
        toolchain: Value,
    ) -> Result<Self> {
        let namespace = namespace.into();
        if namespace.is_empty() || namespace.chars().any(char::is_control) {
            return Err(Error::InvalidInput(
                "cache namespace must be nonempty and contain no control characters".into(),
            ));
        }
        Ok(Self {
            schema_version: CACHE_KEY_SCHEMA_VERSION,
            namespace,
            abi,
            host: HostIdentity::current(),
            request,
            inputs: BTreeMap::new(),
            toolchain,
        })
    }

    /// Adds a named file-content digest to this key.
    pub fn with_input_file(
        mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::InvalidInput(
                "cache input name cannot be empty".into(),
            ));
        }
        let digest = digest_file(path.as_ref())?;
        if self.inputs.insert(name.clone(), digest).is_some() {
            return Err(Error::InvalidInput(format!(
                "duplicate cache input name {name:?}"
            )));
        }
        Ok(self)
    }

    /// Returns the canonical JSON representation used as the digest input.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }

    /// Returns this key's DeepGEMM-compatible 64-bit digest.
    pub fn digest(&self) -> Result<String> {
        Ok(digest_bytes(&self.canonical_json()?))
    }
}

/// One file emitted into an artifact directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFile {
    /// Path relative to the artifact directory.
    pub path: PathBuf,
    /// Compiler-provided lowercase SHA-256 digest retained as provenance.
    ///
    /// Cache validation uses the DeepGEMM-compatible digest in completion.json.
    pub sha256: String,
}

/// External runtime library required by the linked module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLibrary {
    /// Absolute compiler-environment library path.
    pub path: PathBuf,
    /// Compiler-provided lowercase SHA-256 digest retained as provenance.
    ///
    /// Cache validation uses the DeepGEMM-compatible digest in completion.json.
    pub sha256: String,
}

/// Common portion of a CuTeDSL compiler artifact manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    /// Compiler manifest schema version.
    pub schema_version: u32,
    /// Exported invocation ABI.
    pub abi: Abi,
    /// Dynamic symbol exported by the linked module.
    pub entry_symbol: String,
    /// Named files emitted into the artifact directory.
    pub artifacts: BTreeMap<String, ArtifactFile>,
    /// Runtime libraries that must be validated and loaded first.
    #[serde(default)]
    pub runtime_libraries: Vec<RuntimeLibrary>,
    /// TVM C-runtime ABI version observed by the compiler, when applicable.
    #[serde(default)]
    pub tvm_ffi_runtime_version: Option<String>,
    /// Library-specific compiler metadata retained verbatim.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn cache_key_digest_fixture_is_stable() {
        let key = CacheKey {
            schema_version: CACHE_KEY_SCHEMA_VERSION,
            namespace: "fixture/kernel".into(),
            abi: Abi::TvmFfi,
            host: HostIdentity {
                architecture: "aarch64".into(),
                operating_system: "linux".into(),
                environment: "gnu".into(),
            },
            request: json!({"b": 2, "a": 1}),
            inputs: BTreeMap::from([("source".into(), "0".repeat(16))]),
            toolchain: json!({"version": "1"}),
        };
        assert_eq!(key.digest().unwrap(), "d2b24f755de6464d");
    }
}
