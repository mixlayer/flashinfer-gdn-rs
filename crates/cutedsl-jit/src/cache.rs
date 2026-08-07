use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::command::CompilerCommand;
use crate::digest::{digest_file, verify_digest, write_json_file};
use crate::model::{ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactFile, ArtifactManifest, CacheKey};
use crate::{Error, Result};

const COMPLETION_SCHEMA_VERSION: u32 = 2;
const DIGEST_ALGORITHM: &str = "deepgemm-fnv1a64";
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
struct CompletionRecord {
    schema_version: u32,
    digest_algorithm: String,
    cache_key_digest: String,
    cache_key: CacheKey,
    manifest_digest: String,
    content_digests: ContentDigests,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ContentDigests {
    artifacts: BTreeMap<String, String>,
    runtime_libraries: Vec<RuntimeLibraryDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RuntimeLibraryDigest {
    path: PathBuf,
    digest: String,
}

/// A validated compiler artifact and its parsed manifest.
#[derive(Debug, Clone)]
pub struct Artifact {
    directory: PathBuf,
    cache_digest: Option<String>,
    manifest: ArtifactManifest,
    content_digests: ContentDigests,
}

impl Artifact {
    /// Validates an artifact that was produced outside ArtifactCache.
    pub fn from_manifest_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let directory = path.parent().ok_or_else(|| {
            Error::InvalidInput(format!(
                "manifest has no parent directory: {}",
                path.display()
            ))
        })?;
        load_manifest_directory(directory, None)
    }

    /// Artifact directory containing manifest.json and generated files.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Content-addressed cache digest, if this artifact came from a cache.
    pub fn cache_digest(&self) -> Option<&str> {
        self.cache_digest.as_deref()
    }

    /// Parsed and validated compiler manifest.
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Returns one named artifact file after manifest validation.
    pub fn file(&self, name: &str) -> Result<PathBuf> {
        let file = self
            .manifest
            .artifacts
            .get(name)
            .ok_or_else(|| Error::InvalidManifest {
                path: self.directory.join("manifest.json"),
                message: format!("manifest has no artifact named {name:?}"),
            })?;
        Ok(self.directory.join(&file.path))
    }
}

/// Content-addressed, interprocess-safe CuTeDSL artifact cache.
#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
}

impl ArtifactCache {
    /// Uses root for artifacts, locks, failures, and quarantined corrupt entries.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Cache root selected by this instance.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns an existing artifact or runs compiler exactly once per cache key.
    pub fn prepare(&self, key: &CacheKey, compiler: &CompilerCommand) -> Result<Artifact> {
        self.get_or_build(key, |directory| compiler.run(directory))
    }

    /// Returns an existing artifact or invokes a library-specific build closure.
    pub fn get_or_build<F>(&self, key: &CacheKey, build: F) -> Result<Artifact>
    where
        F: FnOnce(&Path) -> Result<()>,
    {
        self.ensure_layout()?;
        let digest = key.digest()?;
        let artifact_directory = self.root.join("artifacts").join(&digest);

        if artifact_directory.is_dir()
            && let Ok(artifact) = load_completed_artifact(&artifact_directory, &digest)
        {
            return Ok(artifact);
        }

        let lock_path = self.root.join("locks").join(format!("{digest}.lock"));
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                Error::io(
                    format!("failed to open artifact lock {}", lock_path.display()),
                    error,
                )
            })?;
        lock.lock()
            .map_err(|error| Error::io(format!("failed to lock artifact key {digest}"), error))?;

        if artifact_directory.is_dir() {
            match load_completed_artifact(&artifact_directory, &digest) {
                Ok(artifact) => return Ok(artifact),
                Err(_) => self.quarantine(&artifact_directory, &digest)?,
            }
        }

        let staging = create_unique_directory(&self.root.join("staging"), &digest)?;
        if let Err(error) = build(&staging) {
            return Err(self.preserve_failure(staging, &digest, error));
        }

        let staged_artifact = match load_manifest_directory(&staging, Some(digest.clone())) {
            Ok(artifact) => artifact,
            Err(error) => return Err(self.preserve_failure(staging, &digest, error)),
        };
        if staged_artifact.manifest().abi != key.abi {
            let error = Error::InvalidManifest {
                path: staging.join("manifest.json"),
                message: format!(
                    "worker emitted {:?} for {:?} cache key",
                    staged_artifact.manifest().abi,
                    key.abi
                ),
            };
            return Err(self.preserve_failure(staging, &digest, error));
        }
        let finalize = (|| {
            let manifest_path = staging.join("manifest.json");
            let completion = CompletionRecord {
                schema_version: COMPLETION_SCHEMA_VERSION,
                digest_algorithm: DIGEST_ALGORITHM.to_owned(),
                cache_key_digest: digest.clone(),
                cache_key: key.clone(),
                manifest_digest: digest_file(&manifest_path)?,
                content_digests: staged_artifact.content_digests.clone(),
            };
            write_json_file(&staging.join("completion.json"), &completion)?;
            sync_artifact(&staged_artifact)?;
            sync_file(&staging.join("completion.json"))?;
            sync_directory(&staging)
        })();
        if let Err(error) = finalize {
            return Err(self.preserve_failure(staging, &digest, error));
        }

        if let Err(error) = fs::rename(&staging, &artifact_directory) {
            let error = Error::io(
                format!(
                    "failed to publish artifact {} to {}",
                    staging.display(),
                    artifact_directory.display()
                ),
                error,
            );
            return Err(self.preserve_failure(staging, &digest, error));
        }
        sync_directory(&self.root.join("artifacts"))?;
        load_completed_artifact(&artifact_directory, &digest)
    }

    fn ensure_layout(&self) -> Result<()> {
        for name in ["artifacts", "locks", "staging", "failures", "corrupt"] {
            let path = self.root.join(name);
            fs::create_dir_all(&path).map_err(|error| {
                Error::io(
                    format!("failed to create cache directory {}", path.display()),
                    error,
                )
            })?;
        }
        Ok(())
    }

    fn quarantine(&self, artifact: &Path, digest: &str) -> Result<()> {
        let destination = unique_path(&self.root.join("corrupt"), digest);
        fs::rename(artifact, &destination).map_err(|error| {
            Error::io(
                format!(
                    "failed to quarantine corrupt artifact {} at {}",
                    artifact.display(),
                    destination.display()
                ),
                error,
            )
        })
    }

    fn preserve_failure(&self, staging: PathBuf, digest: &str, error: Error) -> Error {
        let destination = unique_path(&self.root.join("failures"), digest);
        match fs::rename(&staging, &destination) {
            Ok(()) => Error::BuildFailed {
                message: error.to_string(),
                output_dir: destination,
            },
            Err(rename_error) => Error::BuildFailed {
                message: format!(
                    "{error}; additionally failed to move {} to {}: {rename_error}",
                    staging.display(),
                    destination.display()
                ),
                output_dir: staging,
            },
        }
    }
}

fn load_completed_artifact(directory: &Path, expected_digest: &str) -> Result<Artifact> {
    let completion_path = directory.join("completion.json");
    let bytes = fs::read(&completion_path).map_err(|error| {
        Error::io(
            format!(
                "failed to read completion record {}",
                completion_path.display()
            ),
            error,
        )
    })?;
    let completion: CompletionRecord =
        serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: completion_path.clone(),
            source,
        })?;
    if completion.schema_version != COMPLETION_SCHEMA_VERSION {
        return Err(Error::InvalidManifest {
            path: completion_path.clone(),
            message: format!(
                "unsupported completion schema {}, expected {}",
                completion.schema_version, COMPLETION_SCHEMA_VERSION
            ),
        });
    }
    if completion.digest_algorithm != DIGEST_ALGORITHM {
        return Err(Error::InvalidManifest {
            path: completion_path.clone(),
            message: format!(
                "unsupported digest algorithm {:?}, expected {:?}",
                completion.digest_algorithm, DIGEST_ALGORITHM
            ),
        });
    }
    let recorded_digest = completion.cache_key.digest()?;
    if completion.cache_key_digest != expected_digest || recorded_digest != expected_digest {
        return Err(Error::InvalidManifest {
            path: completion_path,
            message: format!(
                "cache key mismatch: directory={expected_digest}, record={}, key={recorded_digest}",
                completion.cache_key_digest
            ),
        });
    }
    verify_digest(
        &directory.join("manifest.json"),
        &completion.manifest_digest,
    )?;
    let artifact = load_manifest_directory(directory, Some(expected_digest.to_owned()))?;
    if artifact.content_digests != completion.content_digests {
        return Err(Error::InvalidManifest {
            path: directory.join("completion.json"),
            message:
                "artifact or runtime-library content digests do not match the completion record"
                    .into(),
        });
    }
    if artifact.manifest().abi != completion.cache_key.abi {
        return Err(Error::InvalidManifest {
            path: directory.join("manifest.json"),
            message: format!(
                "artifact ABI {:?} does not match cache-key ABI {:?}",
                artifact.manifest().abi,
                completion.cache_key.abi
            ),
        });
    }
    Ok(artifact)
}

fn load_manifest_directory(directory: &Path, cache_digest: Option<String>) -> Result<Artifact> {
    let manifest_path = directory.join("manifest.json");
    let bytes = fs::read(&manifest_path).map_err(|error| {
        Error::io(
            format!(
                "failed to read artifact manifest {}",
                manifest_path.display()
            ),
            error,
        )
    })?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: manifest_path.clone(),
            source,
        })?;
    validate_manifest(&manifest_path, &manifest)?;
    let content_digests = capture_content_digests(directory, &manifest)?;
    Ok(Artifact {
        directory: directory.to_path_buf(),
        cache_digest,
        manifest,
        content_digests,
    })
}

fn validate_manifest(manifest_path: &Path, manifest: &ArtifactManifest) -> Result<()> {
    if manifest.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION {
        return Err(Error::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: format!(
                "unsupported schema {}, expected {}",
                manifest.schema_version, ARTIFACT_MANIFEST_SCHEMA_VERSION
            ),
        });
    }
    if manifest.entry_symbol.is_empty() || manifest.entry_symbol.contains('\0') {
        return Err(Error::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: "entry_symbol must be nonempty and contain no NUL".into(),
        });
    }
    if !manifest.artifacts.contains_key("module") {
        return Err(Error::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: "manifest has no module artifact".into(),
        });
    }
    for (name, artifact) in &manifest.artifacts {
        validate_artifact_path(manifest_path, name, artifact)?;
    }
    for runtime in &manifest.runtime_libraries {
        if !runtime.path.is_absolute() {
            return Err(Error::InvalidManifest {
                path: manifest_path.to_path_buf(),
                message: format!(
                    "runtime library path must be absolute: {}",
                    runtime.path.display()
                ),
            });
        }
    }
    Ok(())
}

fn capture_content_digests(
    directory: &Path,
    manifest: &ArtifactManifest,
) -> Result<ContentDigests> {
    let artifacts = manifest
        .artifacts
        .iter()
        .map(|(name, artifact)| Ok((name.clone(), digest_file(&directory.join(&artifact.path))?)))
        .collect::<Result<_>>()?;
    let runtime_libraries = manifest
        .runtime_libraries
        .iter()
        .map(|runtime| {
            Ok(RuntimeLibraryDigest {
                path: runtime.path.clone(),
                digest: digest_file(&runtime.path)?,
            })
        })
        .collect::<Result<_>>()?;
    Ok(ContentDigests {
        artifacts,
        runtime_libraries,
    })
}

fn validate_artifact_path(manifest_path: &Path, name: &str, artifact: &ArtifactFile) -> Result<()> {
    let mut components = artifact.path.components();
    let has_component = components.next().is_some();
    let valid = has_component
        && artifact
            .path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if !valid {
        return Err(Error::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: format!(
                "artifact {name:?} has unsafe relative path {}",
                artifact.path.display()
            ),
        });
    }
    Ok(())
}

fn sync_artifact(artifact: &Artifact) -> Result<()> {
    sync_file(&artifact.directory.join("manifest.json"))?;
    if artifact.directory.join("build.log").is_file() {
        sync_file(&artifact.directory.join("build.log"))?;
    }
    for file in artifact.manifest.artifacts.values() {
        sync_file(&artifact.directory.join(&file.path))?;
    }
    Ok(())
}

fn sync_file(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| Error::io(format!("failed to sync {}", path.display()), error))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            Error::io(
                format!("failed to sync directory {}", path.display()),
                error,
            )
        })
}

fn create_unique_directory(parent: &Path, prefix: &str) -> Result<PathBuf> {
    for _ in 0..100 {
        let path = unique_path(parent, prefix);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(Error::io(
                    format!("failed to create staging directory {}", path.display()),
                    error,
                ));
            }
        }
    }
    Err(Error::InvalidInput(format!(
        "could not allocate a unique staging directory under {}",
        parent.display()
    )))
}

fn unique_path(parent: &Path, prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        "{prefix}.{}.{}.{}",
        std::process::id(),
        timestamp,
        counter
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::model::Abi;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let path = create_unique_directory(&std::env::temp_dir(), name).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn key() -> CacheKey {
        CacheKey::new(
            "test/kernel",
            Abi::TvmFfi,
            json!({"shape": [1, 2, 3]}),
            json!({"compiler": "test"}),
        )
        .unwrap()
    }

    fn write_fake_artifact(directory: &Path, contents: &[u8]) -> Result<()> {
        let module = directory.join("module.so");
        fs::write(&module, contents)
            .map_err(|error| Error::io(format!("failed to write {}", module.display()), error))?;
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            abi: crate::Abi::TvmFfi,
            entry_symbol: "__tvm_ffi_test".into(),
            artifacts: BTreeMap::from([(
                "module".into(),
                ArtifactFile {
                    path: "module.so".into(),
                    sha256: "test-compiler-provenance".into(),
                },
            )]),
            runtime_libraries: vec![],
            tvm_ffi_runtime_version: None,
            metadata: BTreeMap::new(),
        };
        write_json_file(&directory.join("manifest.json"), &manifest)
    }

    #[test]
    fn cache_hit_does_not_rebuild() {
        let root = TestDirectory::new("cutedsl-jit-cache-hit");
        let cache = ArtifactCache::new(&root.0);
        let builds = Arc::new(AtomicU64::new(0));
        for _ in 0..2 {
            let builds = Arc::clone(&builds);
            cache
                .get_or_build(&key(), move |directory| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    write_fake_artifact(directory, b"module")
                })
                .unwrap();
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_prepare_builds_once() {
        let root = TestDirectory::new("cutedsl-jit-concurrent");
        let cache = Arc::new(ArtifactCache::new(&root.0));
        let builds = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let builds = Arc::clone(&builds);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_build(&key(), |directory| {
                        builds.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(50));
                        write_fake_artifact(directory, b"module")
                    })
                    .unwrap()
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn corrupt_artifact_is_quarantined_and_rebuilt() {
        let root = TestDirectory::new("cutedsl-jit-corrupt");
        let cache = ArtifactCache::new(&root.0);
        let builds = Arc::new(AtomicU64::new(0));
        let first = {
            let builds = Arc::clone(&builds);
            cache
                .get_or_build(&key(), move |directory| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    write_fake_artifact(directory, b"first")
                })
                .unwrap()
        };
        fs::write(first.file("module").unwrap(), b"corrupt").unwrap();
        {
            let builds = Arc::clone(&builds);
            cache
                .get_or_build(&key(), move |directory| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    write_fake_artifact(directory, b"second")
                })
                .unwrap();
        }
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(fs::read_dir(root.0.join("corrupt")).unwrap().count(), 1);
    }

    #[test]
    fn failed_build_is_preserved() {
        let root = TestDirectory::new("cutedsl-jit-failure");
        let cache = ArtifactCache::new(&root.0);
        let error = cache
            .get_or_build(&key(), |directory| {
                fs::write(directory.join("partial"), b"partial").unwrap();
                Err(Error::InvalidInput("deliberate failure".into()))
            })
            .unwrap_err();
        match error {
            Error::BuildFailed { output_dir, .. } => {
                assert!(output_dir.join("partial").is_file());
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn manifest_rejects_parent_path() {
        let root = TestDirectory::new("cutedsl-jit-unsafe-path");
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            abi: Abi::TvmFfi,
            entry_symbol: "__tvm_ffi_test".into(),
            artifacts: BTreeMap::from([(
                "module".into(),
                ArtifactFile {
                    path: "../module.so".into(),
                    sha256: "0".repeat(64),
                },
            )]),
            runtime_libraries: vec![],
            tvm_ffi_runtime_version: None,
            metadata: BTreeMap::new(),
        };
        write_json_file(&root.0.join("manifest.json"), &manifest).unwrap();
        assert!(matches!(
            Artifact::from_manifest_path(root.0.join("manifest.json")),
            Err(Error::InvalidManifest { .. })
        ));
    }

    #[test]
    fn interprocess_prepare_helper() {
        let Some(root) = std::env::var_os("CUTEDSL_JIT_TEST_PROCESS_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let cache = ArtifactCache::new(&root);
        cache
            .get_or_build(&key(), |directory| {
                let count_path = root.join("build-count");
                let mut count = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(count_path)
                    .unwrap();
                writeln!(count, "build").unwrap();
                thread::sleep(Duration::from_millis(100));
                write_fake_artifact(directory, b"module")
            })
            .unwrap();
    }

    #[test]
    fn interprocess_prepare_builds_once() {
        let root = TestDirectory::new("cutedsl-jit-process");
        let executable = std::env::current_exe().unwrap();
        let mut children = Vec::new();
        for _ in 0..4 {
            children.push(
                Command::new(&executable)
                    .args([
                        "--exact",
                        "cache::tests::interprocess_prepare_helper",
                        "--nocapture",
                    ])
                    .env("CUTEDSL_JIT_TEST_PROCESS_ROOT", &root.0)
                    .stdout(Stdio::null())
                    .spawn()
                    .unwrap(),
            );
        }
        for mut child in children {
            assert!(child.wait().unwrap().success());
        }
        let count = fs::read_to_string(root.0.join("build-count")).unwrap();
        assert_eq!(count.lines().count(), 1);
    }
}
