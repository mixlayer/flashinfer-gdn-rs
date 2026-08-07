#![deny(unsafe_op_in_unsafe_fn)]
//! Framework-independent CuTeDSL compilation, artifact caching, and module loading.
//!
//! Library adapters provide specialization JSON, exact compiler-input digests, and
//! an out-of-process worker command. This crate supplies stable cache identities,
//! interprocess locking, failure preservation, atomic publication, artifact
//! validation, and the common TVM FFI module lifetime boundary.

mod cache;
mod command;
mod digest;
mod error;
mod model;
mod tvm;

pub use cache::{Artifact, ArtifactCache};
pub use command::CompilerCommand;
pub use error::{Error, Result};
pub use model::{
    ARTIFACT_MANIFEST_SCHEMA_VERSION, Abi, ArtifactFile, ArtifactManifest, CacheKey, HostIdentity,
    RuntimeLibrary,
};
pub use tvm::{TvmFfiVersion, TvmModule};
