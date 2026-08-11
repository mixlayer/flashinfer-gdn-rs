use std::io;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::Duration;

/// Error returned by the CuTeDSL artifact runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An input or cache contract was invalid.
    #[error("invalid CuTeDSL JIT input: {0}")]
    InvalidInput(String),

    /// A filesystem operation failed.
    #[error("{context}: {source}")]
    Io {
        /// Operation and path being accessed.
        context: String,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },

    /// A JSON document could not be encoded or decoded.
    #[error("failed to process JSON at {path}: {source}")]
    Json {
        /// JSON source or destination.
        path: PathBuf,
        /// Serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// A cache entry or compiler manifest violated its schema.
    #[error("invalid artifact manifest at {path}: {message}")]
    InvalidManifest {
        /// Manifest path.
        path: PathBuf,
        /// Validation failure.
        message: String,
    },

    /// A content hash did not match the manifest.
    #[error("digest mismatch for {path}: expected {expected}, found {actual}")]
    DigestMismatch {
        /// File whose content was checked.
        path: PathBuf,
        /// Manifest digest.
        expected: String,
        /// Computed digest.
        actual: String,
    },

    /// The compiler worker could not be started.
    #[error("failed to start compiler worker {program}: {source}")]
    CompilerSpawn {
        /// Worker executable.
        program: PathBuf,
        /// Spawn error.
        #[source]
        source: io::Error,
    },

    /// The compiler worker exceeded its configured deadline.
    #[error("compiler worker exceeded timeout {timeout:?}")]
    CompilerTimedOut {
        /// Configured timeout.
        timeout: Duration,
    },

    /// The compiler process returned a nonzero status.
    #[error("compiler worker exited with {status}")]
    CompilerExited {
        /// Worker exit status.
        status: ExitStatus,
    },

    /// A failed build was retained for diagnosis.
    #[error("artifact build failed: {message}; worker output retained at {output_dir}")]
    BuildFailed {
        /// Original build error.
        message: String,
        /// Preserved staging directory containing `build.log` and partial outputs.
        output_dir: PathBuf,
    },

    /// A dynamic library could not be loaded or queried.
    #[error("dynamic loading failed: {0}")]
    DynamicLoad(String),

    /// The linked TVM runtime does not match the artifact manifest.
    #[error("TVM FFI runtime version mismatch: expected {expected}, found {actual}")]
    TvmVersionMismatch {
        /// Version recorded by the compiler.
        expected: String,
        /// Version reported by the loaded C runtime.
        actual: String,
    },

    /// A TVM safe-call returned an error after its raised object was released.
    #[error("TVM FFI safe-call failed with status {status}: {kind}: {message}")]
    TvmCall {
        /// Safe-call status code.
        status: i32,
        /// TVM error kind.
        kind: String,
        /// TVM error message.
        message: String,
    },
}

impl Error {
    pub(crate) fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

/// Result type used by the CuTeDSL artifact runtime.
pub type Result<T, E = Error> = std::result::Result<T, E>;
