/// Error returned by framework-independent GDN validation and launch operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor descriptor or cross-tensor contract was invalid.
    #[error("invalid tensor {name:?}: {message}")]
    InvalidTensor {
        /// Argument name.
        name: &'static str,
        /// Failed contract.
        message: String,
    },

    /// The generic compiler, artifact runtime, or generated entrypoint failed.
    #[error(transparent)]
    Runtime(#[from] flashinfer_gdn_sys::JitError),
}

impl Error {
    pub(crate) fn tensor(name: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidTensor {
            name,
            message: message.into(),
        }
    }
}

/// Result type for safe GDN operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;
