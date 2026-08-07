//! Native loader smoke test for a generated TVM FFI artifact.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use cutedsl_jit::{Artifact, Error as JitError, TvmModule};

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("usage: cutedsl-jit-smoke <manifest.json>"))?;
    let artifact = Artifact::from_manifest_path(manifest_path)?;
    let module = TvmModule::load(artifact)?;

    // Exercise the native safe-call boundary without allocating tensors: wrong
    // arity must return a structured TVM error before dereferencing the args.
    let mut result = [0_u64; 2];
    // SAFETY: null arguments are valid when the argument count is zero; result is
    // sixteen-byte storage for TVMFFIAny. The wrapper validates arity first.
    let error = unsafe {
        module.call_raw(
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            result.as_mut_ptr().cast(),
        )
    }
    .expect_err("TVM safe-call unexpectedly accepted an empty argument list");
    if !matches!(error, JitError::TvmCall { .. }) {
        return Err(error.into());
    }

    println!(
        "loaded {} with TVM FFI {}; safe-call error path passed",
        module.artifact().manifest().entry_symbol,
        module.version()
    );
    Ok(())
}
