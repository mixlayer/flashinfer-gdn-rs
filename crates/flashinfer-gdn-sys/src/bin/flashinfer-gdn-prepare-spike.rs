//! Compile/load driver for the first pretransposed-decode specialization.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use cutedsl_jit::TvmModule;
use flashinfer_gdn_sys::PretransposeDecodeCompiler;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let python = arguments.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::other("usage: flashinfer-gdn-prepare-spike <compiler-python> <cache-root>")
    })?;
    let cache_root = arguments.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::other("usage: flashinfer-gdn-prepare-spike <compiler-python> <cache-root>")
    })?;
    if arguments.next().is_some() {
        return Err(io::Error::other("unexpected additional arguments").into());
    }

    let compiler = PretransposeDecodeCompiler::from_managed_python(python, cache_root)?;
    let artifact = compiler.prepare()?;
    let digest = artifact.cache_digest().unwrap_or("unmanaged").to_owned();
    let directory = artifact.directory().to_path_buf();
    let module = TvmModule::load(artifact)?;
    println!(
        "prepared {} at {} with TVM FFI {}",
        digest,
        directory.display(),
        module.version()
    );
    Ok(())
}
