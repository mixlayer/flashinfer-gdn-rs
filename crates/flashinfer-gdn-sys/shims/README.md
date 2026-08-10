# GDN CuTeDSL shims

The versioned compiler shim loads the pinned FlashInfer 0.6.16.post2 BF16-state
decode implementation from the source tree supplied by Cargo, constructs the exact
specialization, compiles with `--enable-tvm-ffi`, and exports a loadable AOT
artifact. It neither installs FlashInfer nor imports a copy from site-packages.

`compile_bf16_state_decode.py` covers both supported requests:

- same-slot single-token decode (`T=1`); and
- checkpointed MTP decode (`T>=2`) with every post-token state scattered into a
  caller-selected slot in the main state pool.

The request records compile-time `T`, dimensions, GPU architecture, upstream
kernel family, tile size, packed-FMA selection, and checkpoint behavior. The shim
extracts an auditable Torch-free projection from
`flashinfer/gdn_kernels/gdn_decode_bf16_state.py` before invoking CuTeDSL. This
adapter is intentionally specific to the pinned FlashInfer source revision.

## Compiler environment

`GdnHandle::new()` lazily initializes one process-wide compiler environment and
uses the common CuTeDSL cache root for artifacts. Resolution is
`CUTEDSL_JIT_CACHE_DIR`, then `$XDG_CACHE_HOME/cutedsl-jit`, then
`$HOME/.cache/cutedsl-jit`. On the first call it runs the embedded preparation
helper against the locked Linux aarch64/Python 3.12/CUDA 13 requirements. The
helper validates an existing immutable environment or installs and atomically
publishes it under that cache. Neither a Python path nor a cache path is required
by the public Rust API. `GdnHandle::with_cache_root` remains available when only
the artifact cache needs an explicit location.

For production image provisioning, the same environment can still be prepared
ahead of time from the workspace root:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-aarch64-py312.lock
```

`CUTEDSL_JIT_PYTHON` selects a pre-provisioned interpreter and validates it without
modification. `CUTEDSL_JIT_BASE_PYTHON` selects the bootstrap interpreter used for
managed installation; it defaults to `python3`. `CUTEDSL_JIT_CACHE_DIR` controls
the environment cache. Offline hosts can set `CUTEDSL_JIT_OFFLINE=1` and
`CUTEDSL_JIT_WHEELHOUSE=/path/to/wheels`.

The Candle acceptance executable uses the same default cache:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode
```

## Artifact contract

The Rust handle generates each specialization request and computes a canonical key
from the request, shim and support files, requirements lock, consumed FlashInfer
source, compiler environment, host target, and host C compiler. A cold miss is
built in a staging directory and atomically published. A warm hit validates and
loads the same entry without invoking Python. Compiler output is retained in
`build.log`.

The generated `module.so`, manifest, projected kernel source, and external runtime
library identities form the load contract. The runtime uses the generated TVM FFI
entrypoint; a cubin by itself is not sufficient because it lacks the host wrapper.
