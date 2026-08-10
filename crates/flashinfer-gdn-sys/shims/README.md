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

From the workspace root, prepare the tested Linux aarch64/Python 3.12/CUDA 13
environment with:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-aarch64-py312.lock \
  --cache-root .cutedsl-jit-cache/compiler
```

Use the Python path printed by that command when constructing either Rust compiler.
For example, the Candle acceptance executables accept the path as their first
argument:

```shell
COMPILER_PYTHON=.cutedsl-jit-cache/compiler/envs/<environment-digest>/bin/python

RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode -- \
  "$COMPILER_PYTHON" .cutedsl-jit-cache/runtime
```

## Artifact contract

The Rust compilers generate the JSON request and compute a canonical key from the
request, shim and support files, requirements lock, consumed FlashInfer source,
compiler environment, host target, and host C compiler. A cold miss is built in a
staging directory and atomically published. A warm hit validates and loads the same
entry without invoking Python. Compiler output is retained in `build.log`.

The generated `module.so`, manifest, projected kernel source, and external runtime
library identities form the load contract. The runtime uses the generated TVM FFI
entrypoint; a cubin by itself is not sufficient because it lacks the host wrapper.
