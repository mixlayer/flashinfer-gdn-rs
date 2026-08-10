# GDN CuTeDSL shims

These versioned Python shims load the pinned FlashInfer 0.6.16.post2 implementation
from the Cargo-provided source tree, construct specialization inputs, compile with
`--enable-tvm-ffi`, and export a loadable AOT artifact. They do not install
FlashInfer or assume it is present in site-packages.

The implemented requests cover pretransposed and non-transposed float-state decode,
plus same-slot BF16-state T=1 and checkpointed `T>=2` MTP decode. Each compiler parses
the exact upstream file and emits an auditable, Torch-free kernel projection into
the artifact before invoking CuTeDSL. The non-transposed request records whether
FlashInfer's small- or large-batch implementation is compiled and preserves its
compact per-batch state contract without changing the upstream launch logic.
BF16-state requests record compile-time `T`, the upstream-selected
ILP4, dedicated T=1 wide-vector, or general MTP wide-vector implementation, tile
size, packed-FMA choice, and compact-pool per-token scatter mode. These source
adapters are specific to the pinned FlashInfer revision.

From the workspace root, prepare the current Linux aarch64/Python 3.12/CUDA 13
environment with:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-aarch64-py312.lock \
  --cache-root .cutedsl-jit-cache/compiler
```

Use the `python` path printed by that command with the Rust-owned cache path:

```shell
COMPILER_PYTHON=.cutedsl-jit-cache/compiler/envs/<environment-digest>/bin/python
cargo run -p flashinfer-gdn-sys --bin flashinfer-gdn-prepare-spike -- \
  "$COMPILER_PYTHON" .cutedsl-jit-cache/runtime
```

This computes a canonical key from the request, shim, requirements lock, upstream
kernel source, compiler environment, host target, and C compiler. A cold miss is
compiled in a staging directory and atomically published; later calls validate and
load the same entry without invoking Python. Worker output is stored in `build.log`.

The underlying compiler and launch harness can still be invoked directly for
development:

```shell
COMPILER_PYTHON=.cutedsl-jit-cache/compiler/envs/<environment-digest>/bin/python
ARTIFACT_DIR=.cutedsl-jit-cache/artifacts/pretranspose-sm121

"$COMPILER_PYTHON" \
  crates/flashinfer-gdn-sys/shims/compile_pretranspose_decode.py \
  --request crates/flashinfer-gdn-sys/shims/requests/pretranspose_decode_sm121_bf16.json \
  --flashinfer-root crates/flashinfer-gdn-sys/vendor/flashinfer \
  --output-dir "$ARTIFACT_DIR"

cargo run -p cutedsl-jit --bin cutedsl-jit-smoke -- \
  "$ARTIFACT_DIR/manifest.json"

"$COMPILER_PYTHON" \
  crates/flashinfer-gdn-sys/shims/run_native_smoke.py \
  --manifest "$ARTIFACT_DIR/manifest.json"
```

The Rust executable tests digest validation, dynamic loading, symbol resolution, and
the structured TVM error path. The native harness additionally launches the kernel
with real CUDA tensors and verifies the zero-input result.

Shim and shared artifact-helper contents, the consumed upstream source file, the
exact request and environment lock, the generated kernel source, and the host linker
identity all participate in the artifact contract. Rust plans generate the decode
requests and validate their runtime tensor contracts.
