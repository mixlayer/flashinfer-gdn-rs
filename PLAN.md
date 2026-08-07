# flashinfer-gdn-rs implementation plan

## Status

The workspace, first compiler feasibility slice, and generic TVM artifact runtime
are implemented. The pinned pretransposed float-state decode kernel can be projected
from upstream source, compiled through a Rust-owned content-addressed cache, loaded
from Rust, and launched through a native CUDA smoke harness. The safe Rust GDN
operation and Candle adapter are still scaffolds; this is not yet a consumer-facing
binding.

The initial source baseline is **FlashInfer 0.6.16.post2**, Git tag
`v0.6.16.post2`, commit `c498513a891d424e9ebb2518a1a3c53122dbf257`.

The initial implementation will use CuTeDSL's generated **TVM FFI ABI** for all
FlashInfer GDN kernels. The reusable JIT layer will leave room for a CuTe packed-ABI
backend, but implementing or maintaining two GDN ABI paths is not part of the first
release.

Decode is expected to run under CUDA Graphs in common deployments. Consequently,
the design optimizes for safe compilation, stable graph capture, and cache reuse;
shaving a small amount of host-side dispatch work from each eager invocation is not
a sufficient reason to replace TVM FFI's generated tensor validation with a custom
packed ABI.

## Goals

- Bind FlashInfer's Gated Delta Network (GDN) decode and prefill kernels from Rust.
- Compile CuTeDSL shims ahead of first use for the requested specialization and GPU.
- Cache loadable native artifacts on the host and reuse them across processes.
- Keep the compilation, artifact, cache, and loading machinery reusable by other
  CuTeDSL libraries.
- Provide a framework-independent safe Rust API and a separate Candle adapter.
- Make decode APIs safe to warm up, capture, and replay with CUDA Graphs.
- Preserve upstream kernel behavior and specialization choices instead of porting
  the kernels or launch policy to Rust.

## Non-goals for the first release

- Shipping a complete matrix of precompiled kernels.
- Reimplementing CuTeDSL, FlashInfer's kernel selection, or TVM FFI.
- Calling Python or invoking a compiler from the steady-state launch path.
- Supporting PyTorch tensors directly.
- Implementing both TVM FFI and CuTe packed invocation for every GDN kernel.
- Hiding distributed setup for context-parallel prefill behind the local kernel API.

## Terminology

- **Chunked prefill** processes a sequence in bounded token chunks and carries GDN
  state between chunks.
- **CP prefill** means context-parallel prefill. It distributes sequence work across
  ranks; it is not another name for chunked prefill.
- **Specialization** is one compiled combination of kernel, GPU architecture,
  compile-time dimensions, dtypes, optional features, and relevant layout choices.
- **Artifact** is the complete loadable output and metadata for one specialization.
  With TVM FFI, the primary runtime artifact is a host shared library containing the
  generated TVM wrapper and embedded/lowered device code. A cubin alone is not
  sufficient because it does not contain the TVM host entrypoint.

## Decisions

### TVM FFI is the initial GDN ABI

FlashInfer already compiles its GDN paths with `--enable-tvm-ffi`. GDN uses dynamic
shapes and strides, optional state/checkpoint tensors, explicit streams, and several
specialization-dependent layouts. The generated TVM wrapper gives us:

- a uniform `DLTensor` representation for framework-owned tensors;
- generated dtype, device, rank, shape, stride, and alignment checks;
- explicit representation of optional arguments;
- structured error propagation; and
- behavior that follows the upstream Python implementation.

The CuTe packed ABI has a smaller raw call boundary, but its tensor structs and
argument list are specialization-specific. Using it for GDN would require generated
Rust packers plus our own exact copy of every layout constraint. That cost is not
justified by an unmeasured eager-launch saving, particularly for graph-replayed
decode.

This is an ABI decision, not a decision to expose TVM types publicly. We will keep a
narrow internal TVM layer and source raw C layouts from official TVM FFI headers or
a suitable upstream raw-bindings crate. We will not hand-maintain approximations of
the ABI.

### Streams remain explicit

Current GDN entrypoints take a `cuda.CUstream` explicitly. The Rust API will do the
same and will not depend on TVM's thread-local environment stream. This keeps launch
ordering visible and makes integration with Candle and CUDA Graph capture direct.

### The artifact cache stores linked modules

The compilation worker will export the generated host object and link it into a
loadable shared library against the matching CuTeDSL and TVM FFI runtimes. The cache
may retain cubin/PTX and generated headers for inspection, but the linked module and
manifest are the runtime contract.

### JIT is out of process

CuTeDSL is a Python compiler stack. A short-lived Python worker will perform imports,
specialization, compilation, export, and host linking. The Rust process communicates
through a versioned request/result format and never embeds Python. This isolates
compiler failures and prevents Python runtime state from entering the launch path.

### Python dependencies use an isolated managed environment

Cargo's build script will not run pip, modify the invoking Python installation, or
download a platform toolchain. Build scripts may run during cross-compilation and in
offline or sandboxed builds, and their outputs are not a suitable home for a runtime
compiler environment.

On an artifact cache miss, an explicit `prepare` operation will ensure that a pinned
compiler environment exists. The default environment lives alongside the CuTeDSL
cache, keyed by a lockfile digest, Python ABI, target platform, architecture, and CUDA
major version:

```text
$XDG_CACHE_HOME/cutedsl-jit/
├── envs/<environment-digest>/
└── artifacts/<artifact-digest>/
```

Environment installation follows these rules:

- Locate a compatible host Python executable; do not install Python itself.
- Create a new staging venv and install a platform-specific, fully pinned lock with
  `python -m pip --only-binary=:all: --require-hashes`.
- Validate imports and exact versions, write a completion manifest, and atomically
  publish the environment. A per-digest lock prevents concurrent pip installs.
- Never upgrade or repair an environment in place. A changed lock produces a new
  digest and a new immutable environment.
- Never invoke pip, Python, or the compiler from the steady-state launch path or
  while a CUDA stream is being captured.
- Preserve installer output in a log and return an actionable error on unavailable
  wheels, network failure, disk exhaustion, or version mismatch.

The lock is intentionally smaller than FlashInfer's general `requirements.txt`. We
will not install the `flashinfer-python` project because the pinned Cargo-provided
source tree is the compiler input. The first tested lock contains:

- CPython 3.12 on aarch64;
- `nvidia-cutlass-dsl[cu13]` 4.7.0;
- `apache-tvm-ffi` 0.1.13.post2;
- `cuda-python` 13.3.1; and
- fully hashed binary-only transitive dependencies.

PyTorch is not required for the first pretransposed decode specialization. The shim
parses the exact pinned upstream module, selects top-level kernel and launch-JIT
definitions before the Torch-facing compiler-helper boundary, removes the otherwise
unused import, and emits that projection as `kernel_source.py` in the artifact. It
fails if the expected boundary moves or the selected AST references Torch. Both the
full upstream file and generated projection are content-hashed in the manifest, so
the relationship remains inspectable. The resulting object was byte-identical to
one compiled after importing the full module in a Torch-bearing environment.

This is deliberately a version-specific source adapter, not a mock module. Each
remaining GDN source must be evaluated independently because some query `torch.cuda`
at module scope. If a later kernel cannot be isolated safely, Torch becomes a
separate, explicit lock variant rather than an undeclared dependency of this one.

Two deployment overrides are required:

- `CUTEDSL_JIT_PYTHON=/path/to/python` selects a pre-provisioned environment. It is
  validated but never mutated.
- An offline mode plus a wheelhouse setting allows administrators to pre-stage all
  locked wheels. Offline mode fails before compilation if the environment or a
  required wheel is missing.

A small CLI should expose the same operation as the Rust API so images and hosts can
run `prepare-toolchain` and precompile known specializations during provisioning.
Automatic preparation may be convenient in development, but production services
should normally prepare the environment and artifacts before accepting traffic.

The generated `module.so` depends on CuTeDSL and TVM FFI runtime libraries from this
environment. Their absolute locations and content digests are recorded in the
artifact manifest, and the loader opens the validated libraries before the module.
The environment therefore remains live for as long as its artifacts are used. We
will only copy those libraries into each artifact if licensing and measured
deployment needs justify the duplication.

### Feasibility slice results

The first slice was exercised on Linux aarch64, CUDA 13.0, and an SM121a device:

- the hashed lock installs 14 packages and does not include Torch;
- the managed environment occupies about 584 MB (the exploratory Torch-bearing
  environment occupied about 1.3 GB);
- CuTeDSL exported a 62 KB host object and a 70 KB linked module for one BF16-input,
  float-state, direct-state, `H=HV=16`, `K=V=128`, `T=1` specialization;
- the linked module has no RPATH/RUNPATH, so the loader must first validate and open
  the exact manifest-recorded CuTeDSL and TVM runtime libraries;
- the Python package reports `apache-tvm-ffi` 0.1.13.post2 while its C runtime reports
  ABI version 0.1.14; the manifest records both values;
- the Rust loader validated artifact/dependency digests, resolved the generated
  safe-call symbol, and exercised structured TVM error ownership; and
- a C++ harness using the official TVM FFI headers launched the generated function
  with real CUDA `DLTensor` arguments and verified zero output for zero input.

The follow-on runtime slice also established that:

- canonical keys include specialization JSON, named source/shim/lock hashes, the
  managed environment identity, host target, and host C compiler version;
- Rust-owned cache keys and completion records use DeepGEMM's 64-bit FNV-1a
  implementation and exact constants. The compiler manifest's SHA-256 fields are
  retained as provenance, but Rust does not recompute them on warm debug builds;
- cold builds run under a per-key interprocess lock, preserve combined worker output,
  publish through same-filesystem atomic rename, and retain failed staging trees;
- warm hits validate the completion record, manifest, artifact files, and external
  runtime libraries without invoking the configured Python executable;
- corrupt entries are moved to a quarantine directory and rebuilt; and
- the same cached artifact still passes the native zero-input CUDA launch.

These results establish compiler/export/link/load/launch feasibility. They do not
yet establish numerical equivalence on nonzero data, CUDA Graph safety, or a stable
public Rust API.

### First release has one GDN backend

The generic manifest identifies an ABI backend so the cache format can later support
`cute-packed`. GDN manifests will select `tvm-ffi`; there will not be a runtime ABI
fallback for the same specialization. A future packed backend should first be proven
on a small pointer/scalar-oriented kernel such as the relevant MoonEP cases.

## Workspace

```text
crates/
├── cutedsl-jit/
│   └── Framework-agnostic compiler worker protocol, artifact manifest, cache,
│       TVM module loading, and invocation primitives.
├── flashinfer-gdn-sys/
│   └── FlashInfer source/shim discovery, GDN specialization schemas, shim
│       templates, and unsafe typed entrypoint bindings.
├── flashinfer-gdn/
│   └── Framework-independent validation, launch descriptors, plans, workspace
│       requirements, and safe GDN operations.
└── candle-flashinfer-gdn/
    └── Candle CUDA tensor conversion, allocation, stream extraction, and
        ergonomic tensor APIs.
```

`cutedsl-jit` must not mention GDN or FlashInfer in its artifact model. Conversely,
kernel signatures and FlashInfer version knowledge stay out of its core cache and
loader.

## Compilation and load flow

1. The safe GDN layer validates the operation and constructs a specialization key.
2. `flashinfer-gdn-sys` renders a versioned Python shim request for that key.
3. `cutedsl-jit` canonicalizes the request and computes the cache digest.
4. On a cache hit, Rust validates the manifest and loads the shared module.
5. On a miss, Rust takes a per-digest interprocess lock and rechecks the cache.
6. A Python worker imports the pinned FlashInfer/CuTeDSL sources, creates dynamic
   tensor layouts, and invokes `cute.compile(..., options="--enable-tvm-ffi")`.
7. The worker exports the host object and device code, links a shared module, and
   reports the entrypoint plus dependency metadata.
8. Rust verifies the result and atomically publishes a completed artifact directory.
9. The sys layer resolves the TVM safe-call symbol once and holds the loaded module
   alive for all plans and graph executions that use it.

Compilation errors must preserve the worker log, command, shim digest, and artifact
staging directory path in the Rust error. A failed or interrupted build must never
look like a cache hit.

## Artifact manifest and cache key

Each artifact directory should contain at least:

```text
manifest.json
completion.json
module.so
build.log
```

Debug configurations may also retain the rendered shim, generated object, cubin,
PTX, and generated C header.

For source-projected kernels such as the first decode slice, `kernel_source.py` is
also retained and hashed. Runtime library content hashes, the TVM runtime ABI
version, and the managed environment digest are recorded even though environment
paths themselves are not portable.

Cache-key schema 2 and completion schema 2 use lowercase, 16-digit
DeepGEMM-compatible FNV-1a digests. `completion.json` names the digest algorithm and
records independent digests for the manifest, each generated artifact, and each
external runtime library. This keeps corruption detection while avoiding the poor
debug-build performance of an in-process Rust SHA-256 implementation. The Python
worker may additionally emit SHA-256 values in `manifest.json`; those are compiler
provenance rather than the Rust cache-validity contract.

The canonical key must include every input that can affect code or ABI:

- schema and manifest format versions;
- library/shim name and shim content digest;
- FlashInfer source revision or packaged-source digest;
- CuTeDSL, TVM FFI, CUDA toolkit, and host compiler versions;
- target triple, libc compatibility, and GPU compute capability;
- ABI backend and compile/export/link flags;
- kernel family and all compile-time dimensions, dtypes, feature flags, and layout
  classes; and
- hashes of any additional source files consumed by the shim.

Driver version and visible device identity should be recorded diagnostically. They
only belong in the key if testing shows that the generated artifact depends on them.

The default cache should follow platform cache conventions, with an environment
override such as `CUTEDSL_JIT_CACHE_DIR`. Publishing uses staging directories,
`fsync` where appropriate, atomic rename, and per-key file locking. Cache entries are
immutable after publication.

## Narrow TVM runtime boundary

The runtime only needs the subset required to invoke exported, void-returning GDN
functions:

- official `DLDevice`, `DLDataType`, and `DLTensor` layouts;
- official `TVMFFIAny` and type indices;
- the generated safe-call function signature;
- error extraction and reference release; and
- dynamic loading of the kernel module and matching TVM FFI runtime.

Inputs and outputs remain owned by the caller. The Candle adapter constructs borrowed
`DLTensor` views with explicit element strides. We do not create TVM Tensor objects,
use the TVM global registry, ask TVM to allocate outputs, or transfer DLPack
ownership. Scalar and stream arguments are packed directly into `TVMFFIAny` values.

The module loader must validate the TVM FFI ABI/runtime version before exposing an
entrypoint. Loaded modules are process-local and cached behind thread-safe shared
ownership.

## CUDA Graph contract

Graph support is a first-class requirement for decode:

- JIT compilation, module loading, symbol resolution, output allocation, and
  workspace allocation happen during plan creation or explicit warmup, before
  capture begins.
- A cache miss observed during stream capture returns a specific error; it never
  starts Python, takes a long-lived build lock, or invokes a compiler.
- The caller supplies the capture stream explicitly, and every launch uses it.
- Launches perform no device synchronization, implicit allocation, logging, or file
  I/O.
- Kernel modules and plan-owned metadata remain alive at least as long as any graph
  executable that references their kernel nodes.
- Tensor and workspace addresses used by a captured graph remain stable across
  replay unless the higher-level graph owner explicitly updates graph parameters.
- Host-side `DLTensor` descriptors may be temporary: CUDA capture records the values
  passed to the resulting kernel launches. Device buffers and loaded modules are the
  objects whose lifetimes must span replay.
- Plans are associated with a CUDA device and specialization. Reusing a plan on a
  different device is rejected.
- APIs provide an explicit `prepare`/warmup path so applications can compile every
  required batch/shape variant before capture.

Graph tests must exercise capture followed by multiple replays with changed input
contents, optional state pools, and more than one CUDA stream. We should also verify
that a cold specialization fails predictably when requested inside capture.

## Initial GDN kernel scope

The implementation order is a vertical slice followed by breadth:

1. Pretransposed decode with float state, including indexed state pools.
2. Non-transposed decode.
3. BF16-state decode and multi-token prediction (MTP) variants.
4. Chunked prefill for the supported SM90, SM100, and SM120 paths.
5. Context-parallel prefill for SM90 and SM120. FlashInfer 0.6.16.post2 does not
   provide an SM100 CP kernel, even though its ordinary chunked prefill supports
   SM100.

Each operation starts from the exact upstream Python entrypoint. Its Rust
specialization schema must distinguish compile-time values from TVM-validated
runtime values. Optional features that change generated code produce distinct cache
entries.

## Safe API direction

`flashinfer-gdn` will accept framework-independent tensor descriptors containing a
CUDA pointer, device, dtype, shape, and element strides. Its responsibilities are:

- cross-tensor shape and device validation not already expressible in one TVM
  argument;
- architecture and specialization selection;
- output and workspace layout calculation;
- plan preparation and warmup;
- module lifetime and graph-safety invariants; and
- forwarding a caller-provided CUDA stream.

`candle-flashinfer-gdn` will verify CUDA storage and contiguous/strided layouts,
convert Candle offsets into the correct device pointers, allocate outputs and
workspace outside capture, and translate errors into Candle errors. It must not
duplicate kernel selection or cache policy.

## Source and version policy

`flashinfer-gdn-sys` pins FlashInfer 0.6.16.post2 at commit
`c498513a891d424e9ebb2518a1a3c53122dbf257`. Its crate source package will contain
the required FlashInfer source tree under `vendor/flashinfer`, using the exact pinned
Git submodule during development and real packaged file contents in a Cargo
publication. The runtime does not clone FlashInfer or fetch its source over the
network.

This mirrors the DeepGEMM source handoff:

1. `flashinfer-gdn-sys/build.rs` starts from `CARGO_MANIFEST_DIR`, which points at the
   copy Cargo already checked out for the dependent build.
2. Unless `FLASHINFER_ROOT` is set for development, the selected root is
   `CARGO_MANIFEST_DIR/vendor/flashinfer`.
3. The build script canonicalizes the root, checks the expected release metadata and
   required GDN files, and emits both `cargo:rustc-env` and Cargo metadata containing
   the selected path.
4. `flashinfer-gdn-sys` exposes the compiled-in root as the runtime default. A runtime
   `FLASHINFER_ROOT` override is accepted after the same validation, primarily for
   source development.
5. The compiler worker receives the selected root explicitly and loads the required
   modules without installing the FlashInfer package.

The baked path is appropriate when Cargo builds and runs the dependent on the same
host or container. A binary copied elsewhere must either carry the source tree and
set `FLASHINFER_ROOT`, or ship with all required specializations precompiled. This is
the same portability boundary as any runtime compiler that consumes Cargo-vendored
source.

Artifact manifests always record the release, commit, and actual source-tree digest;
an override with modified files cannot alias the official-release cache entry.
CuTeDSL, TVM FFI, and CUDA-facing Python packages are pinned as a tested toolchain
rather than discovered independently from arbitrary Python environments. Any future
Torch-bearing lock is a separate environment variant.

## Testing and acceptance gates

### CPU-only tests

The cache-key ordering, thread/process contention, worker success/nonzero/timeout,
failure preservation, corruption quarantine/recovery, and atomic hit paths now have
CPU tests. The remaining CPU gates are:

- canonical manifest fixtures that remain stable across schema evolution;
- TVM/DLPack layout and type-index checks against official headers;
- GDN specialization-key completeness; and
- tensor validation and output/workspace layout calculations.

### GPU tests

- build and load one artifact for every supported kernel family and architecture;
- numerical comparison with the upstream FlashInfer Python APIs;
- variable batch, sequence, state-pool indexing, and non-compact supported layouts;
- optional argument combinations and expected validation failures;
- concurrent first use from multiple threads and processes;
- warm cache operation with Python unavailable;
- CUDA Graph capture and repeated replay; and
- module/cache lifetime behavior while graphs remain alive.

### ABI decision benchmark

Benchmark eager TVM safe-call overhead and captured/replayed decode separately. The
packed ABI should only be reconsidered for GDN if it produces a material end-to-end
benefit after graph replay and its generated validation/packing burden is accounted
for. The benchmark is an acceptance measurement, not a prerequisite for starting
with the upstream TVM ABI.

## Milestones

1. **Workspace and contracts (complete)** — crate scaffolding, this plan, pinned
   source metadata, first manifest schema, environment lock, and worker request.
2. **Generic TVM artifact runtime (complete)** — canonical keys, interprocess cache,
   subprocess timeout/logging, failure and corruption retention, AOT export/link,
   atomic publication, manifest validation, and the TVM module loader are proven by
   CPU tests and the GDN GPU smoke path.
3. **Decode vertical slice** — pretransposed decode, Candle integration, numerical
   tests, explicit warmup, and CUDA Graph replay.
4. **Decode coverage** — non-transposed, BF16-state, and MTP variants.
5. **Prefill coverage** — chunked prefill followed by context-parallel prefill.
6. **Hardening** — supported version matrix, reproducible source distribution,
   concurrent cache tests, diagnostics, examples, and benchmarks.
7. **Reuse validation** — compile a representative MoonEP kernel through the same
   artifact/cache pipeline and decide whether to implement `cute-packed` as a second
   backend.

## Open questions

- Should raw TVM FFI definitions come from an upstream Rust sys crate or bindgen
  output checked against the pinned official headers?
- What exact package versions form the first CUDA 12 lock, and which architectures
  need it? The first CUDA 13 aarch64/Python 3.12 lock is established.
- Can every remaining GDN module be isolated from Torch as safely as pretransposed
  decode, especially modules that query `torch.cuda` at module scope?
- What exact dynamic-layout classes should be shared between decode specializations
  without causing unnecessary recompilation?
- Which component owns persistent decode workspaces when Candle CUDA Graph helpers
  also manage capture pools?
- Do SM90, SM100, and SM120 require separate worker environments or can one pinned
  CuTeDSL installation produce the full supported matrix?
