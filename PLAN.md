# flashinfer-gdn-rs implementation plan

## Scope

This workspace is a deliberately narrow Rust integration for GDN kernels in
**FlashInfer 0.6.16.post2** (`v0.6.16.post2`, commit
`c498513a891d424e9ebb2518a1a3c53122dbf257`). It supports three operations:

1. same-slot single-token decode (`T=1`); and
2. checkpointed multi-token-prediction decode (`T>=2`); and
3. non-context-parallel variable-length prefill.

All three paths are implemented end to end: a pinned CuTeDSL source projection, a
content-addressed compiler cache, a typed TVM FFI boundary, safe Rust validation,
and Candle tensor adapters.

The following are out of scope for this crate's initial integration:

- FP32-state pretransposed or non-transposed decode;
- context-parallel prefill;
- split read/write state pools;
- PyTorch adapters;
- a second CuTe packed-ABI launch path; and
- direct integration into `modeld-core` or Qwen 3.5.

Those kernels can be reconsidered after the decode integration is deployed. They
must not complicate the current public API, build inputs, or test matrix.

## Workspace responsibilities

```text
crates/
├── cutedsl-jit/
│   └── Framework-independent compiler protocol, artifact cache, manifest
│       validation, dynamic loading, and TVM FFI primitives.
├── flashinfer-gdn-sys/
│   └── Pinned FlashInfer source/shim discovery, decode/prefill specialization
│       schemas, compilation, and unsafe typed entrypoints.
├── flashinfer-gdn/
│   └── Framework-independent tensor contracts, validation, prepared plans, and
│       explicit-stream launches.
└── candle-flashinfer-gdn/
    └── Unified Candle decode/prefill dispatch, CUDA tensor conversion,
        state-pool interface, and in-place state mutation.
```

`cutedsl-jit` remains generic so later CuTeDSL integrations can reuse the compiler,
cache, and loader without depending on GDN or Candle.

## Common state-pool contract

The Candle boundary is standardized around one contiguous V-major BF16 state pool:

```text
state:         [P, HV, V, K] BF16
state_indices: [B]           int32
```

The pinned kernels require `K=V=128`. `state_indices[b]` names the input state for
batch item `b`.

The same pool-plus-indices contract is retained on every architecture. BF16 decode
and SM100/SM103 prefill support pool indexing natively. SM90 and SM120/SM121
prefill only accept compact float32 `[B,...]` state, so their Candle plans own the
compact buffer and enqueue fused gather/cast, kernel, and fused scatter/cast work
on one stream. All fallback allocations occur during plan preparation.

Writable pool indices must be in range and unique for concurrently executing batch
items. The Rust layer validates tensor shapes, dtypes, contiguity, device identity,
and aliasing. Index values reside on the GPU; ownership and collision-free slot
assignment remain responsibilities of the caller.

The public integration exposes one reusable opaque `GdnHandle`, model-owned
`GdnDecode` and `GdnPrefill` runtimes, unified input structures, and opaque plans.
The first handle lazily initializes a process-wide locked Python environment;
handles own only the selected environment identity plus their artifact-cache,
source-tree, and compiler-timeout configuration.

`GdnDecode::new(&handle, &device, config)` records `H/HV/K/V`, captures the
Candle CUDA device and stream, and queries the GPU architecture, device ordinal,
and SM count once at model load. It also retains loaded modules by their effective
specialization. `GdnDecode::prepare(&x)` reads `B/T` from the input tensor's leading
two dimensions, selects `T=1` or MTP and the upstream variant internally, then
allocates fresh plan-owned dummy tensors.
`DecodePlan::forward` dispatches to the prepared backend. The only mode-specific
input is optional `[B,T]` `checkpoint_indices`, which is required for MTP and
ignored for `T=1`.

`GdnPrefill::new` records the same model/device properties and selects SM90,
SM100/SM103, or SM120/SM121 once. `prepare(&q, &cu_seqlens)` fixes `N/B` and owns
all architecture auxiliaries. `PrefillPlan::forward` always receives the BF16 pool
and int32 indices irrespective of the selected low-level state contract.

## `T=1` decode

The Candle input contract is:

```text
state:         [P, HV, 128, 128] BF16, mutable
a_log:         [HV]              float32
a:             [B, 1, HV]        BF16
dt_bias:       [HV]              float32
q:             [B, 1, H, 128]    BF16
k:             [B, 1, H, 128]    BF16
v:             [B, 1, HV, 128]   BF16
beta:          [B, 1, HV]        BF16
state_indices: [B]               int32
output:        [B, 1, HV, 128]   BF16
```

`HV` must be a positive multiple of `H`. The state selected by each index is read
and updated in the same slot.

Specialization construction reproduces FlashInfer's upstream dispatch using the
fixed batch size, `HV`, and device SM count:

- `B*HV < 512`: ILP4;
- `512 <= B*HV < 1024`: dedicated `T=1` wide-vector kernel with `tile_v=64`; and
- `B*HV >= 1024`: dedicated `T=1` wide-vector kernel with `tile_v=128`.

The ILP4 tile calculation also follows the pinned source. The selected family and
tile are part of the artifact key and are checked again when preparing a Candle
plan.

## MTP decode

MTP adds compile-time `T>=2` and checkpoint destinations:

```text
a:                  [B, T, HV]        BF16
q/k:                [B, T, H, 128]    BF16
v:                  [B, T, HV, 128]   BF16
beta:               [B, T, HV]        BF16
checkpoint_indices: [B, T]            int32
output:              [B, T, HV, 128]  BF16
```

The remaining inputs and main pool match `T=1`. For each batch item the kernel
starts from `state_indices[b]` and writes `h_1...h_T` into
`checkpoint_indices[b,0...T]` in the same pool. Checkpoint slots must be fresh,
mutually distinct, and non-overlapping with the input slots during the launch.

After speculative sampling accepts `A` tokens:

- `A=0`: keep the original `state_indices[b]` slot active;
- `A>0`: make `checkpoint_indices[b,A-1]` the active state; and
- release the unselected checkpoint slots through the caller's normal pool
  allocator.

MTP dispatch is:

- `B*HV < 128`: ILP4 with `tile_v=16`;
- `128 <= B*HV < 512`: wide-vector with `tile_v=32`;
- `512 <= B*HV < 1024`: wide-vector with `tile_v=64`; and
- `B*HV >= 1024`: wide-vector with `tile_v=128`.

`T`, the kernel family, and tile size all participate in the artifact key.

## Non-context-parallel prefill

The public Candle contract is:

```text
state:         [P, HV, 128, 128] BF16, mutable
state_indices: [B]               int32
q/k:           [N, H, 128]       BF16
v:             [N, HV, 128]      BF16
alpha:         [N, HV]           float32, multiplicative forget gate
beta:          [N, HV]           float32, update gate
cu_seqlens:    [B+1]             int32
output:        [N, HV, 128]      BF16
```

Q and K are normalized before this API; the pinned prefill kernels do not use the
Torch API's normalization flag. SM100/SM103 reads and writes the selected BF16
pool slots directly. SM90 and SM120/SM121 gather/cast those slots to compact
float32 state, convert cumulative lengths to int64, invoke the upstream kernel,
then scatter/cast final state back. Checkpoint-enabled specializations retain the
same indexed BF16 main-state contract and expose compact BF16 checkpoint rows
described by per-sequence cumulative row offsets. SM90/SM120 use plan-owned
float32 checkpoint scratch and a post-kernel cast; SM100 writes BF16 checkpoints
directly. Context-parallel prefill remains out of scope.

## Compilation and ABI

The GDN shims compile with CuTeDSL's generated TVM FFI ABI. The generated wrapper
provides the DLPack tensor boundary, validation, optional-argument representation,
explicit CUDA stream argument, and structured error reporting used by the pinned
FlashInfer implementation. The loadable artifact is a linked host module containing
the wrapper and lowered device code; a cubin alone is not a complete runtime
artifact.

Compilation runs in a short-lived Python subprocess only on an artifact-cache miss.
There is no Python or compiler work in `forward`, CUDA Graph capture, or replay.
Cache keys use the same 64-bit FNV-1a scheme as `deepgemm-rs` and include:

- the canonical specialization request;
- compiler shim and helper contents;
- the exact consumed FlashInfer kernel source;
- the locked compiler environment identity;
- host target and C compiler identity; and
- the selected ABI backend.

Cold builds use a per-key interprocess lock and publish atomically. Warm loads
validate the completion record, manifest, linked module, and external runtime
libraries without invoking Python. Invalid entries are quarantined and rebuilt.

## FlashInfer source policy

`flashinfer-gdn-sys` resolves its source tree during the dependent Cargo build:

1. `FLASHINFER_ROOT`, when explicitly set; otherwise
2. `crates/flashinfer-gdn-sys/vendor/flashinfer`, pinned to the release commit.

The reduced build requires `LICENSE`, the BF16 decode source, the SM100 chunked
prefill source and scheduler, and the SM90/SM120 delta-rule sources and directly
imported helpers. Every consumed file is included in the corresponding artifact
digest. Published crate sources need not retain Git metadata; a source checkout
that does retain metadata is checked against the pinned commit by default.

## Python dependencies

Cargo build scripts do not run pip or mutate the invoking Python environment.
`GdnHandle::new()` lazily locates, validates, or installs the locked compiler
environment on first use and stores it in a process-wide singleton. It uses
`CUTEDSL_JIT_CACHE_DIR`, then `$XDG_CACHE_HOME/cutedsl-jit`, then
`$HOME/.cache/cutedsl-jit` for both environments and artifacts. Later handles reuse
the selected interpreter and toolchain identity without rerunning the helper.
`GdnHandle::with_cache_root` can override only the artifact-cache location.

Production images may provision the same environment ahead of time:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-linux-py312.lock
```

The lock is binary-only, fully hashed, and does not install FlashInfer or PyTorch.
It contains the CuTeDSL, TVM FFI, and CUDA Python packages needed by the compiler.
`CUTEDSL_JIT_PYTHON` selects a validated pre-provisioned interpreter;
`CUTEDSL_JIT_BASE_PYTHON`, `CUTEDSL_JIT_CACHE_DIR`, `CUTEDSL_JIT_OFFLINE`, and
`CUTEDSL_JIT_WHEELHOUSE` configure managed preparation. Production images should
prepare the environment and needed artifacts during provisioning rather than on a
serving request.

## CUDA Graph lifecycle

`GdnDecode` and `GdnPrefill` are created with the model and remain alive across
forwards. Plans are created for fixed runtime shapes before graph capture. The
first effective specialization may compile or load an artifact; later plans reuse the loaded module. Every plan
allocates and owns fresh batch-sized dummy tensors. `forward` has no special
preflight, warmup, or rebinding API; the application uses the same call in eager
and capture execution. A plan must remain alive as long as a captured graph can
reference its auxiliaries. The surrounding graph owner remains responsible for its
normal warmup, stable tensor addresses, capture/replay, and recurrent-state
restoration required by its warmup policy.

## Verification and integration milestones

Completed workspace coverage includes:

- specialization validation and upstream dispatch boundary tests;
- artifact-key, cache, corruption, and loader tests in `cutedsl-jit`;
- safe Rust shape, dtype, device, and alias validation;
- eager numerical acceptance for `T=1` ILP4 and wide-vector variants using indexed
  BF16 pools; and
- eager numerical acceptance for `T=2` ILP4 and wide-vector variants, including all
  checkpoint writes; and
- cross-compilation of every prefill backend plus eager SM121 numerical acceptance
  covering output and indexed BF16 state updates.

CUDA Graph replay verification, context-parallel prefill, and additional
FlashInfer kernel families remain deferred.
