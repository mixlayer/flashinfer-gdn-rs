# flashinfer-gdn-rs implementation plan

## Scope

This workspace is a deliberately narrow Rust integration for the BF16-state GDN
decode kernels in **FlashInfer 0.6.16.post2** (`v0.6.16.post2`, commit
`c498513a891d424e9ebb2518a1a3c53122dbf257`). It supports two operations:

1. same-slot single-token decode (`T=1`); and
2. checkpointed multi-token-prediction decode (`T>=2`).

Both paths are implemented end to end: a pinned CuTeDSL source projection, a
content-addressed compiler cache, a typed TVM FFI boundary, safe Rust validation,
and Candle tensor adapters.

The following are out of scope for this crate's initial integration:

- FP32-state pretransposed or non-transposed decode;
- chunked or context-parallel prefill;
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
│   └── Pinned FlashInfer source/shim discovery, BF16 decode specialization
│       schemas, compilation, and unsafe typed entrypoints.
├── flashinfer-gdn/
│   └── Framework-independent tensor contracts, validation, prepared plans, and
│       explicit-stream launches.
└── candle-flashinfer-gdn/
    └── Unified Candle decode dispatch, CUDA tensor conversion, state-pool
        interface, and in-place state mutation for T=1 and MTP.
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

The same pool-plus-indices contract is retained on every architecture. The current
BF16 FlashInfer kernels support pool indexing natively, so the adapter passes the
pool and indices through without a preliminary copy. The Candle crate also retains
a private, plan-owned gather/scatter workspace for a future backend that accepts
only compact `[B,...]` state. That fallback allocates at plan preparation time and
enqueues gather, kernel, and scatter operations on the plan's CUDA stream, keeping
launches graph-capturable and leaving the public input shape unchanged.

Writable pool indices must be in range and unique for concurrently executing batch
items. The Rust layer validates tensor shapes, dtypes, contiguity, device identity,
and aliasing. Index values reside on the GPU; ownership and collision-free slot
assignment remain responsibilities of the caller.

The Candle crate exposes one `DecodeInputs` structure and one opaque `DecodePlan`.
Passing a `Bf16StateDecodeCompiler` or `Bf16StateMtpCompiler` to
`DecodePlan::prepare` selects the backend. `DecodePlan::forward` dispatches to the
prepared backend. The only mode-specific input is optional `[B,T]`
`checkpoint_indices`, which is required for MTP and ignored for `T=1`.

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

Only `LICENSE` and `flashinfer/gdn_kernels/gdn_decode_bf16_state.py` are required by
the reduced build. The consumed source file is included in the artifact digest, and
the shim checks its expected extraction boundaries. Published crate sources need
not retain Git metadata; a source checkout that does retain metadata is checked
against the pinned commit by default.

## Python dependencies

Cargo build scripts do not run pip or mutate the invoking Python environment. The
compiler environment is provisioned explicitly before preparing artifacts:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-aarch64-py312.lock \
  --cache-root .cutedsl-jit-cache/compiler
```

The lock is binary-only, fully hashed, and does not install FlashInfer or PyTorch.
It contains the CuTeDSL, TVM FFI, and CUDA Python packages needed by the compiler.
Rust receives the resulting managed Python path and validates its immutable
`environment.json` identity. Production images should prepare the environment and
needed artifacts during provisioning rather than on a serving request.

## CUDA Graph lifecycle

Plans are created for a fixed batch before graph capture. Preparation may compile
or load an artifact and allocates any auxiliary tensors. `forward` has no special
preflight, warmup, or rebinding API; the application uses the same call in eager and
capture execution. The surrounding graph owner is responsible for its normal
warmup, stable tensor addresses, capture/replay, and any recurrent-state restoration
required by its warmup policy.

## Verification and integration milestones

Completed workspace coverage includes:

- specialization validation and upstream dispatch boundary tests;
- artifact-key, cache, corruption, and loader tests in `cutedsl-jit`;
- safe Rust shape, dtype, device, and alias validation;
- eager numerical acceptance for `T=1` ILP4 and wide-vector variants using indexed
  BF16 pools; and
- eager numerical acceptance for `T=2` ILP4 and wide-vector variants, including all
  checkpoint writes.

The next integration milestone is to wire the two Candle plans into the consumer's
decode path, using its existing state allocator and CUDA Graph lifecycle. CUDA Graph
replay verification and support for additional FlashInfer kernels are intentionally
deferred until that integration is operational.
