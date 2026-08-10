# candle-flashinfer-gdn

Candle integration for GDN kernels in FlashInfer 0.6.16.post2. It supports:

- single-token (`T=1`) decode; and
- checkpointed multi-token prediction (`T>=2`) decode; and
- non-context-parallel variable-length prefill.

Both decode operations use BF16 inputs, float32 `a_log` and `dt_bias`, and a contiguous
V-major BF16 state pool shaped `[P,HV,V,K]`. The pinned kernels require
`K=V=128`.

Prefill uses BF16 Q/K/V/output, float32 multiplicative `alpha` and update `beta`,
and int32 `cu_seqlens`. Q and K must be L2-normalized by the caller. Optional
checkpointed prefill emits compact BF16 checkpoint rows while retaining the
same indexed BF16 main-state interface.

## State-pool interface

The Candle API uses one `DecodeInputs` type for both kernels and always accepts the
main state pool plus contiguous int32 pool indices. This remains the public
contract across GPU architectures:

- `DecodeInputs::state_indices: [B]` selects each request's input state. The `T=1`
  kernel updates that same slot.
- `DecodeInputs::checkpoint_indices: Option<[B,T]>` is `None` for `T=1`. MTP
  requires it and writes `h_1` through `h_T` into the selected main-pool slots.

FlashInfer's current BF16 decode kernels implement this indirection natively, so
these indices pass directly to the compiled kernel. For prefill, SM100/SM103 also
indexes the BF16 pool natively. SM90 and SM120/SM121 use a plan-owned compact
`[B,HV,V,K]` float32 tensor. `PrefillPlan::forward` fuses pool gather with
BF16-to-F32 conversion, launches FlashInfer, then fuses F32-to-BF16 conversion
with scatter back to the selected slots.

Concurrent writes must target distinct in-range pool slots. For MTP, checkpoint
slots must be fresh, mutually distinct, and must not overlap the input slots while
the kernel is running. After sampling accepts `A` speculative tokens, retain the
input slot for `A=0`; otherwise use `checkpoint_indices[b,A-1]` as the request's
active state.

## Dispatch

`GdnDecode` and `GdnPrefill` record the CUDA device and `H/HV/K/V` model
dimensions once.
`GdnDecode::prepare(&x)` reads `B/T` from the input tensor's leading two dimensions,
chooses single-token versus MTP, and reproduces the pinned upstream dispatch
internally:

- `T=1` uses ILP4 below `B*HV=512`, then the dedicated wide-vector kernel.
- MTP uses ILP4 below `B*HV=128`, then the general wide-vector kernel.

Tile selection inside each family also follows FlashInfer and is validated when a
plan is prepared.

## Lifecycle

1. Create one `GdnHandle::new()`. It uses `CUTEDSL_JIT_CACHE_DIR`, then
   `$XDG_CACHE_HOME/cutedsl-jit`, then `$HOME/.cache/cutedsl-jit`. The first handle
   lazily resolves the process-wide locked Python environment; later handles reuse
   it. Use `GdnHandle::with_cache_root` only when the artifact cache needs an
   explicit location.
2. At model load, create `GdnDecode::new(&handle, &device, config)`. It
   captures the Candle CUDA device and queries its architecture, ordinal, and SM
   count once. Loaded modules remain cached in this object.
3. At the beginning of a model forward, call `GdnDecode::prepare(&x)` with an input
   whose leading dimensions are `[B,T,...]`. The first effective specialization
   compiles or loads its artifact; subsequent plans reuse the loaded module. Each
   plan allocates fresh dummy tensors for its batch size.
4. Call `DecodePlan::forward` with `DecodeInputs` from eager code or a graph capture
   body.
5. Keep `GdnDecode` and the plan alive for as long as a captured graph may reference
   the plan's auxiliary allocations.

There is no GDN-specific warmup or tensor-binding phase. The surrounding graph
runtime remains responsible for its normal eager warmup, capture, replay, and any
state restoration its warmup policy requires.

## Prefill lifecycle

Create `GdnPrefill` at model load. `GdnPrefill::prepare(&q, &cu_seqlens)` reads
`N` and `B`, loads the architecture backend, and allocates a fresh plan's TMA
workspace. SM90 and SM120/SM121 plans additionally allocate compact float32 state
and int64 cumulative-length scratch. All auxiliary allocations and NVRTC
compilation happen in preparation, not `forward`.

`PrefillPlan::forward` accepts the common BF16 state pool plus int32 pool indices,
returns `[N,HV,V]` BF16, and updates the selected pool slots.

Call `prepare_checkpointed(&q, &cu_seqlens, interval, C)` with a positive
multiple-of-64 interval and the caller-provided total row count. The
model-owned `GdnPrefill` caches both ordinary and checkpoint-enabled modules, so
the choice remains launch-specific. Checkpointed launches additionally require mutable compact BF16
`state_checkpoints: [C,HV,V,K]` and int32
`checkpoint_cu_starts: [B+1]`. Checkpoint rows are ordered by sequence and are
independent of the persistent pool slot namespace. The offsets begin at zero, end
at `C`, and assign `floor(sequence_length / interval)` rows to each sequence, so callers can scatter/cast
only rows selected for retention. SM90/SM120 use plan-owned float32 checkpoint
scratch and cast it to this BF16 output; SM100 writes the BF16 output directly.
Only non-context-parallel prefill is currently supported.

## Acceptance executables

The eager BF16 `T=1` acceptance executable covers the ILP4 and wide-vector dispatch
families with indexed pools:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode
```

The MTP executable covers both dispatch families at `T=2`, including every
checkpoint write:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-mtp
```

The architecture-dispatched prefill executable validates output and indexed state
updates:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin gdn-prefill

GDN_PREFILL_CHECKPOINTS=1 RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin gdn-prefill
```

The `target-cpu` setting is needed by the current Candle CPU GEMM dependency on the
aarch64 GB10 development host; it is not part of the GDN artifact cache contract.
