# candle-flashinfer-gdn

Candle integration for the BF16-state GDN decode kernels in FlashInfer
0.6.16.post2. This crate intentionally supports only:

- single-token (`T=1`) decode; and
- checkpointed multi-token prediction (`T>=2`) decode.

Both operations use BF16 inputs, float32 `a_log` and `dt_bias`, and a contiguous
V-major BF16 state pool shaped `[P,HV,V,K]`. The pinned kernels require
`K=V=128`.

## State-pool interface

The Candle API uses one `DecodeInputs` type for both kernels and always accepts the
main state pool plus contiguous int32 pool indices. This remains the public
contract across GPU architectures:

- `DecodeInputs::state_indices: [B]` selects each request's input state. The `T=1`
  kernel updates that same slot.
- `DecodeInputs::checkpoint_indices: Option<[B,T]>` is `None` for `T=1`. MTP
  requires it and writes `h_1` through `h_T` into the selected main-pool slots.

FlashInfer's current BF16 kernels implement this indirection natively, so these
indices pass directly to the compiled kernel. The crate retains a graph-stable
gather/scatter adapter for a future architecture-specific backend that only accepts
compact per-batch state. Such a backend can therefore preserve the same public
pool-plus-indices signature without allocating during a launch.

Concurrent writes must target distinct in-range pool slots. For MTP, checkpoint
slots must be fresh, mutually distinct, and must not overlap the input slots while
the kernel is running. After sampling accepts `A` speculative tokens, retain the
input slot for `A=0`; otherwise use `checkpoint_indices[b,A-1]` as the request's
active state.

## Dispatch

Construct the appropriate specialization using the fixed runtime batch size before
preparing its plan. `T` is part of the MTP specialization and artifact cache key.
The Rust specialization reproduces the pinned upstream dispatch:

- `T=1` uses ILP4 below `B*HV=512`, then the dedicated wide-vector kernel.
- MTP uses ILP4 below `B*HV=128`, then the general wide-vector kernel.

Tile selection inside each family also follows FlashInfer and is validated when a
plan is prepared.

## Lifecycle

1. Create a `Bf16StateDecodeSpecialization` or `Bf16StateMtpSpecialization` for
   the device architecture and fixed batch.
2. Create the matching compiler and pass it to `DecodePlan::prepare` before CUDA
   Graph capture. The compiler type selects the `T=1` or MTP backend. Preparation
   compiles or loads the content-addressed artifact and allocates fixed auxiliary
   tensors.
3. Call `DecodePlan::forward` with `DecodeInputs` from eager code or a graph capture
   body.
4. Keep the plan and captured allocations alive for as long as the graph may run.

There is no GDN-specific warmup or tensor-binding phase. The surrounding graph
runtime remains responsible for its normal eager warmup, capture, replay, and any
state restoration its warmup policy requires.

## Acceptance executables

The eager BF16 `T=1` acceptance executable covers the ILP4 and wide-vector dispatch
families with indexed pools:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```

The MTP executable covers both dispatch families at `T=2`, including every
checkpoint write:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-mtp -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```

The `target-cpu` setting is needed by the current Candle CPU GEMM dependency on the
aarch64 GB10 development host; it is not part of the GDN artifact cache contract.
