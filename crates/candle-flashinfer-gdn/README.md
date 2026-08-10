# candle-flashinfer-gdn

Candle integration for the prepared FlashInfer GDN kernels in this workspace. The
supported `T=1` decode operations are:

- BF16-input, float-state pretransposed `[P,HV,V,K]` pools selected by int32 `[B]`
  indices;
- BF16-input, float-state non-transposed `[P,HV,K,V]` compact pools selected by
  int32 `[B]` indices; and
- BF16-input, BF16-state `[P,HV,V,K]` compact pools selected and updated in the same
  slots by int32 `[B]` indices; and
- checkpointed BF16-state MTP over compile-time `T>=2`, returning `[B,T,HV,V]`
  while writing `h_1…h_T` to caller-selected slots in the same state pool.

Non-transposed decode compiles distinct small- (`B<32`) and large-batch (`B>=32`)
artifacts. Construct its specialization with the same fixed batch passed to
`NontransposeDecodePlan::prepare`.

Every Candle operation presents the same persistent-pool interface: `state` is a
compact pool and `state_indices` selects one input state per batch item. Backends
with native pool indexing receive those tensors directly. Backends with compact
per-batch state contracts use a graph-stable plan-owned workspace and enqueue a
device gather, the FlashInfer kernel, and a device scatter on the same CUDA stream.
Pretranspose selects between those paths from its specialization; nontranspose uses
the fallback because FlashInfer's public nontranspose API is compact-state only.
Both float-state decode adapters accept optional `[B]` write indices, defaulting to
the read indices.
Pool indices must be nonnegative, in range, and unique within a concurrent write.

BF16-state decode fixes `K=V=128` and compiles the upstream ILP4 or wide-vector
implementation selected from `B*HV` and the device SM count. Its state pool must be
compact and 32-byte aligned. The initial adapter intentionally supports same-slot
updates only; split read/write pools are a separate future kernel variant.

BF16-state MTP uses ILP4 below `B*HV=128` and the general wide-vector kernel at or
above that threshold. `T` is compile-time and part of the artifact key. The initial
adapter requires `checkpoint_indices: [B,T]`. These entries name mutually distinct,
fresh slots in the main `[P,HV,V,K]` pool and must not overlap the `[B]` input slots.
After sampling accepts `A` tokens, retain the input slot for `A=0`; otherwise select
`checkpoint_indices[b,A-1]` as the request's active state. Accepted-step fused
recovery and dense intermediate-state caching remain separate specializations.

The intended lifecycle is:

1. Create a `PretransposeDecodeSpecialization` and `PretransposeDecodeCompiler`.
2. Call `PretransposeDecodePlan::prepare` before CUDA Graph capture. This may create
   or load a JIT artifact and allocates the fixed-batch auxiliary tensors.
3. Call `forward` to allocate and return the output. Use the same call from eager
   code and from a graph capture body.
4. Let the surrounding graph runtime perform its usual eager warmup/reference
   passes, pin buffers, capture, and replay. Keep the plan and captured allocations
   alive for as long as the graph can execute.

There is no GDN-specific warmup or tensor-binding phase. For capture-capable Candle
devices, use `Device::new_cuda_with_stream` and disable Candle event tracking before
capture. `modeld-core` already does both and calls the graph module eagerly before
capture. Graph owners are also responsible for restoring recurrent state if their
warmup policy requires it.

The deterministic GPU acceptance executable exercises numerical comparison for the
compact-backend fallback and native indexing, changed input contents, repeated
graph replay, and a second CUDA stream:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin decode-vertical -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```

The `target-cpu` setting is needed by the current Candle CPU GEMM dependency on the
aarch64 GB10 development host; it is not part of the GDN artifact cache contract.

The eager-only non-transposed acceptance test covers both FlashInfer batch classes:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin nontranspose-decode -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```

The eager-only BF16-state acceptance test covers both the ILP4 and wide-vector
dispatch families with indexed pools:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```

The eager-only BF16-state MTP acceptance test covers the ILP4 and general
wide-vector dispatch families at `T=2`, including every checkpoint pool write:

```shell
RUSTFLAGS="-C target-cpu=native" \
cargo run -p candle-flashinfer-gdn --bin bf16-state-mtp -- \
  /path/to/managed-env/bin/python \
  .cutedsl-jit-cache/runtime
```
