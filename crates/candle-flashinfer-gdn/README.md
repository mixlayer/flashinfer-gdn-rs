# candle-flashinfer-gdn

Candle integration for the prepared FlashInfer GDN kernels in this workspace. The
supported operations are BF16-input, float-state `T=1` decode with either:

- pretransposed `[B,HV,V,K]` direct state or indexed state pools; or
- non-transposed `[P,HV,K,V]` compact state pools selected by int32 `[B]` indices.

Non-transposed decode compiles distinct small- (`B<32`) and large-batch (`B>=32`)
artifacts. Construct its specialization with the same fixed batch passed to
`NontransposeDecodePlan::prepare`.
The pool is passed directly to the kernel; the adapter does not gather a temporary
batch-sized state tensor.

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

The deterministic GPU acceptance executable exercises numerical comparison,
direct and indexed state, changed input contents, repeated graph replay, and a
second CUDA stream:

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
