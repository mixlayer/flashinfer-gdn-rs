# flashinfer-gdn-rs

Rust bindings and a Candle adapter for FlashInfer's gated delta net (GDN) decode
and non-context-parallel prefill kernels.

The workspace pins FlashInfer `0.6.16.post2`, compiles its CuTeDSL kernels into
loadable TVM FFI artifacts on demand, and caches those artifacts on the host.
Kernel compilation, dynamic loading, tensor validation, and Candle integration
are separated into reusable layers.

## Supported kernels

The current API supports:

- single-token decode (`T=1`), which updates each selected state-pool slot; and
- multi-token prediction (`T>=2`), which writes a checkpoint after every token;
- variable-length, non-context-parallel prefill on SM90, SM100/SM103, and
  SM120/SM121, with optional compact checkpoint emission.

Both decode modes require:

- BF16 query, key, value, gate, and state tensors;
- float32 `a_log` and `dt_bias` tensors;
- `K=V=128`; and
- a contiguous V-major state pool shaped `[P,HV,V,K]`.

Prefill accepts BF16 Q/K/V, float32 `alpha`/`beta`, int32 cumulative lengths, and
the same indexed V-major BF16 pool used by decode. Q and K must already be L2
normalized; the pinned prefill implementations do not fuse normalization.

## Workspace

- `cutedsl-jit` provides the compiler protocol, content-addressed artifact cache,
  manifest validation, dynamic loader, and TVM FFI primitives.
- `flashinfer-gdn-sys` owns the pinned FlashInfer source projection, CuTeDSL
  compiler shims, specialization schemas, and unsafe typed entrypoints.
- `flashinfer-gdn` provides framework-independent CUDA tensor contracts,
  validation, prepared plans, and explicit-stream launches.
- `candle-flashinfer-gdn` provides model-owned decode/prefill runtimes, unified
  Candle inputs, state-pool indexing, and in-place state updates.

## Requirements

The checked-in compiler lock targets Linux aarch64 and x86_64, Python 3.12, and
CUDA 13. A host C compiler is required to link generated modules. The decode
kernels target compute capability 9.0 or newer. Prefill dispatches
explicitly among SM90, SM100/SM103, and SM120/SM121.

Clone the pinned FlashInfer source submodule with the workspace:

```shell
git clone --recurse-submodules <repository-url>
cd flashinfer-gdn-rs
cargo build --workspace
```

If the repository was cloned without submodules, initialize it before building:

```shell
git submodule update --init --recursive
```

`FLASHINFER_ROOT` may be set at build time to use an explicit compatible
FlashInfer source tree instead of the Cargo-provided source.

## Runtime compiler environment

`GdnHandle::new()` lazily initializes a process-wide, locked Python compiler
environment. It validates an existing environment or installs the pinned
dependencies into a managed cache; it does not modify the invoking Python
environment.

The default cache root is selected in this order:

1. `CUTEDSL_JIT_CACHE_DIR`;
2. `$XDG_CACHE_HOME/cutedsl-jit`; or
3. `$HOME/.cache/cutedsl-jit`.

Additional controls are available:

- `FLASHINFER_GDN_RUNTIME_ROOT` selects a packaged runtime asset directory
  containing `shims/` and `vendor/flashinfer/`.
- `CUTEDSL_JIT_PYTHON` selects a pre-provisioned compatible interpreter.
- `CUTEDSL_JIT_BASE_PYTHON` selects the interpreter used to create a managed
  environment and defaults to `python3`.
- `CUTEDSL_JIT_OFFLINE=1` disables package-index access.
- `CUTEDSL_JIT_WHEELHOUSE` supplies packages for offline installation.

An environment can also be provisioned explicitly:

```shell
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-linux-py312.lock
```

## Candle lifecycle

Create a `GdnHandle` and model-owned `GdnDecode` once:

```rust
use candle_flashinfer_gdn::{GdnDecode, GdnDecodeConfig, GdnHandle};

let handle = GdnHandle::new()?;
let decode = GdnDecode::new(
    &handle,
    &device,
    GdnDecodeConfig::new(query_heads, value_heads, 128, 128),
)?;
```

`GdnDecode` stores the CUDA device, architecture, SM count, model dimensions, and
loaded kernel modules. At the beginning of a forward pass, prepare a plan from an
input tensor whose leading dimensions are `[B,T,...]`:

```rust
let plan = decode.prepare(&x)?;
```

The token dimension chooses the backend: `T=1` selects single-token decode and
`T>=2` selects MTP. Each plan allocates fresh batch-sized compatibility tensors;
loaded kernel modules remain cached in `GdnDecode`.

Run the plan with the unified input structure:

```rust
use candle_flashinfer_gdn::DecodeInputs;

let output = plan.forward(&DecodeInputs {
    state: &state,
    a_log: &a_log,
    a: &a,
    dt_bias: &dt_bias,
    q: &q,
    k: &k,
    v: &v,
    beta: &beta,
    state_indices: &state_indices,
    checkpoint_indices,
})?;
```

The plan and all input tensors must use the same Candle CUDA device. Keep a plan
alive for as long as any captured CUDA graph can reference its auxiliary
allocations.

### Prefill

Create a model-owned prefill runtime alongside the model:

```rust
use candle_flashinfer_gdn::{GdnPrefill, GdnPrefillConfig};

let prefill = GdnPrefill::new(
    &handle,
    &device,
    GdnPrefillConfig::new(query_heads, value_heads, 128, 128),
)?;
```

Prepare from the query's leading token dimension and int32 `cu_seqlens: [B+1]`,
then launch through the architecture-independent input type:

```rust
use candle_flashinfer_gdn::PrefillInputs;

let plan = prefill.prepare(&q, &cu_seqlens)?;
let output = plan.forward(&PrefillInputs {
    state: &state,
    state_indices: &state_indices,
    q: &q,
    k: &k,
    v: &v,
    alpha: &alpha,
    beta: &beta,
    cu_seqlens: &cu_seqlens,
    state_checkpoints: None,
    checkpoint_cu_starts: None,
})?;
```

The output is `[N,HV,V]` BF16 and the selected state slots are updated in place.
SM100/SM103 passes the BF16 pool and indices directly to FlashInfer. SM90 and
SM120/SM121 use plan-owned compact float32 state; `forward` performs a fused
gather/cast before the kernel and a fused scatter/cast afterward.

Checkpointed prefill is selected when preparing the launch-specific plan:

```rust
let plan = prefill.prepare_checkpointed(
    &q,
    &cu_seqlens,
    checkpoint_every_n_tokens,
    total_checkpoints,
)?;
```

For that configuration, every `forward` supplies a mutable compact BF16
`state_checkpoints: [C,HV,V,K]` tensor and int32
`checkpoint_cu_starts: [B+1]`. The main BF16 state remains the full pool selected
by `state_indices`; checkpoint rows are launch-local and are not persistent pool
slot indices. The offsets begin at zero, end at `C`, and each sequence contributes
`floor(sequence_length / checkpoint_every_n_tokens)` rows. This lets the caller scatter/cast only retained rows into its
persistent checkpoint slots. Context-parallel prefill is not currently bound.

## State-pool contract

`DecodeInputs::state` is a mutable contiguous BF16 pool shaped `[P,HV,V,K]`.
`state_indices` is contiguous int32 `[B]` and selects the input state for each
batch item.

For `T=1`, the selected slot is updated in place and `checkpoint_indices` should
be `None`.

For MTP, `checkpoint_indices` is contiguous int32 `[B,T]`. Entry `[b,t]` selects
the state-pool slot that receives request `b`'s state after token `t`. Checkpoint
slots must be in range, mutually distinct for concurrent writes, and disjoint from
the input slots while the kernel is running.

## Verification

Run the host-side workspace suite:

```shell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

On a supported CUDA host, the eager numerical acceptance executables cover both
dispatch families for each mode:

```shell
cargo run -p candle-flashinfer-gdn --bin bf16-state-decode
cargo run -p candle-flashinfer-gdn --bin bf16-state-mtp
cargo run -p candle-flashinfer-gdn --bin gdn-prefill
GDN_PREFILL_CHECKPOINTS=1 \
  cargo run -p candle-flashinfer-gdn --bin gdn-prefill
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
