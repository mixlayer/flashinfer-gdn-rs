# cutedsl-jit

`cutedsl-jit` is the library-independent runtime for CuTeDSL AOT artifacts. A
kernel adapter supplies:

- a namespace, ABI, canonical JSON specialization, and compiler-toolchain identity;
- named hashes of every source, shim, lock, or other compiler input; and
- a `CompilerCommand` with one output-directory placeholder.

`ArtifactCache` hashes that contract, takes an interprocess file lock, and either
returns a validated hit or builds in a unique staging directory. Successful builds
are synced and atomically renamed into `artifacts/<digest>`. Failed workers retain
their command, combined output, and partial files under `failures/`. Invalid cache
entries are moved to `corrupt/` before rebuilding.

Each published entry contains the worker's `manifest.json`, a Rust-owned
`completion.json` with the full cache key, and the generated files. Cache keys and
content validation use the same 64-bit FNV-1a implementation and constants as
DeepGEMM's JIT runtime; this keeps debug builds fast and is identified explicitly in
the completion record. Compiler-provided SHA-256 fields remain in `manifest.json`
as provenance but are not recomputed by Rust. `TvmModule` then opens the
manifest-recorded runtime libraries, verifies the TVM C-runtime version, loads the
generated module, and owns all handles for the module's lifetime.

The crate deliberately has no FlashInfer or GDN concepts. It also does not invoke
pip: environment creation is an explicit provisioning operation, separate from
artifact lookup and steady-state launch.
