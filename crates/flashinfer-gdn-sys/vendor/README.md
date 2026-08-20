# Vendored FlashInfer source

The pinned FlashInfer source tree is checked out at `vendor/flashinfer`.

Initial pin:

- Release: `0.6.16.post2`
- Git tag: `v0.6.16.post2`
- Commit: `c498513a891d424e9ebb2518a1a3c53122dbf257`

The checkout is recorded as a Git submodule at the exact revision above. Cargo
packages include its actual source files, not the submodule's Git metadata, so a
dependent's Cargo checkout is sufficient at runtime. The worker must not clone
FlashInfer or install `flashinfer-python` with pip.

`FLASHINFER_ROOT` remains available as an explicit build-time development override.
Relocated binaries use `FLASHINFER_GDN_RUNTIME_ROOT` to find both this source tree
and the compiler shims.
