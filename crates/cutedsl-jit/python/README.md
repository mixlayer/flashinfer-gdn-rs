# CuTeDSL compiler environment

`prepare_environment.py` is the executable spike for an immutable, cached compiler
environment. It installs from a fully pinned and hashed requirements lock, publishes
the venv atomically, and validates pre-provisioned interpreters without mutating
them. Rust library adapters can call `prepare_python_environment`; a library may
cache its returned environment process-wide.

For the Linux aarch64 or x86_64 CUDA 13 toolchain:

```bash
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-linux-py312.lock
```

Set `CUTEDSL_JIT_PYTHON` to bypass managed installation. Use `--offline` with
`--wheelhouse` for direct CLI use on provisioned hosts without package-index access.
The Rust helper also recognizes `CUTEDSL_JIT_BASE_PYTHON`,
`CUTEDSL_JIT_OFFLINE`, and `CUTEDSL_JIT_WHEELHOUSE`. Its default shared cache root
is `CUTEDSL_JIT_CACHE_DIR`, `$XDG_CACHE_HOME/cutedsl-jit`, or
`$HOME/.cache/cutedsl-jit`, in that order.
