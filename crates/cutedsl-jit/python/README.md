# CuTeDSL compiler environment

`prepare_environment.py` is the executable spike for an immutable, cached compiler
environment. It installs from a fully pinned and hashed requirements lock, publishes
the venv atomically, and validates pre-provisioned interpreters without mutating
them.

For the initial aarch64/CUDA 13 toolchain:

```bash
python3 crates/cutedsl-jit/python/prepare_environment.py \
  --lock crates/flashinfer-gdn-sys/shims/requirements/cu13-aarch64-py312.lock
```

Set `CUTEDSL_JIT_PYTHON` to bypass managed installation. Use `--offline` with
`--wheelhouse` for provisioned hosts without package-index access.
