"""Shared AOT linking and artifact-manifest support for GDN compiler shims."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _TvmFfiVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("patch", ctypes.c_uint32),
    ]


def _tvm_ffi_runtime_version(runtime_libraries: list[str]) -> str:
    path = next(
        Path(path).resolve()
        for path in runtime_libraries
        if Path(path).name == "libtvm_ffi.so"
    )
    library = ctypes.CDLL(str(path))
    get_version = library.TVMFFIGetVersion
    get_version.argtypes = [ctypes.POINTER(_TvmFfiVersion)]
    get_version.restype = None
    version = _TvmFfiVersion()
    get_version(ctypes.byref(version))
    return f"{version.major}.{version.minor}.{version.patch}"


def _compiler_environment() -> dict[str, Any]:
    marker = Path(sys.prefix) / "environment.json"
    if not marker.is_file():
        return {"managed": False}
    with marker.open("rb") as stream:
        metadata = json.load(stream)
    digest = metadata.get("environment_digest")
    if not isinstance(digest, str):
        raise RuntimeError(f"managed environment marker has no digest: {marker}")
    return {
        "managed": True,
        "environment_digest": digest,
        "marker_sha256": sha256(marker),
        "packages": metadata.get("packages", {}),
    }


def finalize_artifact(
    *,
    cc: str,
    request: dict[str, Any],
    request_path: Path,
    shim_path: Path,
    support_path: Path,
    source_file: Path,
    entry_file: Path,
    output_dir: Path,
    object_path: Path,
    runtime_libraries: list[str],
    schema_version: int,
    flashinfer_version: str,
    flashinfer_git_rev: str,
    torch_required: bool,
) -> None:
    """Link one exported object, verify its symbol, and publish its manifest."""
    module_path = output_dir / "module.so"
    manifest_path = output_dir / "manifest.json"
    link_command = [cc, "-shared", "-o", str(module_path), str(object_path)]
    link_command.extend(str(Path(path).resolve()) for path in runtime_libraries)
    link_command.append("-Wl,-z,defs")
    subprocess.run(link_command, check=True)

    symbol = f"__tvm_ffi_{request['symbol']}"
    nm = subprocess.run(
        ["nm", "-D", "--defined-only", str(module_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if symbol not in nm:
        raise RuntimeError(f"linked module does not export {symbol}")

    manifest = {
        "schema_version": schema_version,
        "abi": "tvm-ffi",
        "entry_symbol": symbol,
        "compiler_inputs": {
            "shim": {"path": shim_path.name, "sha256": sha256(shim_path)},
            "support": {"path": support_path.name, "sha256": sha256(support_path)},
            "request": {
                "path": request_path.name,
                "sha256": sha256(request_path),
            },
        },
        "flashinfer": {
            "version": flashinfer_version,
            "git_rev": flashinfer_git_rev,
            "source_file": str(entry_file),
            "source_sha256": sha256(source_file),
        },
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "packages": {
            "apache-tvm-ffi": importlib.metadata.version("apache-tvm-ffi"),
            "cuda-python": importlib.metadata.version("cuda-python"),
            "nvidia-cutlass-dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        },
        "compiler_environment": _compiler_environment(),
        "tvm_ffi_runtime_version": _tvm_ffi_runtime_version(runtime_libraries),
        "torch_required": torch_required,
        "request": request,
        "artifacts": {
            "request": {
                "path": request_path.name,
                "sha256": sha256(request_path),
            },
            "generated_source": {
                "path": "kernel_source.py",
                "sha256": sha256(output_dir / "kernel_source.py"),
            },
            "object": {"path": object_path.name, "sha256": sha256(object_path)},
            "module": {"path": module_path.name, "sha256": sha256(module_path)},
        },
        "runtime_libraries": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256(Path(path).resolve()),
            }
            for path in runtime_libraries
        ],
        "link_command": link_command,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "module": str(module_path)}))
