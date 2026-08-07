#!/usr/bin/env python3
"""Build and run the native pretranspose-decode TVM FFI smoke harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cuda-home", type=Path)
    return parser.parse_args()


def _cuda_home(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(value) if (value := os.environ.get("CUDA_HOME")) else None,
        Path(value) if (value := os.environ.get("CUDA_PATH")) else None,
        Path("/usr/local/cuda"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "include/cuda_runtime_api.h").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not find CUDA_HOME containing cuda_runtime_api.h")


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    artifact_dir = manifest_path.parent
    with manifest_path.open("rb") as stream:
        manifest = json.load(stream)

    import tvm_ffi

    tvm_include = Path(tvm_ffi.__file__).resolve().parent / "include"
    if not (tvm_include / "tvm/ffi/c_api.h").is_file():
        raise FileNotFoundError(f"TVM FFI headers are missing under {tvm_include}")
    cuda_home = _cuda_home(args.cuda_home)
    source = (
        Path(__file__).resolve().parent.parent
        / "native/pretranspose_decode_smoke.cc"
    )
    executable = artifact_dir / "pretranspose_decode_smoke"
    cuda_libdir = cuda_home / "lib64"
    if not cuda_libdir.is_dir():
        target_libdirs = sorted((cuda_home / "targets").glob("*/lib"))
        if not target_libdirs:
            raise FileNotFoundError(f"could not find CUDA libraries under {cuda_home}")
        cuda_libdir = target_libdirs[0]

    command = [
        args.cxx,
        "-std=c++17",
        "-O2",
        f"-I{tvm_include}",
        f"-I{cuda_home / 'include'}",
        str(source),
        f"-L{cuda_libdir}",
        f"-Wl,-rpath,{cuda_libdir}",
        "-lcudart",
        "-ldl",
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True)

    runtime_libraries = [item["path"] for item in manifest["runtime_libraries"]]
    cute_runtime = next(
        path for path in runtime_libraries if Path(path).name == "libcute_dsl_runtime.so"
    )
    tvm_runtime = next(
        path for path in runtime_libraries if Path(path).name == "libtvm_ffi.so"
    )
    module = artifact_dir / manifest["artifacts"]["module"]["path"]
    subprocess.run(
        [
            str(executable),
            str(module),
            manifest["entry_symbol"],
            cute_runtime,
            tvm_runtime,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
