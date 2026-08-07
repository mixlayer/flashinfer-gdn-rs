#!/usr/bin/env python3
"""Compile one FlashInfer GDN non-transposed decode specialization to TVM FFI."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Any


SCHEMA_VERSION = 1
FLASHINFER_VERSION = "0.6.16.post2"
FLASHINFER_GIT_REV = "c498513a891d424e9ebb2518a1a3c53122dbf257"
ENTRY_FILE = Path("flashinfer/gdn_kernels/gdn_decode_nontranspose.py")
SUPPORTED_DTYPES = {"bfloat16", "float16"}
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
GPU_ARCH_RE = re.compile(r"sm_[0-9]+[af]?\Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_request(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        request = json.load(stream)
    if not isinstance(request, dict):
        raise ValueError("compiler request must be a JSON object")

    required = {
        "schema_version",
        "kernel",
        "symbol",
        "gpu_arch",
        "io_dtype",
        "dt_bias_dtype",
        "h",
        "hv",
        "k",
        "v",
        "t",
        "scale",
        "use_qk_l2norm",
        "batch_class",
    }
    missing = required - request.keys()
    unknown = request.keys() - required
    if missing:
        raise ValueError(f"compiler request is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"compiler request has unknown fields: {sorted(unknown)}")

    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {request['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if request["kernel"] != "gdn_decode_nontranspose":
        raise ValueError(f"unsupported kernel {request['kernel']!r}")
    if not isinstance(request["symbol"], str) or not SYMBOL_RE.fullmatch(
        request["symbol"]
    ):
        raise ValueError("symbol must be a C identifier")
    if not isinstance(request["gpu_arch"], str) or not GPU_ARCH_RE.fullmatch(
        request["gpu_arch"]
    ):
        raise ValueError("gpu_arch must look like sm_90a or sm_121a")
    if request["io_dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported io_dtype {request['io_dtype']!r}")
    if request["dt_bias_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError(f"unsupported dt_bias_dtype {request['dt_bias_dtype']!r}")
    for name in ("h", "hv", "k", "v", "t"):
        if not isinstance(request[name], int) or request[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if request["t"] != 1:
        raise ValueError("non-transposed decode only supports t=1")
    if request["hv"] < request["h"] or request["hv"] % request["h"]:
        raise ValueError("hv must be a positive multiple of h")
    if request["k"] != 128:
        raise ValueError("non-transposed decode requires k=128")
    if request["batch_class"] not in {"small", "large"}:
        raise ValueError("batch_class must be 'small' or 'large'")
    if request["batch_class"] == "small":
        if request["v"] < 128 or request["v"] % 128:
            raise ValueError("small-batch decode requires v>=128 divisible by 128")
    elif request["v"] < 32 or request["v"] % 32:
        raise ValueError("large-batch decode requires v>=32 divisible by 32")
    if (
        not isinstance(request["scale"], (int, float))
        or not math.isfinite(request["scale"])
        or request["scale"] <= 0
    ):
        raise ValueError("scale must be a positive finite number")
    if not isinstance(request["use_qk_l2norm"], bool):
        raise ValueError("use_qk_l2norm must be a boolean")
    return request


def _adapt_pool_launch_grid(selected_nodes: list[ast.stmt]) -> None:
    """Make the two launch JITs derive work from indices rather than pool capacity.

    Upstream's public helper passes a direct ``[B*HV,K,V]`` state and therefore
    aliases its first dimension to the amount of launch work. A persistent pool is
    ``[P*HV,K,V]`` instead. The device kernels already index it correctly through
    ``h0_indices``; only the launch-grid calculation needs to use ``B*HV``.
    """
    function_names = {
        "run_gdn_decode_kernel_small_batch_nontranspose",
        "run_gdn_decode_kernel_big_batch_nontranspose",
    }
    transformed: set[str] = set()
    replacement = ast.parse("h0_indices.layout.shape[0] * HV", mode="eval").body
    for statement in selected_nodes:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name not in function_names:
            continue
        matches = [
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "batch_size"
            and isinstance(node.value, ast.Name)
            and node.value.id == "batch_hv_dim"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one direct-state grid assignment in {statement.name}, "
                f"found {len(matches)}"
            )
        matches[0].value = replacement
        transformed.add(statement.name)
    if transformed != function_names:
        raise RuntimeError(
            "pinned source no longer exposes both expected non-transposed launch JITs"
        )


def _load_flashinfer_kernel(source_file: Path, generated_source_file: Path) -> Any:
    """Load a checked, Torch-free projection with a pool-aware launch grid."""
    module_name = "_flashinfer_gdn_decode_nontranspose_v0_6_16_post2"
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    selected_nodes: list[ast.stmt] = []
    found_boundary = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == (
            "_get_compiled_decode_kernel_nontranspose"
        ):
            found_boundary = True
            break
        if isinstance(node, ast.Import):
            aliases = [alias for alias in node.names if alias.name != "torch"]
            if aliases:
                node.names = aliases
                selected_nodes.append(node)
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            continue
        selected_nodes.append(node)
    if not found_boundary:
        raise RuntimeError(
            "pinned non-transposed source no longer contains the expected compiler-helper boundary"
        )
    if any(
        isinstance(node, ast.Name) and node.id == "torch"
        for statement in selected_nodes
        for node in ast.walk(statement)
    ):
        raise RuntimeError("kernel-only source prefix unexpectedly references torch")

    _adapt_pool_launch_grid(selected_nodes)

    kernel_tree = ast.Module(body=selected_nodes, type_ignores=[])
    generated_source = ast.unparse(ast.fix_missing_locations(kernel_tree)) + "\n"
    generated_source_file.write_text(generated_source, encoding="utf-8")

    module = types.ModuleType(module_name)
    module.__file__ = str(generated_source_file)
    sys.modules[module_name] = module
    exec(
        compile(generated_source, str(generated_source_file), "exec"),
        module.__dict__,
    )
    for name in (
        "run_gdn_decode_kernel_small_batch_nontranspose",
        "run_gdn_decode_kernel_big_batch_nontranspose",
    ):
        if not hasattr(module, name):
            raise RuntimeError(f"kernel-only source did not define {name}")
    return module


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


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
        "marker_sha256": _sha256(marker),
        "packages": metadata.get("packages", {}),
    }


def _compile(request: dict[str, Any], source_file: Path, object_path: Path) -> list[str]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.runtime

    module = _load_flashinfer_kernel(
        source_file, object_path.parent / "kernel_source.py"
    )
    run_func = {
        "small": module.run_gdn_decode_kernel_small_batch_nontranspose,
        "large": module.run_gdn_decode_kernel_big_batch_nontranspose,
    }[request["batch_class"]]

    io_dtype = {
        "bfloat16": cutlass.BFloat16,
        "float16": cutlass.Float16,
    }[request["io_dtype"]]
    dt_bias_dtype = {
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
    }[request["dt_bias_dtype"]]

    h = request["h"]
    hv = request["hv"]
    k = request["k"]
    v = request["v"]
    t = request["t"]
    batch = cute.sym_int64(symbol="B")
    pool_hv = cute.sym_int64(symbol="P_times_HV")
    batch_plus_one = cute.sym_int64(symbol="B_plus_one")

    def dynamic_strided(name: str, dtype: Any, shape: tuple[Any, ...]) -> Any:
        strides = tuple(
            1
            if mode == len(shape) - 1
            else cute.sym_int64(symbol=f"{name}_stride_{mode}")
            for mode in range(len(shape))
        )
        return cute.runtime.make_fake_tensor(
            dtype,
            shape,
            strides,
            assumed_align=16,
        )

    cu_seqlens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch_plus_one,), assumed_align=16
    )
    q = dynamic_strided("q", io_dtype, (batch, t, h, k))
    key = dynamic_strided("k", io_dtype, (batch, t, h, k))
    value = dynamic_strided("v", io_dtype, (batch, t, hv, v))
    a = dynamic_strided("a", io_dtype, (batch, t, hv))
    beta = dynamic_strided("b", io_dtype, (batch, t, hv))
    a_log = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (hv,), assumed_align=16
    )
    dt_bias = cute.runtime.make_fake_compact_tensor(
        dt_bias_dtype, (hv,), assumed_align=16
    )
    state = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (pool_hv, k, v),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    state_indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch,), assumed_align=16
    )
    output = dynamic_strided("o", cutlass.BFloat16, (batch, t, hv, v))
    stream = cutlass.runtime.make_fake_stream()

    compile_options = " ".join(
        [
            "--enable-tvm-ffi",
            f"--gpu-arch {request['gpu_arch']}",
            "--opt-level 3",
        ]
    )
    compiled = cute.compile(
        run_func,
        cu_seqlens,
        q,
        key,
        value,
        a,
        beta,
        a_log,
        dt_bias,
        state,
        state_indices,
        output,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=float(request["scale"]),
        T=t,
        H=h,
        HV=hv,
        K=k,
        V=v,
        use_initial_state=True,
        use_qk_l2norm=request["use_qk_l2norm"],
        stream=stream,
        options=compile_options,
    )
    compiled.export_to_c(
        str(object_path),
        function_name=request["symbol"],
        export_only_tvm_ffi_symbols=True,
    )
    return cutlass.runtime.find_runtime_libraries(enable_tvm_ffi=True)


def _link(
    cc: str, object_path: Path, module_path: Path, runtime_libraries: list[str]
) -> list[str]:
    command = [cc, "-shared", "-o", str(module_path), str(object_path)]
    command.extend(str(Path(path).resolve()) for path in runtime_libraries)
    command.append("-Wl,-z,defs")
    subprocess.run(command, check=True)
    return command


def main() -> None:
    args = _parse_args()
    request_path = args.request.resolve(strict=True)
    shim_path = Path(__file__).resolve(strict=True)
    request = _load_request(request_path)
    source_root = args.flashinfer_root.resolve(strict=True)
    source_file = source_root / ENTRY_FILE
    if not source_file.is_file():
        raise FileNotFoundError(
            f"FlashInfer {FLASHINFER_VERSION} non-transposed source is missing: "
            f"{source_file}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    object_path = output_dir / "module.o"
    module_path = output_dir / "module.so"
    manifest_path = output_dir / "manifest.json"

    runtime_libraries = _compile(request, source_file, object_path)
    link_command = _link(args.cc, object_path, module_path, runtime_libraries)
    tvm_ffi_runtime_version = _tvm_ffi_runtime_version(runtime_libraries)

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
        "schema_version": SCHEMA_VERSION,
        "abi": "tvm-ffi",
        "entry_symbol": symbol,
        "compiler_inputs": {
            "shim": {"path": shim_path.name, "sha256": _sha256(shim_path)},
            "request": {
                "path": request_path.name,
                "sha256": _sha256(request_path),
            },
        },
        "flashinfer": {
            "version": FLASHINFER_VERSION,
            "git_rev": FLASHINFER_GIT_REV,
            "source_file": str(ENTRY_FILE),
            "source_sha256": _sha256(source_file),
        },
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "packages": {
            "apache-tvm-ffi": _package_version("apache-tvm-ffi"),
            "cuda-python": _package_version("cuda-python"),
            "nvidia-cutlass-dsl": _package_version("nvidia-cutlass-dsl"),
        },
        "compiler_environment": _compiler_environment(),
        "tvm_ffi_runtime_version": tvm_ffi_runtime_version,
        "torch_required": False,
        "request": request,
        "artifacts": {
            "request": {
                "path": request_path.name,
                "sha256": _sha256(request_path),
            },
            "generated_source": {
                "path": "kernel_source.py",
                "sha256": _sha256(output_dir / "kernel_source.py"),
            },
            "object": {"path": object_path.name, "sha256": _sha256(object_path)},
            "module": {"path": module_path.name, "sha256": _sha256(module_path)},
        },
        "runtime_libraries": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path).resolve()),
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


if __name__ == "__main__":
    main()
