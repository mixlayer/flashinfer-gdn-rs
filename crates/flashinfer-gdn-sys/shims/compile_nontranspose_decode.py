#!/usr/bin/env python3
"""Compile one FlashInfer GDN non-transposed decode specialization to TVM FFI."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import sys
import types
from typing import Any

from _artifact import finalize_artifact


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


def _load_flashinfer_kernel(source_file: Path, generated_source_file: Path) -> Any:
    """Load a checked, Torch-free projection of the upstream kernel source."""
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
    batch_hv = batch * hv
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
        (batch_hv, k, v),
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

    runtime_libraries = _compile(request, source_file, object_path)
    finalize_artifact(
        cc=args.cc,
        request=request,
        request_path=request_path,
        shim_path=shim_path,
        support_path=shim_path.with_name("_artifact.py"),
        source_file=source_file,
        entry_file=ENTRY_FILE,
        output_dir=output_dir,
        object_path=object_path,
        runtime_libraries=runtime_libraries,
        schema_version=SCHEMA_VERSION,
        flashinfer_version=FLASHINFER_VERSION,
        flashinfer_git_rev=FLASHINFER_GIT_REV,
        torch_required=False,
    )


if __name__ == "__main__":
    main()
