#!/usr/bin/env python3
"""Compile one FlashInfer BF16-state decode or MTP specialization to TVM FFI."""

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
ENTRY_FILE = Path("flashinfer/gdn_kernels/gdn_decode_bf16_state.py")
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
        "variant",
        "tile_v",
        "use_packed_fma",
        "same_pool",
    }
    optional = {
        "per_token_pool_scatter",
        "per_token_pool_scatter_flat",
    }
    missing = required - request.keys()
    unknown = request.keys() - required - optional
    if missing:
        raise ValueError(f"compiler request is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"compiler request has unknown fields: {sorted(unknown)}")

    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {request['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if request["kernel"] == "gdn_decode_bf16_state_t1":
        if not isinstance(request["t"], int) or request["t"] != 1:
            raise ValueError("BF16-state T=1 decode requires t=1")
        valid_variants = {"ilp4", "wide_vec_t1"}
        if request.get("per_token_pool_scatter", False) or request.get(
            "per_token_pool_scatter_flat", False
        ):
            raise ValueError("BF16-state T=1 decode does not use pool scatter")
    elif request["kernel"] == "gdn_decode_bf16_state_mtp":
        if not isinstance(request["t"], int) or request["t"] < 2:
            raise ValueError("BF16-state MTP requires t>=2")
        valid_variants = {"ilp4", "wide_vec"}
        if request.get("per_token_pool_scatter") is not True:
            raise ValueError("BF16-state MTP requires per-token pool scatter")
        if request.get("per_token_pool_scatter_flat") is not True:
            raise ValueError("compact BF16-state MTP requires flat pool scatter")
    else:
        raise ValueError(f"unsupported kernel {request['kernel']!r}")
    if not isinstance(request["symbol"], str) or not SYMBOL_RE.fullmatch(
        request["symbol"]
    ):
        raise ValueError("symbol must be a C identifier")
    if not isinstance(request["gpu_arch"], str) or not GPU_ARCH_RE.fullmatch(
        request["gpu_arch"]
    ):
        raise ValueError("gpu_arch must look like sm_90a or sm_121a")
    if request["io_dtype"] != "bfloat16":
        raise ValueError("BF16-state decode requires bfloat16 I/O")
    if request["dt_bias_dtype"] != "float32":
        raise ValueError("BF16-state decode requires float32 dt_bias")
    for name in ("h", "hv", "k", "v", "t", "tile_v"):
        if not isinstance(request[name], int) or request[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if request["k"] != 128 or request["v"] != 128:
        raise ValueError("BF16-state decode requires k=v=128")
    if request["hv"] < request["h"] or request["hv"] % request["h"]:
        raise ValueError("hv must be a positive multiple of h")
    if request["variant"] not in valid_variants:
        raise ValueError(
            f"unsupported variant {request['variant']!r} for {request['kernel']}"
        )
    if request["variant"] == "ilp4":
        valid_tiles = {16, 32, 64, 128}
    elif request["variant"] == "wide_vec_t1":
        valid_tiles = {64, 128}
    else:
        valid_tiles = {32, 64, 128}
    if request["tile_v"] not in valid_tiles:
        raise ValueError(
            f"invalid tile_v {request['tile_v']} for {request['variant']}"
        )
    if (
        not isinstance(request["scale"], (int, float))
        or not math.isfinite(request["scale"])
        or request["scale"] <= 0
    ):
        raise ValueError("scale must be a positive finite number")
    for name in (
        "use_qk_l2norm",
        "use_packed_fma",
        "same_pool",
        "per_token_pool_scatter",
        "per_token_pool_scatter_flat",
    ):
        if name not in request:
            continue
        if not isinstance(request[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if not request["same_pool"]:
        raise ValueError("the initial BF16-state slice requires same_pool=true")
    return request


def _is_num_sms_boundary(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "NUM_SMS"
    )


def _load_flashinfer_kernel(source_file: Path, generated_source_file: Path) -> Any:
    """Load the kernel/JIT prefix without Torch-facing marking or dispatch code."""
    module_name = "_flashinfer_gdn_decode_bf16_state_v0_6_16_post2"
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    selected_nodes: list[ast.stmt] = []
    found_boundary = False
    skipped_helpers = {
        "_mark_batch_dynamic",
        "_mark_slot_dynamic",
        "_mark_index_dynamic",
    }
    for node in tree.body:
        if _is_num_sms_boundary(node):
            found_boundary = True
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in skipped_helpers:
                continue
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
            "pinned BF16-state source no longer contains the NUM_SMS boundary"
        )
    if any(
        isinstance(node, ast.Name) and node.id == "torch"
        for statement in selected_nodes
        for node in ast.walk(statement)
    ):
        raise RuntimeError("BF16-state kernel projection unexpectedly references torch")

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
        "run_gdn_decode_bf16state_mtp_ilp4",
        "_run_wide_vec",
        "_run_wide_vec_t1",
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
    h = request["h"]
    hv = request["hv"]
    k = request["k"]
    v = request["v"]
    t = request["t"]
    batch = cute.sym_int64(symbol="B")
    pool = cute.sym_int64(symbol="P")
    flat_pool = cute.sym_int64(symbol="P_HV")
    per_token_pool_scatter = request.get("per_token_pool_scatter", False)
    per_token_pool_scatter_flat = request.get("per_token_pool_scatter_flat", False)

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
            assumed_align=32,
        )

    state = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (pool, hv, v, k),
        stride_order=(3, 2, 1, 0),
        assumed_align=32,
    )
    if per_token_pool_scatter_flat:
        intermediate = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (flat_pool, v, k),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
    else:
        intermediate = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, 1, 1, k),
            stride_order=(3, 2, 1, 0),
            assumed_align=32,
        )
    a_log = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (hv,), assumed_align=32
    )
    a = dynamic_strided("a", cutlass.BFloat16, (batch, t, hv))
    dt_bias = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (hv,), assumed_align=32
    )
    q = dynamic_strided("q", cutlass.BFloat16, (batch, t, h, k))
    key = dynamic_strided("k", cutlass.BFloat16, (batch, t, h, k))
    value = dynamic_strided("v", cutlass.BFloat16, (batch, t, hv, v))
    beta = dynamic_strided("b", cutlass.BFloat16, (batch, t, hv))
    output = dynamic_strided("o", cutlass.BFloat16, (batch, t, hv, v))
    state_indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch,), assumed_align=32
    )
    accepted_steps = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch,), assumed_align=32
    )
    ssm_state_indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (batch, t),
        stride_order=(1, 0),
        assumed_align=32,
    )
    stream = cutlass.runtime.make_fake_stream()
    compile_options = " ".join(
        [
            "--enable-tvm-ffi",
            f"--gpu-arch {request['gpu_arch']}",
            "--generate-line-info",
            "--opt-level 3",
        ]
    )

    if request["variant"] == "wide_vec_t1":
        compiled = cute.compile(
            module._run_wide_vec_t1,
            state,
            intermediate,
            a_log,
            a,
            dt_bias,
            q,
            key,
            value,
            beta,
            output,
            state_indices,
            state_indices,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=float(request["scale"]),
            HV=hv,
            T=t,
            H=h,
            K=k,
            V=v,
            tile_v=request["tile_v"],
            use_qk_l2norm=request["use_qk_l2norm"],
            disable_state_update=False,
            cache_intermediate_states=False,
            use_packed_fma=request["use_packed_fma"],
            same_pool=request["same_pool"],
            stream=stream,
            options=compile_options,
        )
    elif request["variant"] == "wide_vec":
        compiled = cute.compile(
            module._run_wide_vec,
            state,
            intermediate,
            a_log,
            a,
            dt_bias,
            q,
            key,
            value,
            beta,
            output,
            state_indices,
            state_indices,
            accepted_steps,
            ssm_state_indices,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=float(request["scale"]),
            HV=hv,
            T=t,
            H=h,
            K=k,
            V=v,
            tile_v=request["tile_v"],
            use_qk_l2norm=request["use_qk_l2norm"],
            disable_state_update=False,
            cache_intermediate_states=False,
            use_packed_fma=request["use_packed_fma"],
            same_pool=request["same_pool"],
            disable_output=False,
            recovery_steps=0,
            per_request_accepted_steps=False,
            per_token_pool_scatter=per_token_pool_scatter,
            per_token_pool_scatter_flat=per_token_pool_scatter_flat,
            stream=stream,
            options=compile_options,
        )
    else:
        compiled = cute.compile(
            module.run_gdn_decode_bf16state_mtp_ilp4,
            state,
            intermediate,
            a_log,
            a,
            dt_bias,
            q,
            key,
            value,
            beta,
            output,
            state_indices,
            state_indices,
            accepted_steps,
            ssm_state_indices,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=float(request["scale"]),
            HV=hv,
            T=t,
            H=h,
            K=k,
            V=v,
            tile_v_param=request["tile_v"],
            use_qk_l2norm=request["use_qk_l2norm"],
            disable_state_update=False,
            cache_intermediate_states=False,
            use_packed_fma=request["use_packed_fma"],
            same_pool=request["same_pool"],
            disable_output=False,
            per_request_accepted_steps=False,
            per_token_pool_scatter=per_token_pool_scatter,
            per_token_pool_scatter_flat=per_token_pool_scatter_flat,
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
            f"FlashInfer {FLASHINFER_VERSION} BF16-state source is missing: "
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
