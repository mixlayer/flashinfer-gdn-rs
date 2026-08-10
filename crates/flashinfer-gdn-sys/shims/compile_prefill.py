#!/usr/bin/env python3
"""Compile one non-context-parallel FlashInfer GDN prefill specialization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import types
from typing import Any

from _artifact import finalize_artifact


SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 2
FLASHINFER_VERSION = "0.6.16.post2"
FLASHINFER_GIT_REV = "c498513a891d424e9ebb2518a1a3c53122dbf257"
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
GPU_ARCH_RE = re.compile(r"sm_[0-9]+[af]?\Z")

DELTA_RULE_FILES = (
    "alpha.py",
    "collective_inverse_hmma.py",
    "collective_store_tma.py",
    "helpers.py",
    "schedule.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def _architecture_number(gpu_arch: str) -> int:
    digits = "".join(
        character
        for character in gpu_arch.removeprefix("sm_")
        if character.isdigit()
    )
    return int(digits)


def _load_request(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        request = json.load(stream)
    if not isinstance(request, dict):
        raise ValueError("compiler request must be a JSON object")
    required = {
        "schema_version",
        "kernel",
        "symbol",
        "backend",
        "gpu_arch",
        "io_dtype",
        "state_dtype",
        "h",
        "hv",
        "k",
        "v",
        "scale",
        "num_sms",
        "use_state_indices",
        "checkpoint_every_n_tokens",
    }
    missing = required - request.keys()
    unknown = request.keys() - required
    if missing:
        raise ValueError(f"compiler request is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"compiler request has unknown fields: {sorted(unknown)}")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {request['schema_version']!r}; "
            f"expected {REQUEST_SCHEMA_VERSION}"
        )
    backend = request["backend"]
    identities = {
        "sm90": ("gdn_prefill_sm90", "flashinfer_gdn_prefill_sm90"),
        "sm100": ("gdn_prefill_sm100", "flashinfer_gdn_prefill_sm100"),
        "sm120": ("gdn_prefill_sm120", "flashinfer_gdn_prefill_sm120"),
    }
    if backend not in identities:
        raise ValueError(f"unsupported prefill backend {backend!r}")
    if (request["kernel"], request["symbol"]) != identities[backend]:
        raise ValueError(f"request identity does not match backend {backend}")
    if not isinstance(request["symbol"], str) or not SYMBOL_RE.fullmatch(
        request["symbol"]
    ):
        raise ValueError("symbol must be a C identifier")
    gpu_arch = request["gpu_arch"]
    if not isinstance(gpu_arch, str) or not GPU_ARCH_RE.fullmatch(gpu_arch):
        raise ValueError("gpu_arch must look like sm_90a or sm_121a")
    arch = _architecture_number(gpu_arch)
    if backend == "sm90" and arch != 90:
        raise ValueError("SM90 prefill must target sm_90a")
    if backend == "sm100" and arch not in (100, 103):
        raise ValueError("SM100 prefill must target sm_100a or sm_103a")
    if backend == "sm120" and arch not in (120, 121):
        raise ValueError("SM120 prefill must target sm_120a or sm_121a")
    if request["io_dtype"] != "bfloat16":
        raise ValueError("the initial prefill slice requires bfloat16 I/O")
    expected_state_dtype = "bfloat16" if backend == "sm100" else "float32"
    if request["state_dtype"] != expected_state_dtype:
        raise ValueError(
            f"{backend} prefill requires {expected_state_dtype} kernel state"
        )
    for name in ("h", "hv", "k", "v", "num_sms"):
        if not isinstance(request[name], int) or request[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if request["hv"] < request["h"] or request["hv"] % request["h"]:
        raise ValueError("hv must be a positive multiple of h")
    if request["k"] != 128 or request["v"] != 128:
        raise ValueError("GDN prefill requires k=v=128")
    if (
        not isinstance(request["scale"], (int, float))
        or not math.isfinite(request["scale"])
        or request["scale"] <= 0
    ):
        raise ValueError("scale must be a positive finite number")
    if not isinstance(request["use_state_indices"], bool):
        raise ValueError("use_state_indices must be a boolean")
    if request["use_state_indices"] != (backend == "sm100"):
        raise ValueError("only SM100 prefill supports state indices")
    checkpoint_every_n_tokens = request["checkpoint_every_n_tokens"]
    if (
        not isinstance(checkpoint_every_n_tokens, int)
        or checkpoint_every_n_tokens < 0
        or (
            checkpoint_every_n_tokens > 0
            and checkpoint_every_n_tokens % 64 != 0
        )
    ):
        raise ValueError(
            "checkpoint_every_n_tokens must be zero or a positive multiple of 64"
        )
    return request


def _package(name: str, directory: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(directory / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(directory)]
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Python module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_delta_rule_imports(root: Path) -> Path:
    flashinfer = root / "flashinfer"
    kernels = flashinfer / "gdn_kernels"
    delta = kernels / "delta_rule_dsl"
    _package("flashinfer", flashinfer)
    _package("flashinfer.gdn_kernels", kernels)
    _package("flashinfer.gdn_kernels.delta_rule_dsl", delta)

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = fake_torch

    utils = types.ModuleType("flashinfer.utils")
    utils.get_device_sm_count = lambda *_args, **_kwargs: 0
    utils._get_cache_buf = lambda *_args, **_kwargs: None
    sys.modules["flashinfer.utils"] = utils

    cache = types.ModuleType(
        "flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache"
    )

    class KeyedCompileMixin:
        def manual_cache_key(self, *_names: str) -> None:
            return None

    cache.KeyedCompileMixin = KeyedCompileMixin
    cache.cached_compile = lambda *_args, **_kwargs: None
    cache.sm12x_compile_options = lambda *_args, **_kwargs: ()
    sys.modules[cache.__name__] = cache

    for filename in DELTA_RULE_FILES:
        _load_module(
            f"flashinfer.gdn_kernels.delta_rule_dsl.{Path(filename).stem}",
            delta / filename,
        )
    return delta


def _load_delta_rule(root: Path, backend: str) -> tuple[Any, Path]:
    delta = _install_delta_rule_imports(root)
    filename = "delta_rule_sm90.py" if backend == "sm90" else "delta_rule_sm120.py"
    module = _load_module(
        f"flashinfer.gdn_kernels.delta_rule_dsl.{Path(filename).stem}",
        delta / filename,
    )
    class_name = (
        "_FullyFusedDeltaRuleSm90"
        if backend == "sm90"
        else "_FullyFusedDeltaRuleSm120"
    )
    return getattr(module, class_name), delta / filename


def _load_sm100(root: Path) -> tuple[Any, Path]:
    flashinfer = root / "flashinfer"
    kernels = flashinfer / "gdn_kernels"
    blackwell = kernels / "blackwell"
    _package("flashinfer", flashinfer)
    _package("flashinfer.gdn_kernels", kernels)
    _package("flashinfer.gdn_kernels.blackwell", blackwell)
    _load_module(
        "flashinfer.gdn_kernels.blackwell.gated_delta_net_tile_scheduler",
        blackwell / "gated_delta_net_tile_scheduler.py",
    )
    source = blackwell / "gated_delta_net_chunked.py"
    module = _load_module(
        "flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked", source
    )
    return module.GatedDeltaNetChunkedKernel, source


def _compact_tensor(cute: Any, dtype: Any, shape: tuple[Any, ...], align: int = 16):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def _compile_delta_rule(
    request: dict[str, Any], root: Path, object_path: Path
) -> tuple[list[str], Path]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.runtime
    import tvm_ffi  # noqa: F401

    # CuTeDSL resolves postponed annotations through the function's module
    # globals when it inspects the exported JIT callable.
    globals()["cute"] = cute
    globals()["cutlass"] = cutlass

    kernel_class, source_file = _load_delta_rule(root, request["backend"])
    h = request["h"]
    hv = request["hv"]
    d = request["k"]
    total = cute.sym_int64(symbol="N")
    batch = cute.sym_int64(symbol="B")
    checkpoint_count = cute.sym_int64(symbol="C")
    cu_count = cute.sym_int64(symbol="CU")
    checkpoint_interval = request["checkpoint_every_n_tokens"]
    checkpoints_enabled = checkpoint_interval > 0

    class ExportBase:
        def __init__(self):
            self.kernel = kernel_class(
                True, True, True, checkpoints_enabled, cutlass.BFloat16
            )

        def invoke(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            state_checkpoints,
            checkpoint_cu_starts,
            tensormaps: cute.Tensor,
            stream,
        ):
            q_tma = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], q.shape[1]),
                    stride=(q.stride[0], q.stride[2], q.stride[1]),
                ),
            )
            k_tma = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[2], k.shape[0], k.shape[1]),
                    stride=(k.stride[2], k.stride[0], k.stride[1]),
                ),
            )
            v_tma = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], v.shape[1]),
                    stride=(v.stride[2], v.stride[0], v.stride[1]),
                ),
            )
            o_tma = cute.make_tensor(
                output.iterator,
                cute.make_layout(
                    (output.shape[2], output.shape[0], output.shape[1]),
                    stride=(
                        output.stride[2],
                        output.stride[0],
                        output.stride[1],
                    ),
                ),
            )
            alpha_flat = cute.make_tensor(
                alpha.iterator, cute.make_layout((cute.size(alpha),), stride=(1,))
            )
            beta_flat = cute.make_tensor(
                beta.iterator, cute.make_layout((cute.size(beta),), stride=(1,))
            )
            initial_flat = cute.make_tensor(
                initial_state.iterator,
                cute.make_layout((cute.size(initial_state),), stride=(1,)),
            )
            output_state_flat = cute.make_tensor(
                output_state.iterator,
                cute.make_layout((cute.size(output_state),), stride=(1,)),
            )
            checkpoint_flat = (
                cute.make_tensor(
                    state_checkpoints.iterator,
                    cute.make_layout((cute.size(state_checkpoints),), stride=(1,)),
                )
                if checkpoints_enabled
                else None
            )
            num_seqs = cutlass.Int32(cu_seqlens.shape[0] - 1)
            total_checkpoints = (
                cutlass.Int32(state_checkpoints.shape[0])
                if checkpoints_enabled
                else cutlass.Int32(1)
            )
            self.kernel(
                q_tma,
                k_tma,
                v_tma,
                o_tma,
                alpha_flat,
                beta_flat,
                output_state_flat,
                initial_flat,
                checkpoint_flat,
                checkpoint_cu_starts,
                tensormaps,
                cu_seqlens,
                cutlass.Float32(request["scale"]),
                cutlass.Int32(h),
                cutlass.Int32(h),
                cutlass.Int32(hv),
                cutlass.Int32(hv),
                num_seqs,
                total_checkpoints,
                cutlass.Int32(checkpoint_interval),
                num_seqs * cutlass.Int32(hv),
                stream,
            )

    class Export(ExportBase):
        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            tensormaps: cute.Tensor,
            stream,
        ):
            self.invoke(
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                output,
                output_state,
                cu_seqlens,
                None,
                None,
                tensormaps,
                stream,
            )

    class CheckpointExport(ExportBase):
        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            state_checkpoints: cute.Tensor,
            checkpoint_cu_starts: cute.Tensor,
            tensormaps: cute.Tensor,
            stream,
        ):
            self.invoke(
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                output,
                output_state,
                cu_seqlens,
                state_checkpoints,
                checkpoint_cu_starts,
                tensormaps,
                stream,
            )

    q = _compact_tensor(cute, cutlass.BFloat16, (total, h, d))
    k = _compact_tensor(cute, cutlass.BFloat16, (total, h, d))
    v = _compact_tensor(cute, cutlass.BFloat16, (total, hv, d))
    alpha = _compact_tensor(cute, cutlass.Float32, (total, hv))
    beta = _compact_tensor(cute, cutlass.Float32, (total, hv))
    initial_state = _compact_tensor(cute, cutlass.Float32, (batch, hv, d, d))
    output = _compact_tensor(cute, cutlass.BFloat16, (total, hv, d))
    output_state = _compact_tensor(cute, cutlass.Float32, (batch, hv, d, d))
    cu_seqlens = _compact_tensor(cute, cutlass.Int64, (cu_count,), align=8)
    state_checkpoints = _compact_tensor(
        cute, cutlass.Float32, (checkpoint_count, hv, d, d)
    )
    checkpoint_cu_starts = _compact_tensor(
        cute, cutlass.Int64, (cu_count,), align=8
    )
    tensormaps = _compact_tensor(
        cute, cutlass.Uint8, (request["num_sms"] * 128,), align=128
    )
    stream = cutlass.runtime.make_fake_stream()
    option_parts = [
        "--enable-tvm-ffi",
        f"--gpu-arch {request['gpu_arch']}",
        "--generate-line-info",
        "--opt-level 3",
    ]
    if request["backend"] == "sm120":
        # Match the pinned source's required post-compile SM12x safety check.
        option_parts.append("--keep-ptx")
        option_parts.append(f"--dump-dir {shlex.quote(str(object_path.parent))}")
    options = " ".join(option_parts)
    compile_args = [
        CheckpointExport() if checkpoints_enabled else Export(),
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        output,
        output_state,
        cu_seqlens,
    ]
    if checkpoints_enabled:
        compile_args.extend([state_checkpoints, checkpoint_cu_starts])
    compile_args.extend([tensormaps, stream])
    compiled = cute.compile(*compile_args, options=options)
    if request["backend"] == "sm120":
        ptx = getattr(compiled, "__ptx__", "")
        ptx_path = None
        if isinstance(ptx, str) and "\n" not in ptx:
            candidate = Path(ptx)
            if not candidate.is_file():
                candidate = object_path.parent / candidate
            if candidate.is_file():
                ptx_path = candidate
                ptx = candidate.read_text(encoding="utf-8")
        forbidden = (
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes.L2::cache_hint"
        )
        if not ptx:
            raise RuntimeError("unable to inspect generated SM12x prefill PTX")
        if forbidden in ptx:
            raise RuntimeError(
                "SM12x prefill generated unsupported cluster-scoped TMA loads"
            )
        if ptx_path is not None:
            ptx_path.unlink()
        for generated_ptx in object_path.parent.glob("*.ptx"):
            generated_ptx.unlink()
    compiled.export_to_c(
        str(object_path),
        function_name=request["symbol"],
        export_only_tvm_ffi_symbols=True,
    )
    return cutlass.runtime.find_runtime_libraries(enable_tvm_ffi=True), source_file


def _compile_sm100(
    request: dict[str, Any], root: Path, object_path: Path
) -> tuple[list[str], Path]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.runtime

    globals()["cute"] = cute
    globals()["cutlass"] = cutlass

    kernel_class, source_file = _load_sm100(root)
    h = request["h"]
    hv = request["hv"]
    d = request["k"]
    total = cute.sym_int64(symbol="N")
    batch = cute.sym_int64(symbol="B")
    pool = cute.sym_int64(symbol="P")
    checkpoint_count = cute.sym_int64(symbol="C")
    cu_count = cute.sym_int64(symbol="CU")
    checkpoint_interval = request["checkpoint_every_n_tokens"]
    checkpoints_enabled = checkpoint_interval > 0

    class ExportBase:
        def __init__(self):
            self.kernel = kernel_class(
                io_dtype=cutlass.BFloat16,
                acc_dtype=cutlass.Float32,
                state_dtype=cutlass.BFloat16,
                mma_tiler_qk=(64, 64, 128),
                mma_tiler_qs=(128, 64, 128),
                mma_tiler_qkv=(128, 64, 64),
                mma_tiler_kv=(128, 128, 64),
                max_active_clusters=request["num_sms"],
                num_sm=request["num_sms"],
                is_GQA=False,
                use_initial_state=True,
                store_final_state=True,
                enable_checkpoints=checkpoints_enabled,
                is_persistent=True,
            )

        def invoke(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            state_indices: cute.Tensor,
            state_checkpoints,
            checkpoint_cu_starts,
            tensormaps: cute.Tensor,
            stream,
        ):
            self.kernel(
                q,
                k,
                v,
                alpha,
                beta,
                output,
                cu_seqlens,
                initial_state,
                output_state,
                state_indices,
                state_checkpoints,
                checkpoint_cu_starts,
                cutlass.Int32(checkpoint_interval),
                cutlass.Float32(request["scale"]),
                tensormaps,
                stream,
            )

    class Export(ExportBase):
        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            state_indices: cute.Tensor,
            tensormaps: cute.Tensor,
            stream,
        ):
            self.invoke(
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                output,
                output_state,
                cu_seqlens,
                state_indices,
                None,
                None,
                tensormaps,
                stream,
            )

    class CheckpointExport(ExportBase):
        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            alpha: cute.Tensor,
            beta: cute.Tensor,
            initial_state: cute.Tensor,
            output: cute.Tensor,
            output_state: cute.Tensor,
            cu_seqlens: cute.Tensor,
            state_indices: cute.Tensor,
            state_checkpoints: cute.Tensor,
            checkpoint_cu_starts: cute.Tensor,
            tensormaps: cute.Tensor,
            stream,
        ):
            self.invoke(
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                output,
                output_state,
                cu_seqlens,
                state_indices,
                state_checkpoints,
                checkpoint_cu_starts,
                tensormaps,
                stream,
            )

    q = _compact_tensor(cute, cutlass.BFloat16, (total, h, d))
    k = _compact_tensor(cute, cutlass.BFloat16, (total, h, d))
    v = _compact_tensor(cute, cutlass.BFloat16, (total, hv, d))
    alpha = _compact_tensor(cute, cutlass.Float32, (total, hv))
    beta = _compact_tensor(cute, cutlass.Float32, (total, hv))
    initial_state = _compact_tensor(cute, cutlass.BFloat16, (pool, hv, d, d))
    output = _compact_tensor(cute, cutlass.BFloat16, (total, hv, d))
    output_state = _compact_tensor(cute, cutlass.BFloat16, (pool, hv, d, d))
    cu_seqlens = _compact_tensor(cute, cutlass.Int32, (cu_count,), align=4)
    state_indices = _compact_tensor(cute, cutlass.Int32, (batch,), align=4)
    state_checkpoints = _compact_tensor(
        cute, cutlass.BFloat16, (checkpoint_count, hv, d, d)
    )
    checkpoint_cu_starts = _compact_tensor(
        cute, cutlass.Int32, (cu_count,), align=4
    )
    tensormaps = _compact_tensor(
        cute, cutlass.Uint8, (request["num_sms"] * 4 * 128,), align=16
    )
    stream = cutlass.runtime.make_fake_stream()
    options = " ".join(
        [
            "--enable-tvm-ffi",
            f"--gpu-arch {request['gpu_arch']}",
            "--generate-line-info",
            "--opt-level 2",
        ]
    )
    compile_args = [
        CheckpointExport() if checkpoints_enabled else Export(),
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        output,
        output_state,
        cu_seqlens,
        state_indices,
    ]
    if checkpoints_enabled:
        compile_args.extend([state_checkpoints, checkpoint_cu_starts])
    compile_args.extend([tensormaps, stream])
    compiled = cute.compile(*compile_args, options=options)
    compiled.export_to_c(
        str(object_path),
        function_name=request["symbol"],
        export_only_tvm_ffi_symbols=True,
    )
    return cutlass.runtime.find_runtime_libraries(enable_tvm_ffi=True), source_file


def main() -> None:
    args = _parse_args()
    request_path = args.request.resolve(strict=True)
    shim_path = Path(__file__).resolve(strict=True)
    request = _load_request(request_path)
    source_root = args.flashinfer_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    object_path = output_dir / "module.o"

    if request["backend"] == "sm100":
        runtime_libraries, source_file = _compile_sm100(
            request, source_root, object_path
        )
    else:
        runtime_libraries, source_file = _compile_delta_rule(
            request, source_root, object_path
        )
    if not source_file.is_file():
        raise FileNotFoundError(f"pinned prefill source is missing: {source_file}")
    (output_dir / "kernel_source.py").write_text(
        source_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    entry_file = source_file.relative_to(source_root)
    finalize_artifact(
        cc=args.cc,
        request=request,
        request_path=request_path,
        shim_path=shim_path,
        support_path=shim_path.with_name("_artifact.py"),
        source_file=source_file,
        entry_file=entry_file,
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
