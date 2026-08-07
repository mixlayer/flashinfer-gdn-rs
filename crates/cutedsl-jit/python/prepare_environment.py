#!/usr/bin/env python3
"""Create or validate an immutable Python environment for a CuTeDSL worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import venv


ENVIRONMENT_SCHEMA_VERSION = 1
LOCKED_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Environment cache root (defaults to CUTEDSL_JIT_CACHE_DIR or XDG cache)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Validate and use a pre-provisioned interpreter without modifying it",
    )
    parser.add_argument(
        "--base-python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter used to create a managed venv",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable package-index access; normally paired with --wheelhouse",
    )
    parser.add_argument("--wheelhouse", type=Path)
    return parser.parse_args()


def _default_cache_root() -> Path:
    if value := os.environ.get("CUTEDSL_JIT_CACHE_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "cutedsl-jit"
    return Path.home() / ".cache" / "cutedsl-jit"


def _python_path(environment: Path) -> Path:
    return environment / "bin" / "python"


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _environment_digest(lock: Path, base_python: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema={ENVIRONMENT_SCHEMA_VERSION}\n".encode())
    digest.update(f"platform={sys.platform}\n".encode())
    digest.update(f"machine={platform.machine()}\n".encode())
    digest.update(f"python={sys.version_info.major}.{sys.version_info.minor}\n".encode())
    digest.update(f"base_python={base_python.resolve()}\n".encode())
    digest.update(lock.read_bytes())
    return digest.hexdigest()


def _locked_packages(lock: Path) -> dict[str, str]:
    packages = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if match := LOCKED_REQUIREMENT_RE.match(line):
            packages[match.group(1)] = match.group(2)
    if not packages:
        raise ValueError(f"requirements lock contains no exact package pins: {lock}")
    return packages


def _probe(interpreter: Path, package_names: list[str]) -> dict[str, object]:
    code = r"""
import importlib.metadata
import json
import sys
import cutlass.runtime

names = json.loads(sys.argv[1])
print(json.dumps({
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "packages": {name: importlib.metadata.version(name) for name in names},
    "runtime_libraries": cutlass.runtime.find_runtime_libraries(enable_tvm_ffi=True),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(interpreter), "-c", code, json.dumps(package_names)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            f"failed to probe compiler environment with {interpreter}: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _validate(
    interpreter: Path, expected_packages: dict[str, str]
) -> dict[str, object]:
    if not interpreter.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {interpreter}")
    probe = _probe(interpreter, sorted(expected_packages))
    actual = probe.get("packages")
    if actual != expected_packages:
        raise RuntimeError(
            "compiler environment package mismatch: "
            f"expected {expected_packages}, found {actual}"
        )
    libraries = probe.get("runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise RuntimeError("compiler environment did not report runtime libraries")
    for value in libraries:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"reported runtime library does not exist: {path}")
    return probe


def _install(
    staging: Path,
    lock: Path,
    expected_packages: dict[str, str],
    *,
    offline: bool,
    wheelhouse: Path | None,
) -> dict[str, object]:
    if staging.exists():
        shutil.rmtree(staging)
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(staging)
    interpreter = _python_path(staging)
    command = [
        str(interpreter),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--require-hashes",
        "--requirement",
        str(lock),
    ]
    if offline:
        command.append("--no-index")
    if wheelhouse is not None:
        command.extend(["--find-links", str(wheelhouse.resolve(strict=True))])

    log_path = staging / "install.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return _validate(interpreter, expected_packages)


def _publish_marker(
    environment: Path, digest: str, lock: Path, probe: dict[str, object]
) -> None:
    marker = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "environment_digest": digest,
        "lock": str(lock),
        **probe,
    }
    with (environment / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = _parse_args()
    lock = args.lock.resolve(strict=True)
    expected_packages = _locked_packages(lock)

    selected_python = args.python or (
        Path(value) if (value := os.environ.get("CUTEDSL_JIT_PYTHON")) else None
    )
    if selected_python is not None:
        interpreter = _absolute_without_resolving_symlinks(selected_python)
        probe = _validate(interpreter, expected_packages)
        print(
            json.dumps(
                {
                    "managed": False,
                    "python": str(interpreter),
                    **probe,
                },
                sort_keys=True,
            )
        )
        return

    if sys.version_info[:2] != (3, 12) or platform.machine() != "aarch64":
        raise RuntimeError(
            "this initial lock supports CPython 3.12 on aarch64 only; "
            "provide CUTEDSL_JIT_PYTHON or add a lock for this platform"
        )

    base_python = args.base_python.resolve(strict=True)
    cache_root = (args.cache_root or _default_cache_root()).expanduser().resolve()
    environments = cache_root / "envs"
    environments.mkdir(parents=True, exist_ok=True)
    digest = _environment_digest(lock, base_python)
    environment = environments / digest
    lock_path = environments / f"{digest}.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        interpreter = _python_path(environment)
        marker = environment / "environment.json"
        if marker.is_file():
            probe = _validate(interpreter, expected_packages)
        else:
            staging = environments / f"{digest}.tmp.{os.getpid()}"
            probe = _install(
                staging,
                lock,
                expected_packages,
                offline=args.offline,
                wheelhouse=args.wheelhouse,
            )
            _publish_marker(staging, digest, lock, probe)
            if environment.exists():
                shutil.rmtree(environment)
            os.replace(staging, environment)
            interpreter = _python_path(environment)
            probe = _validate(interpreter, expected_packages)
            _publish_marker(environment, digest, lock, probe)

    print(
        json.dumps(
            {
                "managed": True,
                "environment": str(environment),
                "environment_digest": digest,
                "python": str(interpreter),
                **probe,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
