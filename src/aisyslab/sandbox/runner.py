"""Run model-written Python in a separate process with limits.

What the process gets: a fresh temporary directory with the files it needs, an empty
environment (no API keys, no PATH), the standard library only (``-I -S``), limits on CPU time,
memory, file size and open files, a wall-clock timeout, and Python's network and subprocess calls
replaced by errors. What it is not: a security boundary against code written to break out. For
that, run it in a container or a microVM (E2B, Daytona, Firecracker); this runner is the local,
offline form of the same contract.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PRELUDE = """
import sys as _sys, os as _os
_sys.path.insert(0, _os.getcwd())  # -I leaves the script's directory off the path; the given files sit here
import builtins as _b, socket as _socket, subprocess as _subprocess
def _disabled(*_a, **_k):
    raise PermissionError("disabled in this sandbox")
_socket.socket = _disabled
_socket.create_connection = _disabled
_subprocess.Popen = _disabled
_subprocess.run = _disabled
_os.system = _disabled
_os.execv = _disabled
_os.execve = _disabled
_os.fork = _disabled
del _b, _socket, _subprocess, _os, _sys
"""


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "seconds": round(self.seconds, 2),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _limits(cpu_s: int, memory_mb: int) -> None:  # pragma: no cover - runs in the child
    import resource

    for limit, value in (
        (resource.RLIMIT_CPU, cpu_s),
        (resource.RLIMIT_FSIZE, 10 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
    ):
        # not every limit exists on every platform (macOS ignores RLIMIT_AS)
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(limit, (value, value))


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-limit // 2 :]


def run_python(
    code: str,
    *,
    files: Mapping[str, str] | None = None,
    timeout_s: float = 10.0,
    memory_mb: int = 512,
    max_output: int = 6000,
) -> SandboxResult:
    """Run ``code`` with ``files`` next to it; returns its bounded output and how it ended."""
    with tempfile.TemporaryDirectory(prefix="aisyslab-sandbox-") as tmp:
        for name, content in (files or {}).items():
            path = Path(tmp) / name
            if path.resolve().parent != Path(tmp).resolve():
                raise ValueError(f"Sandbox files must sit in the working directory: {name}")
            path.write_text(content, encoding="utf-8")
        (Path(tmp) / "main.py").write_text(PRELUDE + "\n" + code, encoding="utf-8")
        started = time.perf_counter()
        preexec = (lambda: _limits(int(timeout_s) + 1, memory_mb)) if os.name == "posix" else None
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-S", "main.py"],
                cwd=tmp,
                env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=preexec,  # limits are set in the child, before Python starts
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return SandboxResult(False, _bounded(out, max_output), "timed out", -9, True, time.perf_counter() - started)
        return SandboxResult(
            done.returncode == 0,
            _bounded(done.stdout, max_output),
            _bounded(done.stderr, max_output // 2),
            done.returncode,
            False,
            time.perf_counter() - started,
        )
