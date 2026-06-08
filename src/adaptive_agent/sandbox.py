from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool


class ExecutionSandbox:
    """Runs untrusted code in an isolated subprocess with output and time limits."""

    def __init__(
        self,
        workspace: Path | str,
        timeout_sec: float,
        max_output_bytes: int,
        network: str = "deny",
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self.max_output_bytes = max_output_bytes
        # Stored for future network-restriction enforcement; not enforced at MVP.
        self.network = network

    def _truncate(self, text: str) -> tuple[str, bool]:
        """Truncate text to max_output_bytes (UTF-8 bytes); return (text, was_truncated)."""
        encoded = text.encode("utf-8")
        if len(encoded) <= self.max_output_bytes:
            return text, False
        return encoded[: self.max_output_bytes].decode("utf-8", "ignore"), True

    def run_code(
        self,
        code: str,
        args: list[str] | None = None,
        stdin: str | None = None,
    ) -> SandboxResult:
        """Write code to a temp file in workspace then execute it."""
        script = self.workspace / "_tool_run.py"
        script.write_text(code, encoding="utf-8")
        return self.run_file(script.name, args=args, stdin=stdin)

    def run_file(
        self,
        rel_path: str,
        args: list[str] | None = None,
        stdin: str | None = None,
    ) -> SandboxResult:
        """Execute a file (relative to workspace) in an isolated subprocess."""
        # Minimal env to reduce side-channel leakage from the parent process.
        env = {"PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1"}
        # -I: isolated mode — ignores PYTHON* env vars, user site-packages, sys.path tweaks.
        cmd = [sys.executable, "-I", rel_path, *(args or [])]
        timed_out = False
        raw_out: str | bytes | None
        raw_err: str | bytes | None
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workspace,
                env=env,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            raw_out, raw_err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            raw_out, raw_err = e.stdout, e.stderr
            code, timed_out = -1, True

        def _to_str(v: str | bytes | None) -> str:
            if v is None:
                return ""
            return v if isinstance(v, str) else v.decode("utf-8", "ignore")

        out_s = _to_str(raw_out)
        err_s = _to_str(raw_err)
        if timed_out:
            err_s += "\n[timeout]"
        out, t1 = self._truncate(out_s)
        err, t2 = self._truncate(err_s)
        return SandboxResult(out, err, code, timed_out, t1 or t2)
