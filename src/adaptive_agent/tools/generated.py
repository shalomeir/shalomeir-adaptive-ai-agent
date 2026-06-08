from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..python_repair import escape_newlines_in_string_literals
from ..sandbox import ExecutionSandbox
from ..schemas import ToolSpec
from .base import Tool, ToolResult

# Template for the runner script that invokes the tool entrypoint.
# Literal braces are doubled so .format() passes them through unchanged;
# only {entrypoint} is substituted.
_RUNNER = """\
import json, sys, importlib.util
from pathlib import Path
tool_path = Path(__file__).with_name("tool.py")
spec = importlib.util.spec_from_file_location("tool", tool_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
payload = json.loads(sys.stdin.read() or "{{}}")
result = getattr(mod, "{entrypoint}")(payload)
print("__RESULT__" + json.dumps(result))
"""


class GeneratedToolManager:
    """Manages dynamically generated tools within a session directory.

    Each tool is written to its own subdirectory under session_dir and
    executed in an isolated subprocess via ExecutionSandbox.
    """

    def __init__(self, session_dir: Path | str, sandbox: ExecutionSandbox) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self._specs: dict[str, ToolSpec] = {}

    def _runner_rel_path(self, name: str) -> str:
        root = self.sandbox.workspace.resolve()
        runner = (self._dir(name) / "_runner.py").resolve()
        if runner != root and root not in runner.parents:
            raise ValueError("생성 도구 세션 디렉터리는 workspace 안에 있어야 합니다")
        return str(runner.relative_to(root))

    def _dir(self, name: str) -> Path:
        d = self.session_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write(self, spec: ToolSpec) -> None:
        """Write tool.py and _runner.py to the tool's directory."""
        d = self._dir(spec.name)
        (d / "tool.py").write_text(escape_newlines_in_string_literals(spec.code), encoding="utf-8")
        (d / "_runner.py").write_text(_RUNNER.format(entrypoint=spec.entrypoint), encoding="utf-8")

    def _invoke(self, spec: ToolSpec, payload: dict[str, Any]) -> ToolResult:
        """Run _runner.py with the workspace as cwd so relative file paths work."""
        try:
            runner_path = self._runner_rel_path(spec.name)
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        res = self.sandbox.run_file(runner_path, stdin=json.dumps(payload))
        if res.timed_out:
            return ToolResult(ok=False, error="실행 시간 초과")
        if res.exit_code != 0:
            return ToolResult(ok=False, error=res.stderr.strip() or "실행 실패")
        # Use the last __RESULT__ line: the runner emits exactly one as its final
        # output, so last-match prevents a tool from forging an earlier result line.
        result_line = None
        for line in res.stdout.splitlines():
            if line.startswith("__RESULT__"):
                result_line = line
        if result_line is not None:
            return ToolResult(ok=True, output=json.loads(result_line[len("__RESULT__") :]))
        return ToolResult(ok=False, error="도구가 결과를 반환하지 않았습니다")

    def smoke_test(self, spec: ToolSpec) -> ToolResult:
        """Write the spec and run it with an empty payload to verify syntax/runtime."""
        self._write(spec)
        return self._invoke(spec, {})

    def create(self, spec: ToolSpec) -> Tool:
        """Register and persist a new tool, then return a callable Tool instance."""
        self._specs[spec.name] = spec
        self._write(spec)
        return self._as_tool(spec)

    def update(self, name: str, code: str) -> Tool:
        """Replace the code of an existing tool and return the updated Tool instance."""
        spec = self._specs[name].model_copy(update={"code": code})
        self._specs[name] = spec
        self._write(spec)
        return self._as_tool(spec)

    def _as_tool(self, spec: ToolSpec) -> Tool:
        # Capture spec in a typed variable so mypy can infer the handler signature.
        captured: ToolSpec = spec

        def handler(payload: dict[str, Any]) -> ToolResult:
            return self._invoke(captured, payload)

        return Tool(
            spec.name,
            spec.description,
            "generated",
            spec.input_schema,
            handler,
            spec.output_schema,
        )

    def specs(self) -> dict[str, ToolSpec]:
        """Return a snapshot of all registered specs."""
        return dict(self._specs)
