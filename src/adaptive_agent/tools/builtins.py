from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Any, Callable

from ..python_repair import escape_newlines_in_string_literals
from ..sandbox import ExecutionSandbox
from .base import Tool, ToolResult


def _resolve(workspace: Path, rel: str) -> Path:
    target = (workspace / rel).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError("workspace 밖 경로 접근은 허용되지 않습니다")
    return target


def _blocked_python_imports(tree: ast.AST) -> list[str]:
    blocked: list[str] = []
    stdlib: set[str] = set(getattr(sys, "stdlib_module_names", set()))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".", 1)[0]]
        for name in names:
            if name == "__future__":
                continue
            if stdlib and name not in stdlib and name not in blocked:
                blocked.append(name)
    return blocked


def _direct_write_path(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        open_path = _open_write_path(node)
        if open_path is not None:
            return open_path
        pathlib_path = _pathlib_write_path(node)
        if pathlib_path is not None:
            return pathlib_path
    return None


def _open_write_path(node: ast.Call) -> str | None:
    if not isinstance(node.func, ast.Name) or node.func.id != "open":
        return None
    if not node.args:
        return None
    mode = "r"
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode_value = node.args[1].value
        if isinstance(mode_value, str):
            mode = mode_value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode_value = keyword.value.value
            if isinstance(mode_value, str):
                mode = mode_value
    if not any(flag in mode for flag in ("w", "a", "x", "+")):
        return None
    path_arg = node.args[0]
    if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
        return path_arg.value
    return "<dynamic path>"


def _pathlib_write_path(node: ast.Call) -> str | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in {"write_text", "write_bytes", "touch", "mkdir", "unlink", "rename"}:
        return None
    value = node.func.value
    if isinstance(value, ast.Call):
        return _path_constructor_arg(value)
    return None


def _path_constructor_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    if isinstance(node.func, ast.Name) and node.func.id == "Path":
        return _path_arg_label(node.args[0])
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "Path"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pathlib"
    ):
        return _path_arg_label(node.args[0])
    return None


def _path_arg_label(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "<dynamic path>"


def _guard_run_python_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    blocked_imports = _blocked_python_imports(tree)
    if blocked_imports:
        return (
            "runPython 코드는 Python 표준 라이브러리만 사용할 수 있습니다. "
            f"외부 모듈을 import하지 마세요: {', '.join(blocked_imports)}. "
            "별도 패키지 없이 문자열과 표준 라이브러리로 처리하세요."
        )
    return None


_RUN_PYTHON_PATH_GUARD = r"""
import builtins as __adaptive_agent_builtins
import os as __adaptive_agent_os
from pathlib import Path as __AdaptiveAgentPath

__adaptive_agent_workspace = __AdaptiveAgentPath.cwd().resolve()
__adaptive_agent_original_open = __adaptive_agent_builtins.open
__adaptive_agent_original_path_open = __AdaptiveAgentPath.open


def __adaptive_agent_check_path(path):
    if not isinstance(path, (str, bytes, __adaptive_agent_os.PathLike)):
        return path
    raw_path = __AdaptiveAgentPath(path)
    target = raw_path.resolve() if raw_path.is_absolute() else (__adaptive_agent_workspace / raw_path).resolve()
    if target != __adaptive_agent_workspace and __adaptive_agent_workspace not in target.parents:
        raise PermissionError("workspace 밖 경로 접근은 허용되지 않습니다")
    return path


def __adaptive_agent_guarded_open(file, *args, **kwargs):
    __adaptive_agent_check_path(file)
    return __adaptive_agent_original_open(file, *args, **kwargs)


def __adaptive_agent_guarded_path_open(self, *args, **kwargs):
    __adaptive_agent_check_path(self)
    return __adaptive_agent_original_path_open(self, *args, **kwargs)


__adaptive_agent_builtins.open = __adaptive_agent_guarded_open
__AdaptiveAgentPath.open = __adaptive_agent_guarded_path_open
"""


def build_file_tools(workspace: Path | str) -> list[Tool]:
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    def read_file(inp: dict[str, Any]) -> ToolResult:
        try:
            path = _resolve(ws, inp["path"])
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        data = path.read_bytes()
        max_bytes = int(inp.get("maxBytes", 1_048_576))
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", "ignore")
        return ToolResult(
            ok=True, output={"content": text, "bytes": len(data), "truncated": truncated}
        )

    def write_file(inp: dict[str, Any]) -> ToolResult:
        try:
            path = _resolve(ws, inp["path"])
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = inp.get("mode", "overwrite")
        content = inp["content"]
        with path.open("a" if mode == "append" else "w", encoding="utf-8") as fh:
            fh.write(content)
        return ToolResult(
            ok=True, output={"path": inp["path"], "bytesWritten": len(content.encode("utf-8"))}
        )

    def list_files(inp: dict[str, Any]) -> ToolResult:
        try:
            base = _resolve(ws, inp.get("path", "."))
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        ws_root = ws.resolve()
        glob = inp.get("glob")
        recursive = bool(inp.get("recursive", False))
        it = base.rglob(glob or "*") if recursive else base.glob(glob or "*")
        entries = [
            {
                "path": str(p.relative_to(ws_root)),
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else 0,
            }
            for p in it
        ]
        return ToolResult(ok=True, output={"entries": entries})

    return [
        Tool(
            "readFile",
            "허용된 작업 영역 파일을 읽는다",
            "builtin",
            {"type": "object", "required": ["path"]},
            read_file,
        ),
        Tool(
            "writeFile",
            "작업 영역에 파일을 쓴다",
            "builtin",
            {"type": "object", "required": ["path", "content"]},
            write_file,
        ),
        Tool(
            "listFiles", "작업 영역 파일 목록을 조회한다", "builtin", {"type": "object"}, list_files
        ),
    ]


def build_run_python(sandbox: ExecutionSandbox) -> Tool:
    def run_python(inp: dict[str, Any]) -> ToolResult:
        if "code" in inp:
            code = escape_newlines_in_string_literals(str(inp["code"]))
            guard_error = _guard_run_python_code(code)
            if guard_error is not None:
                return ToolResult(ok=False, error=guard_error)
            code = f"{_RUN_PYTHON_PATH_GUARD}\n{code}"
            if "def run(" in code and "__adaptive_agent_result" not in code:
                code += (
                    "\n\nif __name__ == '__main__':\n"
                    "    import json as __adaptive_agent_json\n"
                    "    __adaptive_agent_result = run({})\n"
                    "    if __adaptive_agent_result is not None:\n"
                    "        print(__adaptive_agent_json.dumps(__adaptive_agent_result, "
                    "ensure_ascii=False))\n"
                )
            res = sandbox.run_code(code, args=inp.get("args"), stdin=inp.get("stdin"))
        elif "file" in inp:
            res = sandbox.run_file(inp["file"], args=inp.get("args"), stdin=inp.get("stdin"))
        else:
            return ToolResult(ok=False, error="code 또는 file 중 하나가 필요합니다")
        ok = res.exit_code == 0 and not res.timed_out
        # On failure, surface stderr as the error so the model can read what broke
        # and self-correct on the next turn — mirrors GeneratedToolManager._invoke.
        error = None
        if not ok:
            if res.timed_out:
                error = "스크립트 실행이 시간을 초과했습니다"
            else:
                error = (
                    res.stderr.strip()
                    or f"스크립트가 비정상 종료했습니다 (종료 코드 {res.exit_code})"
                )
        return ToolResult(
            ok=ok,
            error=error,
            output={
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exitCode": res.exit_code,
                "timedOut": res.timed_out,
                "truncated": res.truncated,
            },
        )

    return Tool(
        "runPython",
        "제한된 Python 스크립트를 격리 실행한다. input.code에 최상위 스크립트를 넣는다 — "
        "함수 본문이 아니므로 return을 쓰지 말고 결과는 print로 출력한다. 스크립트는 "
        "workspace를 cwd로 실행하므로 data.json 같은 상대 경로를 직접 열 수 있다.",
        "builtin",
        {"type": "object"},
        run_python,
    )


def build_search_docs(docs_dir: Path | str) -> Tool:
    base = Path(docs_dir)

    def search_docs(inp: dict[str, Any]) -> ToolResult:
        query = inp["query"].lower()
        limit = int(inp.get("limit", 5))
        results: list[dict[str, Any]] = []
        if base.exists():
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                text = path.read_text("utf-8", "ignore")
                count = text.lower().count(query)
                if count:
                    idx = text.lower().find(query)
                    snippet = text[max(0, idx - 40) : idx + 80].replace("\n", " ")
                    results.append(
                        {
                            "docId": path.name,
                            "title": path.stem,
                            "snippet": snippet,
                            "score": float(count),
                        }
                    )
        results.sort(key=lambda r: r["score"], reverse=True)
        return ToolResult(ok=True, output={"results": results[:limit]})

    return Tool(
        "searchDocs",
        "로컬 문서에서 스키마·연산 근거를 조회한다",
        "builtin",
        {"type": "object", "required": ["query"]},
        search_docs,
    )


def build_ask_user(ask: Callable[[str, list[str] | None], str]) -> Tool:
    def ask_user(inp: dict[str, Any]) -> ToolResult:
        answer = ask(inp["question"], inp.get("choices"))
        return ToolResult(ok=True, output={"answer": answer})

    return Tool(
        "askUser",
        "모호성 해소를 위해 사용자에게 묻는다",
        "builtin",
        {"type": "object", "required": ["question"]},
        ask_user,
    )
