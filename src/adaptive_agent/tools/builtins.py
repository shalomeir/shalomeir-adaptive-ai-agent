from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from ..sandbox import ExecutionSandbox
from .base import Tool, ToolResult


def _resolve(workspace: Path, rel: str) -> Path:
    target = (workspace / rel).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError("workspace 밖 경로 접근은 허용되지 않습니다")
    return target


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


def build_normalize_csv(workspace: Path | str) -> Tool:
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    def normalize_csv(inp: dict[str, Any]) -> ToolResult:
        try:
            src = _resolve(ws, inp["src"])
            dst = _resolve(ws, inp["dst"])
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        sort_by = str(inp.get("sortBy", "date"))
        try:
            with src.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
        except FileNotFoundError:
            return ToolResult(ok=False, error=f"입력 파일을 찾을 수 없습니다: {inp['src']}")
        if not rows:
            return ToolResult(ok=False, error="CSV 파일이 비어 있습니다")

        header, body = rows[0], rows[1:]
        if sort_by not in header:
            return ToolResult(ok=False, error=f"정렬 기준 컬럼이 없습니다: {sort_by}")

        seen: set[tuple[str, ...]] = set()
        unique_rows: list[list[str]] = []
        for row in body:
            key = tuple(row)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)

        sort_index = header.index(sort_by)
        unique_rows.sort(key=lambda row: row[sort_index])
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(unique_rows)

        return ToolResult(
            ok=True,
            output={
                "src": inp["src"],
                "dst": inp["dst"],
                "rows": len(unique_rows),
                "removedDuplicates": len(body) - len(unique_rows),
                "sortBy": sort_by,
            },
        )

    return Tool(
        "normalizeCsv",
        "CSV에서 완전히 중복된 행을 제거하고 지정 컬럼 기준 오름차순으로 정렬해 저장한다",
        "builtin",
        {"type": "object", "required": ["src", "dst"]},
        normalize_csv,
    )


def build_run_python(sandbox: ExecutionSandbox) -> Tool:
    def run_python(inp: dict[str, Any]) -> ToolResult:
        if "code" in inp:
            res = sandbox.run_code(inp["code"], args=inp.get("args"), stdin=inp.get("stdin"))
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
        "workspace를 cwd로 실행하므로 events.csv 같은 상대 경로를 직접 열 수 있다.",
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
