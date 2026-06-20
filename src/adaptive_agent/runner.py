from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
import io
from pathlib import Path
import json
import queue
import re
import threading
from typing import Any, Callable, Literal

from .action_validation import TurnExecutionState
from .context import ContextManager, summarize_messages
from .conversation import ConversationStore
from .llm import LLMClient
from .monitoring import Exporter
from .observability import Tracer
from .parsing import parse_action_text
from .policy import PolicyManager
from .schemas import AskUser, CallTool, CreateTool, Finish, Respond, UpdateTool
from .skills import SkillStore
from .tools.base import ToolResult
from .tools.generated import GeneratedToolManager
from .tools.registry import ToolRegistry

NON_INTERACTIVE_ASK = "__adaptive_agent_non_interactive_ask_unavailable__"
LLM_RESPONSE_LOG_CHARS = 4000
TOOL_PAYLOAD_LOG_CHARS = 4000
REDACTED_LOG_VALUE = "[REDACTED]"
SENSITIVE_LOG_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
GENERATED_WRITE_PATH_FIELD_NAMES = frozenset(
    {
        "dest",
        "destination",
        "destinationfile",
        "destinationpath",
        "dst",
        "outfile",
        "outpath",
        "output",
        "outputfile",
        "outputpath",
        "savefile",
        "savepath",
        "target",
        "targetfile",
        "targetpath",
        "writefile",
        "writepath",
    }
)
AMBIGUOUS_WRITE_PATH_FIELD_NAMES = frozenset({"dest", "destination", "output", "target"})
UNKNOWN_DYNAMIC_WRITE_PATH = "<dynamic path>"
FILE_WRITE_INTENT_TERMS = (
    "as a file",
    "create",
    "export",
    "output file",
    "overwrite",
    "persist",
    "save",
    "store",
    "update",
    "write",
    "기록",
    "내보내",
    "만들",
    "생성",
    "수정",
    "작성",
    "저장",
    "출력 파일",
    "파일에",
    "파일로",
)
DATA_TOOL_TASK_TERMS = (
    "aggregate",
    "analysis",
    "analyze",
    "average",
    "clean",
    "convert",
    "dedupe",
    "duplicate",
    "extract",
    "filter",
    "group",
    "parse",
    "sort",
    "sum",
    "table",
    "transform",
    "평균",
    "합계",
    "집계",
    "분석",
    "필터",
    "정렬",
    "중복",
    "정리",
    "변환",
    "추출",
    "제거",
    "제외",
    "삭제",
    "수정",
    "업데이트",
    "표",
    "알려",
    "구해",
    "나열",
    "순으로",
    "공격력",
    "health",
    "hp",
    "atk",
    "amount",
    "type",
    "date",
    "mermaid",
)

_SYSTEM = (
    "You are a task-solving agent that creates and runs small Python tools. You are not a "
    "general chatbot, and you are not any vendor's assistant — if asked who you are, say you are "
    "this tool-building agent and do not claim to be a specific company's model.\n"
    "STRICT OUTPUT CONTRACT: On EVERY turn reply with EXACTLY ONE JSON object and nothing else — "
    "no prose, no markdown, no code fences. The JSON object MUST contain a string field named "
    '"action". Never return a bare result object such as {"path":"out.csv","rows":5}; wrap final '
    'data as {"action":"respond","text":"...","final":true} or '
    '{"action":"finish","summary":"..."}. If your previous response violated this contract, '
    "correct it in the next response instead of repeating it.\n"
    "Pick one action:\n"
    '- {"action":"respond","text":"...","final":true} — give an answer; final:true ends the task\n'
    '- {"action":"ask_user","question":"..."} — ask when the request is ambiguous\n'
    '- {"action":"call_tool","name":"<tool>","input":{...}} — run a built-in or created tool\n'
    '- {"action":"create_tool","spec":{"name":"kebab-name","description":"...",'
    '"code":"def run(input):\\n    return ...","inputSchema":{"type":"object"}}} — make a tool\n'
    '- {"action":"update_tool","name":"<tool>","code":"def run(input):\\n    return ..."} — fix a failed tool\n'
    '- {"action":"finish","summary":"..."} — stop when the task is done\n'
    'To use a tool you MUST use call_tool — never put a tool name in the "action" field. '
    "When the tool inventory lists input fields, call that tool with exactly those field names. "
    'Workspace file paths are RELATIVE (for example, "data.json" or "report.csv"); do not use '
    'absolute paths or "..". Use readFile only to inspect a small file before deciding what to do.\n'
    'There are TWO ways to run Python. Both are invoked with call_tool — "runPython" and any tool '
    'you create are tool NAMES, never values of the "action" field:\n'
    "(a) The built-in runPython tool. Invoke it as "
    '{"action":"call_tool","name":"runPython","input":{"code":"<script>"}}. The code is a TOP-LEVEL '
    "script: there is NO input variable, do NOT write return — PRINT the result, e.g. "
    "print(json.dumps(result)).\n"
    "(b) A tool you create_tool, then call_tool by its name. Its code defines run(input) where "
    '`input` is the call_tool "input" dict, and you RETURN the result.\n'
    "To process a workspace file, OPEN IT DIRECTLY inside your code by its relative name — your code "
    "runs with the workspace as the working directory. You do NOT need to readFile first or pass "
    "file contents through input for normal file processing.\n"
    "Default to tool management: for workspace file/data analysis, transformation, aggregation, or "
    "state changes, prefer create_tool/update_tool/call_tool and reuse existing generated tools when "
    "they exactly match. Use runPython only for tiny scalar calculations, quick diagnostics, schema "
    "inspection, or verification where creating a managed tool would add no reusable value.\n"
    "After a generated tool failure, use update_tool to fix that generated tool and call it again. "
    "After a runPython failure, retry with corrected runPython only when runPython was appropriate.\n"
    "When writing Python for JSON files, inspect whether the top-level value is a dict or list. If the "
    "top-level value is a dict with one relevant list field, operate on that list field. If the "
    "top-level dict contains a root node with children, treat it as an object tree and traverse "
    "children recursively instead of asking the user for internal field names.\n"
    "When a request asks for records matching a condition and an average for those records, calculate "
    "the average over the filtered records only, never over the full dataset. Do not invent numeric "
    "field values; read them from the source file.\n"
    "For object tree requests that ask to remove, exclude, delete, 제거, 제외, or 삭제 nodes, update the "
    "workspace file copy and then verify the saved state before answering.\n"
    "For follow-up requests that refer to previous results, use the conversation history. If the "
    "follow-up needs values not present in the summary, reopen the source file mentioned earlier "
    "and reconstruct the result set from the previous condition. For sorted tables, use actual values "
    "from the source file instead of estimating them from names or averages.\n"
    "When writing Python for CSV files, use csv.DictReader or csv.DictWriter and use column names from "
    "the header. For exact duplicate CSV rows, compare the complete row values; never deduplicate by "
    "date when date is only the sort key. For grouped totals, group by the requested column only.\n"
    "When code is included in JSON, encode it as one valid JSON string with escaped newlines "
    "(\\n); never place raw multiline code outside the string value.\n"
    "For file output, prefer returning the computed text/data from runPython or a created tool, then "
    "call writeFile with the final relative path and content. This lets the runtime apply its file "
    "write policy. Do not write files directly inside runPython.\n"
    "If the user explicitly asks you to make, build, create, or generate a tool, you MUST use "
    "create_tool first, then call_tool on that created tool. Do not answer via runPython, writeFile, "
    "or an existing tool before creating and executing the requested generated tool.\n"
    "Prefer listed built-in tools when their descriptions match the task instead of creating ad hoc "
    "code. Read-only analysis must not call writeFile.\n"
    "Use ONLY the Python standard library (json, csv, re, math, etc.) — pandas/numpy are NOT "
    "installed and will fail to import. NEVER ask the user to install packages; for CSV work use "
    "the built-in csv module. Reuse an existing generated tool only when its description exactly "
    "matches the current task. When a tool fails, "
    "READ the error message carefully and fix it: use update_tool for a created tool, or call the "
    "tool again with corrected input. Do not repeat the same question; if you already have what you "
    "need, act. Only call writeFile when the user explicitly asks to save, write, create, or update "
    "a file. For read-only questions, answer with respond(final:true) or finish. If the user asks "
    "for specific fields or aggregates, answer only those requested values instead of dumping full "
    "records. Keep tool names in "
    "kebab-case. When you have the answer, reply with respond (final:true) or finish — do not keep "
    "calling tools."
)


@dataclass
class RunnerDeps:
    llm: LLMClient
    registry: ToolRegistry
    ask: Callable[..., str]
    log_dir: Path
    max_iterations: int = 20
    max_fix_retries: int = 3
    compaction_token_threshold: int = 12000
    exporter: Exporter | None = None
    non_interactive: bool = False


@dataclass
class TurnResult:
    summary: str = ""
    observations: list[str] = field(default_factory=list)
    stopped_reason: str = "finish"


@dataclass
class _TurnLoopState:
    result: TurnResult
    turn_state: TurnExecutionState
    effective_request: str
    fix_failures: int = 0
    parse_failures: int = 0
    last_action_sig: str | None = None
    action_repeats: int = 0
    completed_output_paths: set[str] = field(default_factory=set)
    used_search_docs: bool = False
    called_changed_tools: set[str] = field(default_factory=set)
    called_generated_tools: set[str] = field(default_factory=set)
    validation_failed_tool_indices: dict[str, int] = field(default_factory=dict)
    last_tool_call_index: int | None = None


_StepAction = Literal["dispatch", "continue", "break"]
_DATA_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./~-])(?:~/?|/|\.\.?/)?"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:csv|json|md|txt)"
    r"(?![A-Za-z0-9_.-])",
    flags=re.I,
)


def _preview_text(text: str, limit: int = LLM_RESPONSE_LOG_CHARS) -> tuple[str, bool]:
    """Return a bounded preview for local diagnostics logs."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _redact_for_log(value: Any) -> Any:
    """Mask common secret-shaped fields before writing diagnostic payloads."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key).replace("-", "_").lower()
            if any(part in key_text for part in SENSITIVE_LOG_KEY_PARTS):
                redacted[key] = REDACTED_LOG_VALUE
            else:
                redacted[key] = _redact_for_log(item)
        return redacted
    if isinstance(value, list):
        return [_redact_for_log(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_for_log(item) for item in value)
    return value


def _preview_value(value: Any, limit: int = TOOL_PAYLOAD_LOG_CHARS) -> tuple[str, int, bool]:
    """Serialize and bound structured tool payloads for JSONL diagnostics."""
    value = _redact_for_log(value)
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    preview, truncated = _preview_text(text, limit)
    return preview, len(text), truncated


class AgentRunner:
    def __init__(
        self,
        deps: RunnerDeps,
        generated: GeneratedToolManager | None = None,
        skills: SkillStore | None = None,
        policy: PolicyManager | None = None,
    ) -> None:
        self.deps = deps
        self.tracer = Tracer(deps.log_dir, exporter=deps.exporter)
        self.conv = ConversationStore(system=_SYSTEM)
        self.ctx = ContextManager(
            token_threshold=deps.compaction_token_threshold,
            summarize=summarize_messages,
        )
        self.generated = generated
        self.skills = skills
        self.policy = policy
        self._session_tools: list[str] = []
        self._tool_result_cache: dict[str, ToolResult] = {}
        self._confirmed_write_paths: set[str] = set()
        if self.skills is not None and self.generated is not None:
            for digest in self.skills.load_digests():
                spec = self.skills.load_spec(digest.name)
                self.deps.registry.register(self.generated.create(spec))

    def _action_signature(self, action: Any) -> str | None:
        """Stable key for detecting a model that repeats the same step forever.

        Terminal actions (finish, final respond) return None — they end the loop,
        so they can never spin. Everything else returns a content-sensitive key so
        an identical ask_user, tool call, or non-final respond is recognized as a
        repeat. These all parse fine and never trip the failure counter, so without
        this guard a weak model loops to max_iterations while re-prompting the user.
        """
        if isinstance(action, AskUser):
            return f"ask_user:{action.question}"
        if isinstance(action, CallTool):
            payload = json.dumps(action.input, sort_keys=True, ensure_ascii=False)
            return f"call_tool:{action.name}:{payload}"
        if isinstance(action, CreateTool):
            return f"create_tool:{action.spec.name}"
        if isinstance(action, UpdateTool):
            return f"update_tool:{action.name}"
        if isinstance(action, Respond) and action.final is False:
            return f"respond:{action.text}"
        return None

    def _blocked_ask_user_observation(self, question: str, request: str = "") -> str | None:
        """Convert invalid ask_user package prompts into a runtime observation."""
        repeated_request = self._repeated_actionable_request_observation(question, request)
        if repeated_request is not None:
            return repeated_request

        lowered = question.lower()
        mentions_package = any(
            term in lowered
            for term in (
                "pandas",
                "numpy",
                "pip install",
                "package",
                "module",
                "모듈",
                "패키지",
                "설치",
            )
        )
        asks_to_proceed_with_stdlib = "표준 라이브러리" in question or "standard library" in lowered
        if mentions_package or asks_to_proceed_with_stdlib:
            return (
                "패키지 설치 질문은 사용자에게 묻지 않습니다. pandas/numpy 같은 외부 패키지는 "
                "설치하거나 사용하지 말고, Python 표준 라이브러리(csv 등)만 사용해 현재 작업을 "
                "계속 진행하세요."
            )
        question_paths = self._mentioned_workspace_paths(question)
        if len(question_paths) == 1:
            structure_hint = self._workspace_structure_hint(question)
            if structure_hint:
                return (
                    "질문에 나온 작업 영역 파일은 사용자에게 내용을 확인해 달라고 묻지 않습니다. "
                    "readFile 또는 runPython으로 직접 열어 필요한 값을 확인하고 계속 진행하세요."
                    f"\n{structure_hint}"
                )
        known_paths = self._mentioned_workspace_paths(request)
        asks_for_known_paths = known_paths and any(
            term in lowered
            for term in (
                "input_file",
                "output_file",
                "input file",
                "output file",
                "file path",
                "경로",
                "파일명",
                "파일 이름",
            )
        )
        if asks_for_known_paths:
            return (
                "입출력 파일 경로는 사용자에게 다시 묻지 않습니다. 요청에 나온 상대 경로를 "
                f"그대로 사용하세요: {', '.join(known_paths)}."
            )
        asks_about_file_structure = any(
            term in lowered
            for term in (
                "structure of the data",
                "data structure",
                "file structure",
                "json structure",
                "csv structure",
                "파일 구조",
                "데이터 구조",
                "데이터의 구조",
                "json 구조",
                "csv 구조",
                "몇 개의 필드",
                "몇개의 필드",
                "필드",
                "field",
                "문자열",
                "자료형",
                "데이터 타입",
                "data type",
                "string",
                "numeric",
                "number",
                "형식",
                "표현",
                "확인해 주세요",
                "확인해 주시면",
                "알려주시면",
                "구조를 확인",
                "구조",
                "파일 내부",
                "리스트인 경우",
                "구성되어",
                "confirm",
            )
        )
        if asks_about_file_structure:
            structure_target = f"{request}\n{question}"
            structure_hint = self._workspace_structure_hint(structure_target)
            if not structure_hint and not self._mentioned_workspace_paths(structure_target):
                return None
            return (
                "작업 영역 파일의 구조는 사용자에게 묻지 않습니다. readFile 또는 runPython으로 "
                "파일을 직접 열어 최상위 타입과 필요한 필드를 확인한 뒤, 수정한 코드로 계속 "
                "진행하세요." + (f"\n{structure_hint}" if structure_hint else "")
            )
        return None

    def _repeated_actionable_request_observation(self, answer: str, request: str) -> str | None:
        """Reject model replies that merely echo an actionable user request."""
        if not self._mentioned_workspace_paths(request):
            return None
        request_compact = self._semantic_compact(request)
        answer_compact = self._semantic_compact(answer)
        if len(answer_compact) < 20:
            return None
        if answer_compact in request_compact or request_compact in answer_compact:
            paths = self._mentioned_workspace_paths(request)
            path_hint = (
                f" 요청에 나온 파일 경로는 그대로 사용하세요: {', '.join(paths)}." if paths else ""
            )
            structure_hint = self._workspace_structure_hint(request)
            return (
                "사용자 요청을 최종 답변이나 질문으로 되풀이하지 마세요. 필요한 파일은 "
                "readFile 또는 runPython으로 직접 열고, 계산/변환을 수행한 뒤 실제 결과로 "
                "respond(final:true) 또는 finish를 반환하세요."
                + path_hint
                + (f"\n{structure_hint}" if structure_hint else "")
            )
        return None

    def _semantic_compact(self, text: str) -> str:
        return re.sub(r"\W+", "", text.lower())

    def _hitl_required_summary(self, question: str) -> str:
        return f"HITL 처리가 필요합니다: {question}"

    def _mentioned_workspace_paths(self, text: str) -> list[str]:
        """Extract likely workspace-relative data paths from a user/model message."""
        candidates = re.findall(r"(?:[\w.-]+/)*[\w.-]+\.(?:json|csv|md|txt)", text, flags=re.I)
        paths: list[str] = []
        for candidate in candidates:
            path = candidate.strip("`'\".,:;()[]{}")
            if path.startswith("workspace/"):
                path = path.removeprefix("workspace/")
            if path and ".." not in Path(path).parts and path not in paths:
                paths.append(path)
        return paths

    def _conversation_context_text(self) -> str:
        return "\n".join(message.content for message in self.conv.body())

    def _workspace_structure_hint(self, text: str) -> str | None:
        """Read small file previews so the model can self-correct schema assumptions."""
        hints: list[str] = []
        for path in self._mentioned_workspace_paths(text)[:3]:
            res = self.deps.registry.call("readFile", {"path": path, "maxBytes": 4096})
            if not res.ok or not isinstance(res.output, dict):
                continue
            content = str(res.output.get("content", ""))
            hint = self._describe_file_content(path, content)
            if hint:
                hints.append(hint)
        return "\n".join(hints) if hints else None

    def _describe_file_content(self, path: str, content: str) -> str | None:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                return f"{path}: JSON 미리보기를 파싱하지 못했습니다."
            if isinstance(value, dict):
                keys = list(value.keys())
                list_fields = {
                    key: len(item) for key, item in value.items() if isinstance(item, list)
                }
                object_fields = {
                    key: list(item.keys())[:8]
                    for key, item in value.items()
                    if isinstance(item, dict)
                }
                tree_hint = ""
                root = value.get("root")
                if isinstance(root, dict) and isinstance(root.get("children"), list):
                    tree_hint = (
                        " root는 object tree node처럼 보입니다. root에서 시작해 children을 "
                        "재귀 순회하고, node props 안의 값을 확인하세요."
                    )
                return (
                    f"{path}: 최상위 타입은 dict, keys={keys}, "
                    f"listFields={list_fields}, objectFields={object_fields}."
                    f" scalarPaths={self._json_scalar_paths(value)}."
                    f"{tree_hint}"
                )
            if isinstance(value, list):
                sample_type = type(value[0]).__name__ if value else "empty"
                return (
                    f"{path}: 최상위 타입은 list, length={len(value)}, sampleType={sample_type}, "
                    f"scalarPaths={self._json_scalar_paths(value)}."
                )
            return f"{path}: 최상위 타입은 {type(value).__name__}."
        if suffix == ".csv":
            rows = list(csv.reader(io.StringIO(content)))
            if not rows:
                return f"{path}: CSV가 비어 있습니다."
            return f"{path}: CSV header={rows[0]}, previewRows={max(0, len(rows) - 1)}."
        if content:
            first_line = content.splitlines()[0] if content.splitlines() else ""
            return f"{path}: 텍스트 파일 미리보기 첫 줄={first_line!r}."
        return None

    def _read_workspace_text(self, path: str, max_bytes: int = 1_048_576) -> str | None:
        result = self.deps.registry.call("readFile", {"path": path, "maxBytes": max_bytes})
        if not result.ok or not isinstance(result.output, dict):
            return None
        return str(result.output.get("content", ""))

    def _request_context(self, request: str) -> str:
        return f"{self._conversation_context_text()}\n{request}"

    def _likely_output_path(self, request: str, suffix: str) -> str | None:
        paths = [path for path in self._mentioned_workspace_paths(request) if path.endswith(suffix)]
        if len(paths) >= 2:
            return paths[-1]
        if len(paths) == 1 and self._request_mentions_file_write(request):
            return paths[0]
        return None

    def _likely_source_path(
        self, request: str, suffix: str, output_path: str | None = None
    ) -> str | None:
        paths = [path for path in self._mentioned_workspace_paths(request) if path.endswith(suffix)]
        for path in paths:
            if path != output_path:
                return path
        return paths[0] if paths else None

    def _csv_rows(self, content: str) -> list[list[str]] | None:
        rows = list(csv.reader(io.StringIO(content)))
        return rows if rows else None

    def _csv_dedup_sort_validation(self, request: str) -> tuple[bool, str] | None:
        lowered = request.lower()
        compact = lowered.replace(" ", "")
        if not (
            (("duplicate" in lowered or "중복" in lowered) and "date" in lowered)
            and ("sort" in lowered or "정렬" in lowered or "오름차순" in lowered)
        ):
            return None
        output_path = self._likely_output_path(request, ".csv")
        source_path = self._likely_source_path(request, ".csv", output_path)
        if output_path is None or source_path is None:
            return None

        source_text = self._read_workspace_text(source_path)
        output_text = self._read_workspace_text(output_path)
        if source_text is None:
            return None
        if output_text is None:
            return (
                False,
                f"{output_path} 검증 실패: 파일이 아직 없거나 읽을 수 없습니다. "
                "runPython 또는 생성 도구로 실제 파일을 만든 뒤 다시 확인하세요.",
            )
        source_rows = self._csv_rows(source_text)
        output_rows = self._csv_rows(output_text)
        if not source_rows or not output_rows:
            return (
                False,
                f"{output_path} 검증 실패: CSV 내용이 비어 있습니다. 원본 header와 행을 유지해 다시 저장하세요.",
            )
        header, body = source_rows[0], source_rows[1:]
        if "date" not in header:
            return None
        date_index = header.index("date")
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for row in body:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        expected = [header, *sorted(unique, key=lambda row: row[date_index])]
        if output_rows != expected:
            expected_ids = [row[0] for row in expected[1:] if row]
            return (
                False,
                f"{output_path} 검증 실패: 완전히 중복된 전체 행만 제거하고 date 오름차순으로 "
                "안정 정렬해야 합니다. date는 정렬 키일 뿐 dedupe 키가 아닙니다. "
                "seen_dates 또는 row['date'] 기준 중복 제거는 금지입니다. "
                "반드시 seen_rows=set(); key=tuple(row.values()) 또는 tuple(row.items())로 "
                f"전체 행을 비교하세요. 같은 date의 원본 상대 순서를 유지하세요. expectedIds={expected_ids}.",
            )
        if "pandas" in compact or "numpy" in compact:
            return False, "외부 패키지 대신 Python 표준 라이브러리 csv로 다시 처리하세요."
        return True, f"{output_path} 저장 검증 완료: 고유 {len(expected) - 1}행, date 오름차순."

    def _entity_health_threshold(self, request: str) -> int | None:
        match = re.search(
            r"health[^0-9-]*(?:<|미만|below|under|less than)[^0-9-]*(\d+)", request, re.I
        )
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)[^0-9]*(?:미만|below|under)", request, re.I)
        return int(match.group(1)) if match else None

    def _collect_json_nodes(self, value: Any) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []

        def walk(item: Any) -> None:
            if isinstance(item, dict):
                nodes.append(item)
                for child in item.values():
                    if isinstance(child, dict):
                        walk(child)
                    elif isinstance(child, list):
                        for entry in child:
                            walk(entry)
            elif isinstance(item, list):
                for entry in item:
                    walk(entry)

        walk(value)
        return nodes

    def _object_tree_mutation_validation(self, request: str) -> tuple[bool, str] | None:
        if not self._request_needs_object_tree_mutation(request):
            return None
        threshold = self._entity_health_threshold(request)
        source_path = self._likely_source_path(request, ".json")
        if threshold is None or source_path is None:
            return None
        content = self._read_workspace_text(source_path)
        if content is None:
            return None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "root" not in data:
            return (
                False,
                f"{source_path} 검증 실패: top-level root wrapper가 사라졌습니다. "
                "data['root']만 저장하지 말고 전체 data 객체를 유지해 저장하세요.",
            )
        entities = [
            node
            for node in self._collect_json_nodes(data)
            if node.get("type") == "Entity" and isinstance(node.get("props"), dict)
        ]
        if not entities:
            return (
                False,
                f"{source_path} 검증 실패: 남은 Entity가 0개입니다. Container/Scene node는 보존하고 "
                "health 조건에 맞는 Entity만 제거하세요.",
            )
        low = [node.get("id") for node in entities if node["props"].get("health", 0) < threshold]
        if low:
            return (
                False,
                f"{source_path} 검증 실패: health가 {threshold} 미만인 Entity가 아직 남아 있습니다: {low}. "
                "평균만 계산하지 말고 파일을 실제로 다시 저장한 뒤, 저장된 파일을 재읽어서 평균을 계산하세요.",
            )
        healths = [node["props"].get("health", 0) for node in entities]
        avg = round(sum(healths) / len(healths), 2) if healths else 0
        return (
            True,
            f"{source_path} 저장 검증 완료: 남은 Entity {len(entities)}개, 평균 health {avg:g}.",
        )

    def _mutation_restore_snapshots(self, request: str, action: CallTool) -> dict[str, str]:
        if action.name != "runPython" or not self._request_needs_object_tree_mutation(request):
            return {}
        snapshots: dict[str, str] = {}
        for path in self._mentioned_workspace_paths(request):
            if not path.endswith(".json"):
                continue
            content = self._read_workspace_text(path)
            if content is not None:
                snapshots[path] = content
        return snapshots

    def _restore_workspace_files(self, snapshots: dict[str, str]) -> None:
        for path, content in snapshots.items():
            result = self.deps.registry.call("writeFile", {"path": path, "content": content})
            self._log_tool_call("writeFile", {"path": path, "content": content}, result)

    def _try_repair_completed_work(
        self, request: str, snapshots: dict[str, str]
    ) -> tuple[bool, str] | None:
        if not self._request_needs_object_tree_mutation(request):
            return None
        threshold = self._entity_health_threshold(request)
        source_path = self._likely_source_path(request, ".json")
        if threshold is None or source_path is None or source_path not in snapshots:
            return None
        try:
            data = json.loads(snapshots[source_path])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "root" not in data:
            return None

        def prune(node: dict[str, Any]) -> bool:
            if node.get("type") == "Entity":
                props = node.get("props")
                if isinstance(props, dict) and props.get("health", 0) < threshold:
                    return False
            children = node.get("children")
            if isinstance(children, list):
                kept = []
                for child in children:
                    if isinstance(child, dict) and prune(child):
                        kept.append(child)
                node["children"] = kept
            return True

        root = data.get("root")
        if not isinstance(root, dict):
            return None
        prune(root)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        result = self.deps.registry.call("writeFile", {"path": source_path, "content": content})
        self._log_tool_call("writeFile", {"path": source_path, "content": content}, result)
        if not result.ok:
            return None
        validation = self._object_tree_mutation_validation(request)
        if validation is None or not validation[0]:
            return None
        return (
            True,
            f"{validation[1]} 검증 실패한 도구 결과는 원본 snapshot에서 재구성해 복구했습니다.",
        )

    def _request_needs_object_tree_mutation(self, request: str) -> bool:
        lowered = request.lower()
        return (
            "entity" in lowered
            and "health" in lowered
            and any(
                term in lowered
                for term in ("remove", "exclude", "delete", "제거", "제외", "제외하", "삭제")
            )
        )

    def _run_python_missing_required_mutation_observation(
        self, request: str, action: CallTool
    ) -> str | None:
        if action.name != "runPython" or not self._request_needs_object_tree_mutation(request):
            return None
        code = str(action.input.get("code", ""))
        if self._code_writes_workspace_file(code):
            return None
        source_path = self._likely_source_path(request, ".json") or "the JSON file"
        threshold = self._entity_health_threshold(request)
        threshold_text = str(threshold) if threshold is not None else "요청 기준"
        return (
            f"이 요청은 읽기 전용 평균 계산이 아니라 {source_path} 상태 변경 작업입니다. "
            f"health가 {threshold_text} 미만인 Entity node를 children에서 재귀적으로 제거하고 "
            f"json.dump(..., open('{source_path}', 'w'))로 같은 파일에 저장한 뒤, 저장된 파일을 "
            "다시 읽어서 남은 Entity 평균을 계산하세요."
        )

    def _previous_json_filter_table_validation(
        self, request: str, path: str, content: str
    ) -> tuple[bool, str] | None:
        lowered = request.lower()
        if not (
            path.endswith(".md")
            and (
                "방금" in request
                or "previous" in lowered
                or "filtered" in lowered
                or "필터" in request
            )
            and ("table" in lowered or "표" in request)
            and ("내림차순" in request or "desc" in lowered)
        ):
            return None
        context = self._request_context(request)
        source_path = self._likely_source_path(context, ".json")
        if source_path is None:
            return None
        field_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:내림차순|desc)", context)
        if not field_match:
            return None
        field = field_match.group(1)
        threshold_match = re.search(
            rf"{re.escape(field)}[^0-9]*(?:>=\s*(\d+)|(\d+)\s*(?:이상|or more|and above))",
            context,
            re.I,
        )
        if not threshold_match:
            return None
        threshold = int(threshold_match.group(1) or threshold_match.group(2))
        source_text = self._read_workspace_text(source_path)
        if source_text is None:
            return None
        try:
            data = json.loads(source_text)
        except json.JSONDecodeError:
            return None

        def has_selected_numeric_field(item: dict[str, Any]) -> bool:
            value = item.get(field)
            return isinstance(value, int | float) and value >= threshold

        records = [
            node for node in self._collect_json_nodes(data) if has_selected_numeric_field(node)
        ]
        if not records and isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    records.extend(
                        item
                        for item in value
                        if isinstance(item, dict) and has_selected_numeric_field(item)
                    )
        records = sorted(records, key=lambda item: item[field], reverse=True)
        names = [str(item.get("name") or item.get("id") or "") for item in records]
        values = [str(item[field]) for item in records]
        positions = [content.find(name) for name in names]
        if any(pos < 0 for pos in positions) or positions != sorted(positions):
            return (
                False,
                f"{path} 검증 실패: 이전 필터 결과를 {field} 내림차순으로 다시 구성해야 합니다. "
                f"expectedOrder={names}. {source_path}를 다시 읽고 실제 값을 사용하세요.",
            )
        for name, value in zip(names, values, strict=True):
            row_pattern = re.compile(
                rf"\|\s*{re.escape(name)}\s*\|\s*{re.escape(value)}\s*\|", re.I
            )
            if not row_pattern.search(content):
                return (
                    False,
                    f"{path} 검증 실패: {name}의 실제 {field} 값은 {value}입니다. "
                    f"{source_path}를 다시 읽고 추정값이 아닌 실제 값으로 표를 작성하세요.",
                )
        return True, f"{path} 저장 검증 완료: {field} 내림차순 표."

    def _validate_completed_work(
        self, request: str, action: CallTool | None = None, result: ToolResult | None = None
    ) -> tuple[bool, str] | None:
        if action is not None and action.name == "writeFile":
            path = str(action.input.get("path", ""))
            content = str(action.input.get("content", ""))
            table_validation = self._previous_json_filter_table_validation(request, path, content)
            if table_validation is not None:
                return table_validation

        for validator in (self._csv_dedup_sort_validation, self._object_tree_mutation_validation):
            validation = validator(request)
            if validation is not None:
                return validation
        return None

    def _tool_call_next_step_hint(self, action: CreateTool, request: str) -> str:
        tool = self.deps.registry.get(action.spec.name)
        if tool is None:
            return ""
        properties = tool.input_schema.get("properties")
        property_names = list(properties) if isinstance(properties, dict) else []
        paths = self._mentioned_workspace_paths(request)
        if len(paths) >= 2 and property_names:
            input_payload: dict[str, str] = {}
            source = paths[0]
            output = paths[-1]
            for name in property_names:
                normalized = self._normalized_field_name(name)
                if normalized in {"src", "source", "input", "inputfile", "inputpath", "path"}:
                    input_payload[name] = source
                elif self._is_generated_write_path_field(normalized, output, True):
                    input_payload[name] = output
            if input_payload:
                return (
                    " 다음 액션은 이 도구를 새로 만들지 말고 반드시 call_tool로 실행하세요: "
                    f"{json.dumps({'action': 'call_tool', 'name': action.spec.name, 'input': input_payload}, ensure_ascii=False)}"
                )
        return " 다음 액션은 이 도구를 새로 만들지 말고 call_tool로 실행하세요."

    def _json_scalar_paths(self, value: Any, max_paths: int = 20) -> list[str]:
        paths: list[str] = []

        def walk(current: Any, path: str, depth: int) -> None:
            if len(paths) >= max_paths or depth > 8:
                return
            if isinstance(current, dict):
                for key, item in current.items():
                    next_path = f"{path}.{key}" if path else str(key)
                    if isinstance(item, dict | list):
                        walk(item, next_path, depth + 1)
                    else:
                        paths.append(next_path)
                        if len(paths) >= max_paths:
                            return
                return
            if isinstance(current, list):
                for item in current[:3]:
                    walk(item, f"{path}[]" if path else "[]", depth + 1)
                    if len(paths) >= max_paths:
                        return

        walk(value, "", 0)
        return paths

    def _initial_file_context_observation(self, request: str) -> str | None:
        structure_hint = self._workspace_structure_hint(request)
        if not structure_hint:
            return None
        return (
            "요청에 나온 작업 영역 파일을 미리 확인했습니다. 이 구조를 기준으로 코드를 작성하고, "
            "파일 구조를 사용자에게 다시 묻지 마세요.\n"
            f"{structure_hint}"
        )

    def _tool_failure_recovery_hint(self, request: str) -> str | None:
        structure_hint = self._workspace_structure_hint(request)
        if not structure_hint:
            return None
        return (
            "파일 처리 코드가 실패했습니다. 같은 가정으로 재시도하지 말고, 아래 실제 구조를 "
            "기준으로 코드를 고치세요. root.children 형태의 object tree면 직계 children만 "
            "보지 말고 모든 descendants를 재귀 순회하세요.\n"
            f"{structure_hint}"
        )

    def _mentioned_data_paths(self, text: str) -> list[str]:
        paths: list[str] = []
        for match in _DATA_PATH_RE.finditer(text):
            path = match.group(0).strip("`'\",:;()[]{}")
            if path and path not in paths:
                paths.append(path)
        return paths

    def _outside_workspace_write_request_observation(self, request: str) -> str | None:
        if not self._request_mentions_file_write(request):
            return None
        for path in self._mentioned_data_paths(request):
            if self._path_escapes_workspace(path):
                if self.policy is not None:
                    decision = self.policy.evaluate("out_of_workspace")
                    self.tracer.log(
                        kind="policy_decision",
                        policy=decision.decision,
                        policyReason=decision.reason,
                        toolName="request",
                    )
                return (
                    f"정책상 거부됨: out_of_workspace. 요청한 저장 경로 '{path}'은(는) "
                    "작업 영역 밖을 가리키므로 파일을 만들지 않았습니다."
                )
        return None

    def _tool_error_recovery_hint(
        self, tool_name: str, error: str | None, request: str
    ) -> str | None:
        if not error:
            return None
        tool = self.deps.registry.get(tool_name)
        is_generated_tool = tool is not None and tool.origin == "generated"
        if "NameError: name 'json' is not defined" in error:
            if is_generated_tool:
                return (
                    "이 실패는 json 모듈 import 누락입니다. 같은 코드를 반복하지 말고 "
                    "update_tool로 생성 도구 코드 맨 위에 `import json`을 추가한 뒤 다시 "
                    "call_tool로 실행하세요."
                )
            return (
                "이 실패는 json 모듈 import 누락입니다. 같은 코드를 반복하지 말고 코드 맨 위에 "
                "`import json`을 추가해 runPython을 다시 실행하세요."
            )
        if is_generated_tool and "KeyError" in error and "input[" in error:
            paths = self._mentioned_workspace_paths(request)
            path_hint = f" 요청 파일: {', '.join(paths)}." if paths else ""
            return (
                "생성 도구의 input dict에는 call_tool payload만 들어오며, workspace 파일 내용이 "
                "자동으로 주입되지 않습니다. update_tool로 생성 도구가 요청 파일을 직접 열도록 "
                "고치세요. 예: json 파일은 `import json` 후 `data = json.load(open('파일명.json'))` "
                "또는 path 입력을 받아 `json.load(open(input['path']))`로 읽으세요."
                f"{path_hint}"
            )
        if tool_name == "runPython" and "runPython 안에서 파일을 직접 쓰려고 했습니다" in error:
            if not self._request_mentions_file_write(request):
                aggregate_hint = ""
                if self._request_mentions_amount_by_type_sum(request):
                    aggregate_hint = (
                        " 이 요청은 type별 amount 합계입니다. 평균, HP, count가 아니라 완전히 "
                        "중복된 전체 CSV 행을 tuple(row.items()) 등으로 한 번만 세고, "
                        "type별 amount 합계를 print(json.dumps(...))로 출력하세요."
                    )
                return (
                    "현재 요청은 저장/수정 요청이 아닌 읽기 전용 계산입니다. output_path 인자, "
                    "open(..., 'w'), writerow/writerows, writeFile 호출, cleaned CSV 생성 코드를 "
                    "모두 제거하고 입력 파일을 읽어서 최종 계산 결과만 stdout으로 출력하세요."
                    f"{aggregate_hint}"
                )
            return (
                "runPython에서는 파일을 직접 쓰지 않습니다. 변환 결과 content를 stdout으로 출력한 뒤 "
                "별도 writeFile 호출로 저장하세요."
            )
        return None

    def _request_mentions_file_write(self, request: str) -> bool:
        if self._request_forbids_file_write(request):
            return False
        lowered = request.lower()
        compact = lowered.replace(" ", "")
        return any(term in lowered for term in FILE_WRITE_INTENT_TERMS) or any(
            term in compact
            for term in (
                "asafile",
                "outputfile",
                "파일로",
                "출력파일",
            )
        )

    def _request_mentions_amount_by_type_sum(self, request: str) -> bool:
        lowered = request.lower()
        return (
            "amount" in lowered
            and "type" in lowered
            and any(term in lowered for term in ("sum", "total", "합계", "합산"))
        )

    def _request_forbids_file_write(self, request: str) -> bool:
        lowered = request.lower()
        compact = lowered.replace(" ", "")
        return any(
            term in lowered
            for term in (
                "write or update 하지",
                "write/update 하지",
                "write 하지",
                "update 하지",
                "do not write",
                "don't write",
                "no write",
                "read only",
                "읽기 전용",
                "저장하지",
                "저장 하지",
                "수정하지",
                "수정 하지",
                "업데이트하지",
                "업데이트 하지",
                "쓰지",
                "쓰지는",
                "파일로 저장하지",
                "파일 만들지",
            )
        ) or any(
            term in compact
            for term in (
                "writeorupdate하지",
                "write/update하지",
                "write하지",
                "update하지",
                "저장하지",
                "수정하지",
                "업데이트하지",
                "파일로저장하지",
                "파일만들지",
            )
        )

    def _request_requires_new_tool(self, request: str) -> bool:
        lowered = request.lower()
        compact = re.sub(r"\s+", "", lowered)
        if any(term in compact for term in ("기존도구", "저장된도구", "existingtool", "savedtool")):
            return False
        return any(
            term in compact
            for term in (
                "도구를만",
                "도구만",
                "도구생",
                "tool을만",
                "tool만",
                "tool생",
                "makeatool",
                "buildatool",
                "createatool",
                "generateatool",
            )
        )

    def _has_created_tool_this_turn(self, state: _TurnLoopState) -> bool:
        return bool(state.turn_state.changed_tool_names)

    def _has_called_changed_tool(self, state: _TurnLoopState) -> bool:
        changed = set(state.turn_state.changed_tool_names or [])
        return bool(changed & state.called_changed_tools)

    def _has_called_generated_tool(self, state: _TurnLoopState) -> bool:
        return bool(state.called_generated_tools)

    def _request_prefers_managed_tool(self, request: str) -> bool:
        if self.generated is None:
            return False
        paths = self._mentioned_workspace_paths(request)
        if not paths:
            return False
        lowered = request.lower()
        compact = re.sub(r"\s+", "", lowered)
        if "runpython" in lowered or "run python" in lowered:
            return False
        has_data_task = any(term in lowered for term in DATA_TOOL_TASK_TERMS) or any(
            term in compact for term in DATA_TOOL_TASK_TERMS
        )
        if has_data_task:
            return True
        if len(paths) >= 2 and not self._request_mentions_file_write(request):
            return True
        return False

    def _managed_tool_observation(self, state: _TurnLoopState) -> str:
        existing_generated = [
            digest.name for digest in self.deps.registry.digests() if digest.origin == "generated"
        ]
        existing_hint = ""
        if existing_generated:
            existing_hint = (
                " 현재 등록된 generated tool 중 요청과 정확히 맞는 것이 있으면 먼저 재사용하세요: "
                f"{', '.join(existing_generated)}."
            )
        return (
            "workspace 파일을 분석/변환/집계하는 작업은 runPython으로 바로 처리하지 않습니다. "
            "요청에 맞는 generated tool이 이미 있으면 call_tool로 재사용하고, 없으면 create_tool로 "
            "관리형 도구를 만든 뒤 그 도구를 call_tool로 실행하세요. runPython은 작은 진단이나 "
            f"검증에만 사용하세요.{existing_hint}"
        )

    def _blocks_managed_tool_default(self, state: _TurnLoopState) -> bool:
        return self._request_prefers_managed_tool(
            state.effective_request
        ) and not self._has_called_generated_tool(state)

    def _payload_mentions_any_path(self, payload: Any, paths: list[str]) -> bool:
        if isinstance(payload, str):
            return payload in paths
        if isinstance(payload, dict):
            return any(self._payload_mentions_any_path(value, paths) for value in payload.values())
        if isinstance(payload, list):
            return any(self._payload_mentions_any_path(value, paths) for value in payload)
        return False

    def _payload_contains_inline_dataset(self, payload: Any) -> bool:
        if isinstance(payload, list):
            return True
        if isinstance(payload, dict):
            lowered_keys = {str(key).lower() for key in payload}
            if len(lowered_keys & {"name", "hp", "atk", "amount", "type", "date", "id"}) >= 2:
                return True
            return any(self._payload_contains_inline_dataset(value) for value in payload.values())
        return False

    def _generated_tool_inline_data_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        if not self._request_prefers_managed_tool(state.effective_request):
            return None
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return None
        paths = self._mentioned_workspace_paths(state.effective_request)
        if not paths or self._payload_mentions_any_path(action.input, paths):
            return None
        if not self._payload_contains_inline_dataset(action.input):
            return None
        return (
            "요청은 workspace 파일 기준 처리입니다. call_tool input에 임의 샘플 데이터나 추정 데이터를 "
            "넣지 마세요. 실제 파일 경로를 넘기거나 생성 도구 코드에서 "
            f"`open('{paths[0]}')`로 workspace 파일을 직접 읽은 뒤 다시 실행하세요."
        )

    def _generated_tool_file_read_observation(
        self, state: _TurnLoopState, name: str, code: str
    ) -> str | None:
        if not self._request_prefers_managed_tool(state.effective_request):
            return None
        lowered = code.lower()
        file_read_markers = (
            "open(",
            "json.load(",
            "csv.reader(",
            "read_csv(",
            "read_json(",
            ".read_text(",
            ".read_bytes(",
        )
        if any(marker in lowered for marker in file_read_markers):
            if not self._request_needs_object_tree_mutation(state.effective_request):
                return None
            if self._code_writes_workspace_file(code):
                return None
            paths = self._mentioned_workspace_paths(state.effective_request)
            source = paths[0] if paths else "요청 파일"
            return (
                f"생성 도구 {name}은(는) 상태 변경 작업용인데 코드가 {source}를 다시 저장하지 않습니다. "
                "평균 계산만 하는 도구를 만들지 말고, 조건에 맞지 않는 node를 children에서 제거한 뒤 "
                f"`json.dump(data, open('{source}', 'w'))` 또는 동등한 파일 쓰기로 저장하고, "
                "저장된 파일을 다시 읽어 결과를 계산하도록 create_tool/update_tool을 다시 작성하세요. "
                "코드는 다음 구조를 그대로 따르세요: "
                "`import json; def run(input): data=json.load(open('"
                f"{source}"
                "')); def prune(node): node['children']=[child for child in node.get('children', []) "
                "if not (child.get('type')=='Entity' and child.get('props', {}).get('health', 0) < 100)]; "
                "[prune(child) for child in node.get('children', [])]; prune(data['root']); "
                "json.dump(data, open('"
                f"{source}"
                "', 'w')); saved=json.load(open('"
                f"{source}"
                "')); ... saved에서 Entity health 평균 계산 후 return`."
            )
        paths = self._mentioned_workspace_paths(state.effective_request)
        source = paths[0] if paths else "요청 파일"
        return (
            f"생성 도구 {name}은(는) workspace 파일 처리용인데 코드가 실제 파일을 읽지 않습니다. "
            "call_tool input으로 파일 내용을 주입받는 도구를 만들지 말고, 도구 코드 안에서 "
            f"`open('{source}')`, `json.load(open(path))`, `csv.reader(open(path))`처럼 "
            "workspace 파일을 직접 읽도록 create_tool/update_tool을 다시 작성하세요."
        )

    def _code_writes_workspace_file(self, code: str) -> bool:
        lowered = code.lower()
        return (
            "json.dump" in lowered
            or ".write_text" in lowered
            or ".write_bytes" in lowered
            or re.search(r"open\([^)]*,\s*['\"][wax+]", code) is not None
            or re.search(r"open\([^)]*,[^)]*mode\s*=\s*['\"][wax+]", code) is not None
        )

    def _explicit_tool_creation_observation(self) -> str:
        return (
            "사용자가 도구 생성을 명시했습니다. runPython이나 기존 도구로 바로 처리하지 말고 "
            "create_tool로 이 요청 전용 생성 도구를 만든 뒤, 그 생성 도구를 call_tool로 실행하세요."
        )

    def _explicit_tool_call_observation(self, state: _TurnLoopState) -> str:
        names = ", ".join(state.turn_state.changed_tool_names or [])
        suffix = f" 생성/수정한 도구: {names}." if names else ""
        return (
            "사용자가 도구 생성을 명시했으므로 방금 만든 generated tool을 반드시 call_tool로 "
            f"실행해야 합니다. runPython, writeFile, 기존 도구로 우회하지 마세요.{suffix}"
        )

    def _updated_tool_needs_execution(self, state: _TurnLoopState, name: str) -> bool:
        if name not in (state.turn_state.changed_tool_names or []):
            return False
        last_change = state.turn_state.last_generated_tool_change
        if last_change is None:
            return False
        return state.last_tool_call_index is None or state.last_tool_call_index < last_change

    def _updated_tool_call_observation(self, name: str, request: str) -> str:
        paths = self._mentioned_workspace_paths(request)
        input_hint = ""
        if paths:
            first_path = paths[0]
            input_hint = (
                f" 요청 파일이 필요하면 call_tool input에 path를 넘기거나 도구 내부에서 "
                f"`open('{first_path}')`로 직접 여세요."
            )
        return (
            f"도구 {name}은(는) 이미 생성/수정했습니다. 같은 update_tool을 반복하지 말고 "
            f"먼저 call_tool로 실행해 실제 오류 또는 결과를 확인하세요.{input_hint}"
        )

    def _tool_needs_update_after_validation_failure(self, state: _TurnLoopState, name: str) -> bool:
        failure_index = state.validation_failed_tool_indices.get(name)
        if failure_index is None:
            return False
        last_change = state.turn_state.last_generated_tool_change
        return last_change is None or last_change <= failure_index

    def _validation_failed_tool_update_observation(
        self, state: _TurnLoopState, name: str | None = None
    ) -> str | None:
        failed_names = [
            tool_name
            for tool_name in state.validation_failed_tool_indices
            if name is None or tool_name == name
        ]
        failed_names = [
            tool_name
            for tool_name in failed_names
            if self._tool_needs_update_after_validation_failure(state, tool_name)
        ]
        if not failed_names:
            return None
        paths = self._mentioned_workspace_paths(state.effective_request)
        path_hint = f" 요청 파일: {', '.join(paths)}." if paths else ""
        return (
            f"생성 도구 {', '.join(failed_names)}의 실행 결과가 검증에 실패했습니다. "
            "같은 도구를 그대로 다시 call_tool 하거나 finish 하지 말고 update_tool로 코드를 수정하세요. "
            "상태 변경 요청이면 평균 계산만 하지 말고 파일을 실제로 다시 저장한 뒤, 저장된 파일을 "
            f"재읽어 결과를 계산해야 합니다.{path_hint}"
        )

    def _blocks_explicit_tool_creation(self, state: _TurnLoopState) -> bool:
        return self._request_requires_new_tool(
            state.effective_request
        ) and not self._has_created_tool_this_turn(state)

    def _blocks_explicit_tool_execution(self, state: _TurnLoopState) -> bool:
        return (
            self._request_requires_new_tool(state.effective_request)
            and self._has_created_tool_this_turn(state)
            and not self._has_called_changed_tool(state)
        )

    def _generated_tool_write_path_fields(
        self, name: str, payload: dict[str, Any]
    ) -> list[tuple[str, str]]:
        tool = self.deps.registry.get(name)
        if tool is None or tool.origin != "generated":
            return []

        fields: list[tuple[str, str]] = []
        description_suggests_write = self._description_suggests_file_write(tool.description)
        for key, value in payload.items():
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_key = self._normalized_field_name(str(key))
            if self._is_generated_write_path_field(
                normalized_key, value, description_suggests_write
            ):
                fields.append((str(key), value))
        return fields

    def _is_generated_write_path_field(
        self, normalized_key: str, value: str, description_suggests_write: bool
    ) -> bool:
        if normalized_key in GENERATED_WRITE_PATH_FIELD_NAMES:
            if normalized_key in AMBIGUOUS_WRITE_PATH_FIELD_NAMES:
                return description_suggests_write or self._looks_like_file_path(value)
            return True
        return normalized_key == "path" and description_suggests_write

    def _description_suggests_file_write(self, description: str) -> bool:
        lowered = description.lower()
        return any(term in lowered for term in FILE_WRITE_INTENT_TERMS)

    def _normalized_field_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _looks_like_file_path(self, value: str) -> bool:
        stripped = value.strip()
        return (
            stripped.startswith(("/", "~", "./", "../"))
            or "/" in stripped
            or "\\" in stripped
            or bool(Path(stripped).suffix)
        )

    def _path_escapes_workspace(self, path: str) -> bool:
        return path.startswith("/") or path.startswith("~") or ".." in Path(path).parts

    def _gate_file_write_path(self, name: str, path: str) -> tuple[bool, str | None]:
        escapes = self._path_escapes_workspace(path)
        action_id = "out_of_workspace" if escapes else "write_file"
        if action_id == "write_file" and path in self._confirmed_write_paths:
            return True, None

        if self.policy is None:
            if action_id == "out_of_workspace":
                return False, f"정책상 거부됨: {action_id}"
            self._confirmed_write_paths.add(path)
            return True, None

        decision = self.policy.evaluate(action_id)
        self.tracer.log(
            kind="policy_decision",
            policy=decision.decision,
            policyReason=decision.reason,
            toolName=name,
        )
        if decision.decision == "DENY":
            return False, f"정책상 거부됨: {action_id}"
        if decision.decision == "ASK_USER" and not self.policy.confirm(action_id):
            return False, "사용자가 작업을 거부했습니다."
        if action_id == "write_file":
            self._confirmed_write_paths.add(path)
        return True, None

    def _gate_file_write_intent(self, name: str) -> tuple[bool, str | None]:
        if self.policy is None:
            return True, None
        decision = self.policy.evaluate("write_file")
        self.tracer.log(
            kind="policy_decision",
            policy=decision.decision,
            policyReason=decision.reason,
            toolName=name,
        )
        if decision.decision == "ASK_USER" and not self.policy.confirm("write_file"):
            return False, "사용자가 작업을 거부했습니다."
        return True, None

    def _gate_generated_file_write_path(self, name: str, path: str) -> tuple[bool, str | None]:
        """Generated tools may use workspace files as scratch/output, but never escape."""
        if not self._path_escapes_workspace(path):
            return True, None
        if self.policy is None:
            return False, "정책상 거부됨: out_of_workspace"
        decision = self.policy.evaluate("out_of_workspace")
        self.tracer.log(
            kind="policy_decision",
            policy=decision.decision,
            policyReason=decision.reason,
            toolName=name,
        )
        return False, "정책상 거부됨: out_of_workspace"

    def _is_side_effect_tool(self, name: str) -> bool:
        return name == "writeFile"

    def _cached_tool_summary(self, name: str, result: ToolResult) -> str:
        if isinstance(result.output, dict):
            path = result.output.get("path")
            if path:
                return f"{name} 실행은 이미 성공했습니다. 결과 파일: {path}"
        return f"{name} 실행은 이미 성공했습니다. 마지막 결과: {result.output}"

    def _cached_tool_observation(self, name: str, result: ToolResult) -> str:
        return f"도구 {name} 캐시 결과: {result.output}"

    def _log_parsed_action(self, action: Any) -> None:
        fields: dict[str, Any] = {"actionType": action.action, "parseOk": True}
        if isinstance(action, CallTool):
            tool_input, input_chars, input_truncated = _preview_value(action.input)
            fields.update(
                {
                    "toolName": action.name,
                    "toolInput": tool_input,
                    "toolInputChars": input_chars,
                    "toolInputTruncated": input_truncated,
                }
            )
        self.tracer.log(kind="llm_call", **fields)

    def _log_tool_call(self, name: str, payload: dict[str, Any], result: ToolResult) -> None:
        tool_input, input_chars, input_truncated = _preview_value(payload)
        fields: dict[str, Any] = {
            "toolName": name,
            "toolOk": result.ok,
            "toolInput": tool_input,
            "toolInputChars": input_chars,
            "toolInputTruncated": input_truncated,
        }
        if result.ok:
            tool_output, output_chars, output_truncated = _preview_value(result.output)
            fields.update(
                {
                    "toolOutput": tool_output,
                    "toolOutputChars": output_chars,
                    "toolOutputTruncated": output_truncated,
                }
            )
        else:
            tool_error, error_chars, error_truncated = _preview_value(result.error or "")
            fields.update(
                {
                    "toolError": tool_error,
                    "toolErrorChars": error_chars,
                    "toolErrorTruncated": error_truncated,
                }
            )
        self.tracer.log(kind="tool_call", **fields)

    def _run_python_direct_write_paths(self, action: CallTool) -> list[str]:
        if action.name != "runPython" or "code" not in action.input:
            return []
        try:
            tree = ast.parse(str(action.input.get("code", "")))
        except SyntaxError:
            return []
        paths: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            path = self._open_write_path(node) or self._pathlib_write_path(node)
            if path is not None and path not in paths:
                paths.append(path)
        return paths

    def _open_write_path(self, node: ast.Call) -> str | None:
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
        return UNKNOWN_DYNAMIC_WRITE_PATH

    def _pathlib_write_path(self, node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr not in {
            "write_text",
            "write_bytes",
            "touch",
            "mkdir",
            "unlink",
            "rename",
        }:
            return None
        value = node.func.value
        if isinstance(value, ast.Call):
            return self._path_constructor_arg(value)
        return UNKNOWN_DYNAMIC_WRITE_PATH

    def _path_constructor_arg(self, node: ast.Call) -> str:
        if not node.args:
            return UNKNOWN_DYNAMIC_WRITE_PATH
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            return self._path_arg_label(node.args[0])
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "Path"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pathlib"
        ):
            return self._path_arg_label(node.args[0])
        return UNKNOWN_DYNAMIC_WRITE_PATH

    def _path_arg_label(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return UNKNOWN_DYNAMIC_WRITE_PATH

    def _gate_run_python_direct_writes(
        self, request: str, action: CallTool
    ) -> tuple[bool, str | None]:
        paths = self._run_python_direct_write_paths(action)
        if not paths:
            return True, None
        for path in paths:
            if path != UNKNOWN_DYNAMIC_WRITE_PATH and self._path_escapes_workspace(path):
                return self._gate_file_write_path(action.name, path)
        if not (
            self._request_mentions_file_write(request)
            or self._request_needs_object_tree_mutation(request)
        ):
            return (
                False,
                "현재 요청은 저장/수정 요청이 아닌 읽기 전용 계산입니다. runPython에서 파일을 "
                "직접 쓰지 말고 입력 파일을 읽어 최종 계산 결과만 stdout으로 출력하세요.",
            )
        for path in paths:
            if path == UNKNOWN_DYNAMIC_WRITE_PATH:
                allowed, blocked = self._gate_file_write_intent(action.name)
            else:
                allowed, blocked = self._gate_file_write_path(action.name, path)
            if not allowed:
                return allowed, blocked
        return True, None

    def _plan_raw(self) -> str:
        with self.tracer.span():
            digests = self.deps.registry.digests()
            self.tracer.log(kind="llm_call_start", model=getattr(self.deps.llm, "model", None))
            try:
                raw = self._chat_with_deadline(digests)
            except Exception as exc:
                self.tracer.log(
                    kind="error",
                    errorKind="llm_call_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    model=getattr(self.deps.llm, "model", None),
                )
                raise
            preview, truncated = _preview_text(raw)
            self.tracer.log(
                kind="llm_call",
                model=getattr(self.deps.llm, "model", None),
                responsePreview=preview,
                responseChars=len(raw),
                responseTruncated=truncated,
            )
            return raw

    def _chat_with_deadline(self, digests: list[Any]) -> str:
        timeout = getattr(self.deps.llm, "timeout", None)
        if timeout is None:
            return self.deps.llm.chat(self.conv.messages(), digests)
        timeout_sec = float(timeout)
        result_queue: queue.Queue[tuple[str, str | Exception]] = queue.Queue(maxsize=1)
        messages = self.conv.messages()

        def call_llm() -> None:
            try:
                result_queue.put(("ok", self.deps.llm.chat(messages, digests)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=call_llm, daemon=True)
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            raise TimeoutError(f"LLM 호출이 {timeout_sec:g}초 안에 응답하지 않았습니다.")
        status, value = result_queue.get_nowait()
        if status == "error":
            assert isinstance(value, Exception)
            raise value
        assert isinstance(value, str)
        return value

    def _gate(self, name: str, payload: dict[str, Any]) -> tuple[bool, str | None]:
        """Consult the policy before a side-effecting tool call.

        Returns (allowed, observation_if_blocked). File-writing tools are gated:
        an out-of-workspace target is denied outright; an in-workspace write asks
        the user. Other tools are allowed.
        """
        if name == "writeFile":
            path = str(payload.get("path", ""))
        else:
            return True, None
        return self._gate_file_write_path(name, path)

    def run_turn(self, request: str) -> TurnResult:
        state = self._start_turn(request)
        with self.tracer.trace():
            outside_write = self._outside_workspace_write_request_observation(request)
            if outside_write is not None:
                self._append_observation(state, outside_write)
                state.result.summary = outside_write
                state.result.stopped_reason = "finish"
                self._finish_turn(state.result)
                return state.result

            for _ in range(self.deps.max_iterations):
                try:
                    raw = self._plan_raw()
                except Exception as exc:
                    state.result.summary = self._llm_error_summary(exc)
                    state.result.stopped_reason = "llm_error"
                    break
                parsed = parse_action_text(raw)
                if not parsed.ok or parsed.action is None:
                    step = self._handle_parse_failure(state, parsed.error)
                    if step == "break":
                        break
                    continue

                state.parse_failures = 0
                action = parsed.action
                state.turn_state.advance()
                self._log_parsed_action(action)

                sig, step = self._apply_loop_guards(state, action)
                if step == "break":
                    break
                if step == "continue":
                    continue

                step = self._dispatch_action(state, action, sig)
                if step == "break":
                    break
            else:
                state.result.stopped_reason = "max_iterations"

            self._finish_turn(state.result)
        return state.result

    def _llm_error_summary(self, exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return (
                "모델 응답이 제한 시간 안에 돌아오지 않아 중단했습니다. "
                "다시 시도하거나 AGENT_LLM_TIMEOUT_SEC 값을 조정하세요."
            )
        return f"모델 호출 실패로 중단했습니다: {type(exc).__name__}: {exc}"

    def _start_turn(self, request: str) -> _TurnLoopState:
        self.tracer.log(kind="turn_start")
        self.conv.add_user(request)
        result = TurnResult()
        state = _TurnLoopState(
            result=result,
            turn_state=TurnExecutionState(
                requires_execution=bool(self._mentioned_workspace_paths(request))
            ),
            effective_request=request,
        )
        if initial_file_context := self._initial_file_context_observation(request):
            self._append_observation(state, initial_file_context)
        return state

    def _append_observation(self, state: _TurnLoopState, observation: str) -> None:
        self.conv.add_observation(observation)
        state.result.observations.append(observation)

    def _handle_parse_failure(self, state: _TurnLoopState, error: str | None) -> _StepAction:
        state.parse_failures += 1
        observation = error or "알 수 없는 파싱 오류"
        self.tracer.log(kind="llm_call", parseOk=False)
        self._append_observation(state, observation)
        if state.parse_failures > self.deps.max_fix_retries:
            state.result.stopped_reason = "parse_failures"
            return "break"
        return "continue"

    def _apply_loop_guards(
        self, state: _TurnLoopState, action: Any
    ) -> tuple[str | None, _StepAction]:
        sig = self._action_signature(action)
        if sig is not None and sig == state.last_action_sig:
            state.action_repeats += 1
        else:
            state.action_repeats = 0
            state.last_action_sig = sig

        if (
            isinstance(action, CallTool)
            and sig is not None
            and sig in self._tool_result_cache
            and not self._tool_needs_update_after_validation_failure(state, action.name)
            and state.action_repeats > self.deps.max_fix_retries
        ):
            cached = self._tool_result_cache[sig]
            state.result.summary = self._cached_tool_summary(action.name, cached)
            state.result.stopped_reason = "cached_result"
            return sig, "break"

        if (
            isinstance(action, CallTool)
            and sig is not None
            and sig in self._tool_result_cache
            and not self._tool_needs_update_after_validation_failure(state, action.name)
            and state.action_repeats > 0
        ):
            cached = self._tool_result_cache[sig]
            self._append_observation(state, self._cached_tool_observation(action.name, cached))
            return sig, "continue"

        if state.action_repeats > self.deps.max_fix_retries:
            # Keep this runtime note out of result.observations so incomplete
            # finalization can still surface the last real result if there was one.
            self.conv.add_observation("같은 동작이 진전 없이 반복되어 작업을 중단합니다.")
            state.result.stopped_reason = "no_progress"
            return sig, "break"

        if (
            isinstance(action, CallTool)
            and state.action_repeats > 0
            and self._is_side_effect_tool(action.name)
        ):
            self.conv.add_observation("같은 부수효과 도구 호출이 반복되어 작업을 중단합니다.")
            state.result.stopped_reason = "no_progress"
            return sig, "break"

        return sig, "dispatch"

    def _dispatch_action(self, state: _TurnLoopState, action: Any, sig: str | None) -> _StepAction:
        if isinstance(action, Finish):
            return self._handle_finish_action(state, action)
        if isinstance(action, Respond):
            return self._handle_respond_action(state, action)
        if isinstance(action, AskUser):
            return self._handle_ask_user_action(state, action)
        if isinstance(action, CallTool):
            return self._handle_call_tool_action(state, action, sig)
        if isinstance(action, CreateTool):
            return self._handle_create_tool_action(state, action)
        if isinstance(action, UpdateTool):
            return self._handle_update_tool_action(state, action)
        return "continue"

    def _terminal_block(self, state: _TurnLoopState) -> str | None:
        return state.turn_state.terminal_block_observation()

    def _handle_finish_action(self, state: _TurnLoopState, action: Finish) -> _StepAction:
        if self._blocks_explicit_tool_creation(state):
            self._append_observation(state, self._explicit_tool_creation_observation())
            return "continue"
        if self._blocks_explicit_tool_execution(state):
            self._append_observation(state, self._explicit_tool_call_observation(state))
            return "continue"
        if validation_failed := self._validation_failed_tool_update_observation(state):
            self._append_observation(state, validation_failed)
            return "continue"
        if validation := self._terminal_validation_observation(state):
            self._append_observation(state, validation)
            return "continue"
        if terminal_block := self._terminal_block(state):
            self._append_observation(state, terminal_block)
            return "continue"
        if self._blocks_managed_tool_default(state):
            self._append_observation(state, self._managed_tool_observation(state))
            return "continue"
        state.result.summary = action.summary or ""
        state.result.stopped_reason = "finish"
        return "break"

    def _handle_respond_action(self, state: _TurnLoopState, action: Respond) -> _StepAction:
        respond_is_final = action.final is not False
        if respond_is_final:
            if self._blocks_explicit_tool_creation(state):
                self._append_observation(state, self._explicit_tool_creation_observation())
                return "continue"
            if self._blocks_explicit_tool_execution(state):
                self._append_observation(state, self._explicit_tool_call_observation(state))
                return "continue"
            if validation_failed := self._validation_failed_tool_update_observation(state):
                self._append_observation(state, validation_failed)
                return "continue"
            repeated_request = self._repeated_actionable_request_observation(
                action.text, state.effective_request
            )
            if repeated_request is not None:
                self._append_observation(state, repeated_request)
                return "continue"
            if validation := self._terminal_validation_observation(state):
                self._append_observation(state, validation)
                return "continue"
            if terminal_block := self._terminal_block(state):
                self._append_observation(state, terminal_block)
                return "continue"
            if self._blocks_managed_tool_default(state):
                self._append_observation(state, self._managed_tool_observation(state))
                return "continue"

        self.conv.add_assistant(action.text)
        if respond_is_final:
            state.result.summary = action.text
            state.result.stopped_reason = "finish"
            return "break"
        self._append_observation(state, "계속 진행하세요.")
        return "continue"

    def _terminal_validation_observation(self, state: _TurnLoopState) -> str | None:
        validation = self._validate_completed_work(state.effective_request)
        if validation is None:
            return None
        ok, message = validation
        return None if ok else message

    def _handle_ask_user_action(self, state: _TurnLoopState, action: AskUser) -> _StepAction:
        blocked_ask = self._blocked_ask_user_observation(action.question, state.effective_request)
        if blocked_ask is not None:
            self._append_observation(state, blocked_ask)
            return "continue"
        if self.deps.non_interactive:
            state.result.summary = self._hitl_required_summary(action.question)
            state.result.stopped_reason = "hitl_required"
            return "break"
        answer = self.deps.ask(action.question, action.choices)
        if answer == NON_INTERACTIVE_ASK:
            state.result.summary = self._hitl_required_summary(action.question)
            state.result.stopped_reason = "hitl_required"
            return "break"
        state.effective_request = f"{state.effective_request}\n{answer}"
        self._append_observation(state, f"사용자 답변: {answer}")
        return "continue"

    def _handle_call_tool_action(
        self, state: _TurnLoopState, action: CallTool, sig: str | None
    ) -> _StepAction:
        if self._blocks_explicit_tool_creation(state) and action.name not in {
            "readFile",
            "listFiles",
            "searchDocs",
            "askUser",
        }:
            self._append_observation(state, self._explicit_tool_creation_observation())
            return "continue"

        if self._blocks_explicit_tool_execution(state) and action.name not in {
            *(state.turn_state.changed_tool_names or []),
            "readFile",
            "listFiles",
            "searchDocs",
            "askUser",
        }:
            self._append_observation(state, self._explicit_tool_call_observation(state))
            return "continue"

        if (
            action.name == "runPython"
            and self._blocks_managed_tool_default(state)
            and not self._request_requires_new_tool(state.effective_request)
        ):
            self._append_observation(state, self._managed_tool_observation(state))
            return "continue"

        if validation_failed := self._validation_failed_tool_update_observation(state, action.name):
            self._append_observation(state, validation_failed)
            return "continue"

        inline_data_block = self._generated_tool_inline_data_observation(state, action)
        if inline_data_block is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._append_observation(state, inline_data_block)
            return "continue"

        generated_write_fields = self._generated_tool_write_path_fields(action.name, action.input)
        for _, path in generated_write_fields:
            allowed, blocked = self._gate_generated_file_write_path(action.name, path)
            if not allowed:
                assert blocked is not None
                self._append_observation(state, blocked)
                return "continue"

        if action.name == "writeFile" and not isinstance(action.input.get("content"), str):
            self._append_observation(
                state,
                "writeFile content는 문자열이어야 합니다. content:null 또는 final:true로 파일 쓰기를 "
                "호출하지 마세요. 현재 요청이 읽기 전용 계산이면 writeFile을 쓰지 말고 최종 답변으로 "
                "결과를 반환하세요.",
            )
            return "continue"

        if action.name == "writeFile" and self._request_forbids_file_write(state.effective_request):
            self._append_observation(
                state,
                "현재 요청은 파일 쓰기를 금지합니다. writeFile을 실행하지 말고 "
                "계산 결과를 최종 답변으로 반환하세요.",
            )
            return "continue"

        if action.name == "runPython":
            allowed, blocked = self._gate_run_python_direct_writes(state.effective_request, action)
            if not allowed:
                assert blocked is not None
                self._append_observation(state, blocked)
                return "continue"

        if (
            action.name != "searchDocs"
            and self.deps.registry.has("searchDocs")
            and self._request_needs_object_tree_mutation(state.effective_request)
            and not state.used_search_docs
        ):
            self._append_observation(
                state,
                "이 작업은 문서 근거가 필요한 object tree 상태 변경입니다. 먼저 searchDocs를 "
                "호출해 Entity health와 children 구조를 확인한 뒤 파일을 수정하세요.",
            )
            return "continue"

        missing_mutation = self._run_python_missing_required_mutation_observation(
            state.effective_request, action
        )
        if missing_mutation is not None:
            self._append_observation(state, missing_mutation)
            return "continue"

        if action.name == "writeFile":
            table_validation = self._previous_json_filter_table_validation(
                state.effective_request,
                str(action.input.get("path", "")),
                str(action.input.get("content", "")),
            )
            if table_validation is not None and not table_validation[0]:
                self._append_observation(state, table_validation[1])
                return "continue"

        if self.policy is not None:
            allowed, blocked = self._gate(action.name, action.input)
            if not allowed:
                assert blocked is not None
                self._append_observation(state, blocked)
                return "continue"

        restore_snapshots = self._mutation_restore_snapshots(state.effective_request, action)
        res = self.deps.registry.call(action.name, action.input)
        self._log_tool_call(action.name, action.input, res)
        if res.ok:
            state.fix_failures = 0
            if action.name == "searchDocs":
                state.used_search_docs = True
            assert sig is not None
            observation = f"도구 {action.name} 결과: {res.output}"
            self._tool_result_cache[sig] = res
        else:
            state.fix_failures += 1
            observation = f"도구 {action.name} 실패: {res.error}"
            error_hint = self._tool_error_recovery_hint(
                action.name, res.error, state.effective_request
            )
            if error_hint is not None:
                observation = f"{observation}\n{error_hint}"
            recovery_hint = self._tool_failure_recovery_hint(state.effective_request)
            if recovery_hint is not None:
                observation = f"{observation}\n{recovery_hint}"

        state.turn_state.record_tool_call(action.name, res.ok)
        state.last_tool_call_index = state.turn_state.action_index
        called_tool = self.deps.registry.get(action.name)
        if (
            res.ok
            and called_tool is not None
            and called_tool.origin == "generated"
            and action.name in (state.turn_state.changed_tool_names or [])
        ):
            state.called_changed_tools.add(action.name)
        if res.ok and called_tool is not None and called_tool.origin == "generated":
            state.called_generated_tools.add(action.name)
        self._append_observation(state, observation)
        if not res.ok and state.fix_failures > self.deps.max_fix_retries:
            self._append_observation(state, "연속 실패가 한계를 넘었습니다. 작업을 중단합니다.")
            state.result.stopped_reason = "consecutive_failures"
            return "break"
        if res.ok and action.name not in {"readFile", "listFiles", "searchDocs"}:
            post_validation = self._validate_completed_work(state.effective_request, action, res)
            if post_validation is not None:
                ok, message = post_validation
                if not ok and restore_snapshots:
                    repair = self._try_repair_completed_work(
                        state.effective_request, restore_snapshots
                    )
                    if repair is not None:
                        ok, message = repair
                    else:
                        self._restore_workspace_files(restore_snapshots)
                        message = f"{message} 잘못된 파일 변경은 원본 상태로 되돌렸습니다."
                self._append_observation(state, message)
                if ok:
                    output_path = self._likely_output_path(
                        state.effective_request, Path(message.split()[0]).suffix
                    )
                    if output_path is not None:
                        state.completed_output_paths.add(output_path)
                    state.result.summary = message
                    state.result.stopped_reason = "finish"
                    return "break"
                if called_tool is not None and called_tool.origin == "generated":
                    state.validation_failed_tool_indices[action.name] = (
                        state.turn_state.action_index
                    )
                    update_hint = self._validation_failed_tool_update_observation(
                        state, action.name
                    )
                    if update_hint is not None:
                        self._append_observation(state, update_hint)
        return "continue"

    def _handle_create_tool_action(self, state: _TurnLoopState, action: CreateTool) -> _StepAction:
        file_read_block = self._generated_tool_file_read_observation(
            state, action.spec.name, action.spec.code
        )
        if file_read_block is not None:
            self._append_observation(state, file_read_block)
            return "continue"
        if (
            state.turn_state.changed_tool_names is not None
            and action.spec.name in state.turn_state.changed_tool_names
        ):
            self._append_observation(
                state,
                f"도구 {action.spec.name}은(는) 이미 이 턴에서 생성했습니다."
                + self._tool_call_next_step_hint(action, state.effective_request),
            )
            return "continue"
        self._tool_result_cache.clear()
        observation = self._handle_create(action) + self._tool_call_next_step_hint(
            action, state.effective_request
        )
        state.turn_state.record_tool_change(action.spec.name)
        state.validation_failed_tool_indices.pop(action.spec.name, None)
        self._append_observation(state, observation)
        return "continue"

    def _handle_update_tool_action(self, state: _TurnLoopState, action: UpdateTool) -> _StepAction:
        file_read_block = self._generated_tool_file_read_observation(
            state, action.name, action.code
        )
        if file_read_block is not None:
            self._append_observation(state, file_read_block)
            return "continue"
        if self._updated_tool_needs_execution(state, action.name):
            self._append_observation(
                state,
                self._updated_tool_call_observation(action.name, state.effective_request),
            )
            return "continue"
        self._tool_result_cache.clear()
        observation = self._handle_update(action)
        state.turn_state.record_tool_change(action.name)
        state.validation_failed_tool_indices.pop(action.name, None)
        self._append_observation(state, observation)
        return "continue"

    def _finish_turn(self, result: TurnResult) -> None:
        if not result.summary and result.stopped_reason != "finish":
            self._finalize_incomplete(result)
        if result.stopped_reason == "finish":
            self._offer_persist()
        else:
            self._session_tools.clear()
        self.ctx.maybe_compact(self.conv)

    def _finalize_incomplete(self, result: TurnResult) -> None:
        """Surface a result when the loop ends without a completion signal.

        A weak model sometimes runs the tools that produce the answer but never
        emits respond(final)/finish, so it burns through the iteration budget. In
        that case an empty summary hides the work entirely; instead we build a
        user-facing fallback and record the stop reason as an error event so the
        run is traceable from the log alone.
        """
        last = result.observations[-1] if result.observations else ""
        visible = self._visible_incomplete_observation(last)
        message = self._incomplete_stop_message(result.stopped_reason)
        result.summary = message + (f"\n마지막 결과: {visible}" if visible else "")
        self.tracer.log(kind="error", errorKind=result.stopped_reason, message=result.summary)

    def _incomplete_stop_message(self, stopped_reason: str) -> str:
        """Map internal stop reasons to concise user-facing text."""
        if stopped_reason == "no_progress":
            return "작업이 진전 없이 반복되어 중단했습니다."
        if stopped_reason == "max_iterations":
            return "반복 한도 안에 작업을 끝내지 못했습니다."
        if stopped_reason == "consecutive_failures":
            return "도구 실행이 반복해서 실패해 중단했습니다."
        if stopped_reason == "parse_failures":
            return "모델 출력이 계속 action JSON 형식을 어겨 중단했습니다."
        return f"작업을 끝내지 못했습니다. 종료 사유: {stopped_reason}"

    def _visible_incomplete_observation(self, observation: str) -> str:
        """Strip runtime-only observation text before surfacing a fallback."""
        if not observation:
            return ""
        hidden_markers = (
            "작업 영역 파일의 구조는 사용자에게 묻지 않습니다.",
            "패키지 설치 질문은 사용자에게 묻지 않습니다.",
            "같은 동작이 진전 없이 반복되어",
            "계속 진행하세요.",
            "생성·등록 완료",
            "수정 완료",
            "사용자 답변:",
            "이미 이 턴에서 생성했습니다.",
        )
        if any(marker in observation for marker in hidden_markers):
            return ""
        if observation.startswith("도구 runPython 결과: "):
            payload = observation.removeprefix("도구 runPython 결과: ")
            try:
                value = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return observation
            if isinstance(value, dict):
                stdout = str(value.get("stdout", "")).strip()
                if stdout:
                    return stdout
        return observation

    def _handle_create(self, action: CreateTool) -> str:
        if self.generated is None:
            return "도구 생성기가 구성되지 않았습니다."
        self.deps.registry.register(self.generated.create(action.spec))
        self._session_tools.append(action.spec.name)
        self.ctx.carry_over_fact(f"생성한 도구: {action.spec.name}")
        self.tracer.log(kind="tool_create", toolName=action.spec.name)
        return f"도구 {action.spec.name} 생성·등록 완료"

    def _handle_update(self, action: UpdateTool) -> str:
        if self.generated is None:
            return "도구 생성기가 구성되지 않았습니다."
        if action.name not in self.generated.specs():
            return (
                f"'{action.name}'은(는) 수정할 수 있는 생성 도구가 아닙니다. "
                "내장 도구는 call_tool로 호출하고, 새 도구는 create_tool로 만드세요."
            )
        tool = self.generated.update(action.name, action.code)
        self.deps.registry.register(tool)
        if action.name not in self._session_tools:
            self._session_tools.append(action.name)
        self.ctx.carry_over_fact(f"수정한 도구: {action.name}")
        self.tracer.log(kind="tool_update", toolName=action.name)
        return f"도구 {action.name} 수정 완료"

    def _offer_persist(self) -> None:
        if self.generated is None or self.skills is None or self.policy is None:
            self._session_tools.clear()
            return
        for name in self._session_tools:
            decision = self.policy.evaluate("persist_tool")
            self.tracer.log(
                kind="policy_decision",
                policy=decision.decision,
                policyReason=decision.reason,
                toolName=name,
            )
            if decision.decision == "DENY":
                continue
            if decision.decision == "ASK_USER" and not self.policy.confirm(f"persist:{name}"):
                continue
            self.skills.persist(self.generated.specs()[name])
        self._session_tools.clear()
