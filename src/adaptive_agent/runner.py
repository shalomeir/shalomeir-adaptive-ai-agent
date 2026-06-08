from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
import io
from pathlib import Path
import json
import re
from typing import Any, Callable

from .context import ContextManager
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
    "Prefer runPython for one-off arithmetic, parsing, sorting, aggregation, and file transformation. "
    "Create a generated tool only when the user asks for a reusable tool or when reuse is clearly useful.\n"
    "After a runPython failure, retry with corrected runPython code unless the user asked for a reusable "
    "tool. Do not switch to update_tool for a built-in tool.\n"
    "When writing Python for JSON files, inspect whether the top-level value is a dict or list. If the "
    "top-level value is a dict with one relevant list field, operate on that list field. If the "
    "top-level dict contains a root node with children, treat it as an object tree and traverse "
    "children recursively instead of asking the user for internal field names.\n"
    "For follow-up requests that refer to previous results, use the conversation history. If the "
    "follow-up needs values not present in the summary, reopen the source file mentioned earlier "
    "and reconstruct the result set from the previous condition.\n"
    "When writing Python for CSV files, use csv.DictReader or csv.DictWriter and use column names from "
    "the header. For exact duplicate CSV rows, compare the complete row values. For grouped totals, "
    "group by the requested column only.\n"
    "When code is included in JSON, encode it as one valid JSON string with escaped newlines "
    '(\\n); never place raw multiline code outside the string value.\n'
    "For file output, prefer returning the computed text/data from runPython or a created tool, then "
    "call writeFile with the final relative path and content. This lets the runtime apply its file "
    "write policy. Do not write files directly inside runPython.\n"
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
    exporter: Exporter | None = None
    non_interactive: bool = False


@dataclass
class TurnResult:
    summary: str = ""
    observations: list[str] = field(default_factory=list)
    stopped_reason: str = "finish"


def _preview_text(text: str, limit: int = LLM_RESPONSE_LOG_CHARS) -> tuple[str, bool]:
    """Return a bounded preview for local diagnostics logs."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


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
            token_threshold=12000,
            summarize=lambda msgs: f"이전 {len(msgs)}개 메시지 요약",
        )
        self.generated = generated
        self.skills = skills
        self.policy = policy
        self._session_tools: list[str] = []
        self._tool_result_cache: dict[str, ToolResult] = {}
        self._suppressed_tool_names: set[str] = set()
        self._confirmed_direct_writes: set[str] = set()
        if self.skills is not None and self.generated is not None:
            for digest in self.skills.load_digests():
                spec = self.skills.load_spec(digest.name)
                self.deps.registry.register(self.generated.create(spec))

    def _direct_conversation_response(self, request: str) -> str | None:
        """Handle lightweight chat that should not enter the tool-planning loop."""
        text = request.strip()
        lowered = text.lower()
        compact = lowered.replace(" ", "")
        if not text:
            return "입력이 비어 있습니다. 짧게라도 말을 걸거나 실행할 작업을 적어 주세요."

        if self._is_runtime_model_question(lowered, compact):
            model = getattr(self.deps.llm, "model", "설정된 LLM")
            base_url = getattr(self.deps.llm, "base_url", "설정된 엔드포인트")
            return (
                f"현재 이 adaptive-agent CLI는 `{model}` 모델을 쓰고 있습니다. "
                f"엔드포인트는 `{base_url}`입니다."
            )

        if compact in {"안녕", "안녕.", "하이", "ㅎㅇ", "hello", "hi"}:
            model = getattr(self.deps.llm, "model", "설정된 LLM")
            return (
                "안녕. 지금은 데모용 adaptive-agent CLI 세션이고, "
                f"로컬 설정상 `{model}`로 응답하고 있어."
            )

        if compact in {"그냥대화", "그냥대화.", "대화", "대화.", "잡담", "잡담."}:
            return "좋아. 도구 실행 말고 그냥 얘기해도 돼. 방금처럼 데모가 딱딱하면 바로 말해줘."

        if compact in {
            "뭐야",
            "뭐야.",
            "뭐냐",
            "뭐냐.",
            "머야",
            "머야.",
            "아니너뭐냐",
            "아니너뭐냐.",
            "너뭐냐",
            "너뭐냐.",
            "너뭐야",
            "너뭐야.",
        }:
            model = getattr(self.deps.llm, "model", "설정된 LLM")
            return (
                "방금 답이 너무 일반적으로 나간 거야. "
                f"정확히는 `{model}` 기반의 adaptive-agent CLI 런타임이야."
            )

        return None

    def _is_runtime_model_question(self, lowered: str, compact: str) -> bool:
        direct_forms = {
            "너무슨모델?",
            "너무슨모델",
            "무슨모델?",
            "무슨모델",
            "모델명?",
            "모델명",
            "whatmodel?",
            "whichmodel?",
            "whatmodelareyou?",
            "whichmodelareyou?",
        }
        if compact in direct_forms:
            return True
        has_model_term = any(term in lowered for term in ("모델", "model", "llm"))
        asks_runtime = any(
            term in lowered
            for term in (
                "너",
                "현재",
                "무슨",
                "뭐",
                "어떤",
                "사용",
                "쓰",
                "which",
                "what",
                "using",
                "are you",
            )
        )
        return has_model_term and asks_runtime

    def _outside_workspace_write_summary(self, request: str) -> str | None:
        if not self._request_allows_file_write(request):
            return None
        candidates = re.findall(r"(?:\.\./|/|~)[^\s`'\";,)]*", request)
        if not candidates:
            return None
        if self.policy is not None:
            decision = self.policy.evaluate("out_of_workspace")
            self.tracer.log(
                kind="policy_decision",
                policy=decision.decision,
                policyReason=decision.reason,
                toolName="request_path",
            )
        return "정책상 거부됨: out_of_workspace. 작업 영역 밖 경로에는 파일을 쓸 수 없습니다."

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
        if isinstance(action, Respond) and not action.final:
            return f"respond:{action.text}"
        return None

    def _blocked_ask_user_observation(self, question: str, request: str = "") -> str | None:
        """Convert invalid ask_user package prompts into a runtime observation."""
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
                "진행하세요."
                + (f"\n{structure_hint}" if structure_hint else "")
            )
        return None

    def _hitl_required_summary(self, question: str) -> str:
        return f"HITL 처리가 필요합니다: {question}"

    def _mentioned_workspace_paths(self, text: str) -> list[str]:
        """Extract likely workspace-relative data paths from a user/model message."""
        candidates = re.findall(
            r"(?:[\w.-]+/)*[\w.-]+\.(?:json|csv|md|txt)", text, flags=re.I
        )
        paths: list[str] = []
        for candidate in candidates:
            path = candidate.strip("`'\".,:;()[]{}")
            if path.startswith("workspace/"):
                path = path.removeprefix("workspace/")
            if path and ".." not in Path(path).parts and path not in paths:
                paths.append(path)
        return paths

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
                    key: len(item)
                    for key, item in value.items()
                    if isinstance(item, list)
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
                    f"{tree_hint}"
                )
            if isinstance(value, list):
                sample_type = type(value[0]).__name__ if value else "empty"
                return f"{path}: 최상위 타입은 list, length={len(value)}, sampleType={sample_type}."
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

    def _request_allows_file_write(self, request: str) -> bool:
        lowered = request.lower()
        return any(
            term in lowered
            for term in (
                "저장",
                "save",
                "write",
                "create",
                "만들",
                "생성",
                "수정",
                "update",
                "제거",
                "remove",
                "delete",
                ".md",
                ".csv로",
                ".txt",
            )
        )

    def _is_side_effect_tool(self, name: str) -> bool:
        return name == "writeFile"

    def _generated_tool_reuse_block(self, name: str, request: str) -> str | None:
        if self._request_allows_file_write(request):
            return None
        tool = self.deps.registry.get(name)
        if tool is None or tool.origin != "generated":
            return None
        text = f"{tool.name} {tool.description}".lower()
        transformation_terms = (
            "write",
            "save",
            "sort",
            "clean",
            "dedup",
            "duplicate",
            "remove",
            "저장",
            "정렬",
            "정리",
            "중복",
            "제거",
        )
        if any(term in text for term in transformation_terms):
            self._suppressed_tool_names.add(tool.name)
            return (
                f"{name} 생성 도구는 파일 변환 작업용으로 보이며 현재 요청은 읽기 전용입니다. "
                "이번 턴 도구 목록에서 제외했습니다. runPython으로 필요한 값을 계산해 최종 답변하세요."
            )
        return None

    def _cached_tool_summary(self, name: str, result: ToolResult) -> str:
        if isinstance(result.output, dict):
            path = result.output.get("path")
            if path:
                return f"{name} 실행은 이미 성공했습니다. 결과 파일: {path}"
        return f"{name} 실행은 이미 성공했습니다. 마지막 결과: {result.output}"

    def _plan_raw(self) -> str:
        with self.tracer.span():
            digests = [
                digest
                for digest in self.deps.registry.digests()
                if digest.name not in self._suppressed_tool_names
            ]
            raw = self.deps.llm.chat(self.conv.messages(), digests)
            preview, truncated = _preview_text(raw)
            self.tracer.log(
                kind="llm_call",
                model=getattr(self.deps.llm, "model", None),
                responsePreview=preview,
                responseChars=len(raw),
                responseTruncated=truncated,
            )
            return raw

    def _gate(self, name: str, payload: dict[str, Any]) -> tuple[bool, str | None]:
        """Consult the policy before a side-effecting tool call.

        Returns (allowed, observation_if_blocked). File-writing tools are gated:
        an out-of-workspace target is denied outright; an in-workspace write asks
        the user. Other tools are allowed.
        """
        assert self.policy is not None
        if name == "writeFile":
            path = str(payload.get("path", ""))
        else:
            return True, None
        escapes = path.startswith("/") or path.startswith("~") or ".." in Path(path).parts
        action_id = "out_of_workspace" if escapes else "write_file"
        if action_id == "write_file" and path in self._confirmed_direct_writes:
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
            self._confirmed_direct_writes.add(path)
        return True, None

    def run_turn(self, request: str) -> TurnResult:
        direct_response = self._direct_conversation_response(request)
        if direct_response is not None:
            self.conv.add_user(request)
            self.conv.add_assistant(direct_response)
            return TurnResult(summary=direct_response)
        outside_write = self._outside_workspace_write_summary(request)
        if outside_write is not None:
            self.conv.add_user(request)
            self.conv.add_assistant(outside_write)
            return TurnResult(summary=outside_write)

        self.conv.add_user(request)
        result = TurnResult()
        self._suppressed_tool_names.clear()
        effective_request = request
        fix_failures = 0
        parse_failures = 0
        last_action_sig: str | None = None
        action_repeats = 0
        with self.tracer.trace():
            for _ in range(self.deps.max_iterations):
                raw = self._plan_raw()
                parsed = parse_action_text(raw)
                if not parsed.ok or parsed.action is None:
                    parse_failures += 1
                    error = parsed.error or "알 수 없는 파싱 오류"
                    self.tracer.log(kind="llm_call", parseOk=False)
                    self.conv.add_observation(error)
                    result.observations.append(error)
                    # A model that keeps emitting invalid JSON will not recover by
                    # looping to max_iterations — stop early instead of spinning.
                    if parse_failures > self.deps.max_fix_retries:
                        result.stopped_reason = "parse_failures"
                        break
                    continue
                parse_failures = 0
                action = parsed.action
                self.tracer.log(kind="llm_call", actionType=action.action, parseOk=True)
                sig = self._action_signature(action)
                if sig is not None and sig == last_action_sig:
                    action_repeats += 1
                else:
                    action_repeats = 0
                    last_action_sig = sig
                if (
                    isinstance(action, CallTool)
                    and sig is not None
                    and sig in self._tool_result_cache
                    and action_repeats > self.deps.max_fix_retries
                ):
                    cached = self._tool_result_cache[sig]
                    result.summary = self._cached_tool_summary(action.name, cached)
                    result.stopped_reason = "cached_result"
                    break
                if (
                    isinstance(action, CallTool)
                    and sig is not None
                    and sig in self._tool_result_cache
                    and action_repeats > 0
                ):
                    cached = self._tool_result_cache[sig]
                    obs = f"도구 {action.name} 캐시 결과: {cached.output}"
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
                if action_repeats > self.deps.max_fix_retries:
                    # Keep the last real observation as the answer: do not push the
                    # stop note into result.observations or finalize would surface
                    # it instead of the actual result the tools already produced.
                    self.conv.add_observation("같은 동작이 진전 없이 반복되어 작업을 중단합니다.")
                    result.stopped_reason = "no_progress"
                    break
                if (
                    isinstance(action, CallTool)
                    and action_repeats > 0
                    and self._is_side_effect_tool(action.name)
                ):
                    self.conv.add_observation(
                        "같은 부수효과 도구 호출이 반복되어 작업을 중단합니다."
                    )
                    result.stopped_reason = "no_progress"
                    break
                if isinstance(action, Finish):
                    result.summary = action.summary or ""
                    result.stopped_reason = "finish"
                    break
                if isinstance(action, Respond):
                    self.conv.add_assistant(action.text)
                    if action.final:
                        result.summary = action.text
                        result.stopped_reason = "finish"
                        break
                    self.conv.add_observation("계속 진행하세요.")
                    continue
                if isinstance(action, AskUser):
                    blocked_ask = self._blocked_ask_user_observation(action.question, effective_request)
                    if blocked_ask is not None:
                        self.conv.add_observation(blocked_ask)
                        result.observations.append(blocked_ask)
                        continue
                    if self.deps.non_interactive:
                        result.summary = self._hitl_required_summary(action.question)
                        result.stopped_reason = "hitl_required"
                        break
                    answer = self.deps.ask(action.question, action.choices)
                    if answer == NON_INTERACTIVE_ASK:
                        result.summary = self._hitl_required_summary(action.question)
                        result.stopped_reason = "hitl_required"
                        break
                    obs = f"사용자 답변: {answer}"
                    effective_request = f"{effective_request}\n{answer}"
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
                if isinstance(action, CallTool):
                    generated_block = self._generated_tool_reuse_block(action.name, effective_request)
                    if generated_block is not None:
                        self.conv.add_observation(generated_block)
                        result.observations.append(generated_block)
                        continue
                    if action.name == "writeFile" and not self._request_allows_file_write(effective_request):
                        obs = (
                            "현재 요청은 파일 저장을 요구하지 않습니다. writeFile을 실행하지 말고 "
                            "계산 결과를 최종 답변으로 반환하세요."
                        )
                        self.conv.add_observation(obs)
                        result.observations.append(obs)
                        continue
                    if self.policy is not None:
                        allowed, blocked = self._gate(action.name, action.input)
                        if not allowed:
                            assert blocked is not None
                            self.conv.add_observation(blocked)
                            result.observations.append(blocked)
                            continue
                    res = self.deps.registry.call(action.name, action.input)
                    self.tracer.log(kind="tool_call", toolName=action.name)
                    if res.ok:
                        fix_failures = 0
                        assert sig is not None
                        obs = f"도구 {action.name} 결과: {res.output}"
                        self._tool_result_cache[sig] = res
                    else:
                        fix_failures += 1
                        obs = f"도구 {action.name} 실패: {res.error}"
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    if not res.ok and fix_failures > self.deps.max_fix_retries:
                        note = "연속 실패가 한계를 넘었습니다. 작업을 중단합니다."
                        self.conv.add_observation(note)
                        result.observations.append(note)
                        result.stopped_reason = "consecutive_failures"
                        break
                    continue
                if isinstance(action, CreateTool):
                    self._tool_result_cache.clear()
                    obs = self._handle_create(action)
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
                if isinstance(action, UpdateTool):
                    self._tool_result_cache.clear()
                    obs = self._handle_update(action)
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
            else:
                result.stopped_reason = "max_iterations"
            if not result.summary and result.stopped_reason != "finish":
                self._finalize_incomplete(result)
            if result.stopped_reason == "finish":
                self._offer_persist()
            else:
                self._session_tools.clear()
            self.ctx.maybe_compact(self.conv)
        return result

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
