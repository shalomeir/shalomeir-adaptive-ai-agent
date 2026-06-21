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
from .python_repair import escape_newlines_in_string_literals
from .schemas import AskUser, CallTool, CreateTool, Finish, Message, Respond, ToolSpec, UpdateTool
from . import source_contracts
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
AMBIGUOUS_FILE_PATH_FIELD_NAMES = frozenset({"file", "filepath", "filename", "name"})
GENERATED_SOURCE_PATH_FIELD_NAMES = frozenset(
    {
        "file",
        "filepath",
        "filename",
        "input",
        "inputfile",
        "inputpath",
        "path",
        "source",
        "sourcefile",
        "sourcepath",
        "src",
    }
)
UNKNOWN_DYNAMIC_WRITE_PATH = "<dynamic path>"
CONTEXT_TOOL_NAMES = {"readFile", "listFiles", "searchDocs", "askUser"}
FILE_WRITE_INTENT_TERMS = (
    "as a file",
    "export",
    "output file",
    "overwrite",
    "persist",
    "save",
    "store",
    "write",
    "기록",
    "내보내",
    "작성",
    "저장",
    "쓰기",
    "씁",
    "출력 파일",
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
    "순으로",
)
CONTEXT_FOLLOWUP_TERMS = (
    "above",
    "again",
    "filtered",
    "last",
    "previous",
    "that",
    "those",
    "그 결과",
    "그걸",
    "그거",
    "다시",
    "방금",
    "앞서",
    "위",
    "이전",
)
TOOL_MATCH_STOPWORDS = {
    "and",
    "csv",
    "data",
    "file",
    "for",
    "from",
    "json",
    "the",
    "tool",
    "with",
}
TOOL_INTENT_ALIASES: dict[str, tuple[str, ...]] = {
    "aggregate": ("aggregate", "group", "집계"),
    "average": ("average", "avg", "mean", "평균"),
    "clean": ("clean", "정리"),
    "count": ("count", "개수", "몇 개", "몇개"),
    "dedupe": ("dedupe", "duplicate", "duplicates", "중복"),
    "delete": ("delete", "remove", "제거", "삭제", "제외"),
    "extract": ("extract", "list", "나열", "추출"),
    "filter": ("filter", "where", "필터", "이상", "이하", "초과", "미만"),
    "sort": ("sort", "order", "sorted", "정렬", "순으로", "오름차순", "내림차순"),
    "sum": ("sum", "total", "합계", "합산"),
    "table": ("table", "markdown", "테이블", "표로", "표를", "표에"),
    "transform": ("convert", "parse", "transform", "변환", "분석"),
    "write": ("output", "save", "write", "저장", "작성", "파일로"),
}
BROAD_TOOL_INTENTS = {"clean", "transform"}
OPTIONAL_TOOL_INTENTS_WITH_CONTENT_MATCH = {"average", "extract", "filter"}
TOOL_ACTION_TERMS = {
    term
    for aliases in TOOL_INTENT_ALIASES.values()
    for alias in aliases
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*|[가-힣]{2,}", alias.lower())
}
TOOL_RESULT_FINALIZATION_HINT = (
    "이 도구 결과는 현재 요청에 대한 최종 근거입니다. 같은 입력으로 같은 도구를 다시 호출하지 "
    "마세요. 요청한 답이 결과에 있으면 다음 action은 respond(final:true) 또는 finish입니다. "
    "최종 답변에는 위 결과에 있는 값만 사용하고, 결과에 없는 숫자나 필드를 추정하지 마세요. "
    "요청을 답하기에 결과가 부족할 때만 다른 도구 호출이나 update_tool을 사용하세요."
)

_SYSTEM = (
    "You are a task-solving agent that creates and runs small Python tools. You are not a "
    "general chatbot and must not claim to be a vendor model.\n"
    "Output exactly one JSON object on every turn: no prose, markdown, or code fences. The object "
    'must contain string field "action". Do not return bare data such as {"path":"out.csv"}; '
    'wrap final user-facing data as {"action":"respond","text":"...","final":true} or '
    '{"action":"finish","summary":"..."}.\n'
    "Allowed actions:\n"
    '- {"action":"respond","text":"...","final":true} — give an answer; final:true ends the task\n'
    '- {"action":"ask_user","question":"..."} — ask when the request is ambiguous\n'
    '- {"action":"call_tool","name":"<tool>","input":{...}} — run a built-in or created tool\n'
    '- {"action":"create_tool","spec":{"name":"kebab-name","description":"...",'
    '"code":"def run(input):\\n    return ...","inputSchema":{"type":"object"}}} — make a tool\n'
    '- {"action":"update_tool","name":"<tool>","code":"def run(input):\\n    return ..."} — fix a failed tool\n'
    '- {"action":"finish","summary":"..."} — stop when the task is done\n'
    'Tool names are used only through call_tool; never put a tool name in "action". Use the exact '
    "input fields shown in the tool inventory. The configured workspace is the default root for "
    "file work. Unless the user explicitly gives another allowed location, treat bare filenames "
    'and relative paths such as "data.json" as files inside that workspace. A referenced file may '
    "not exist; verify by reading/listing the workspace or by running the appropriate tool before "
    'finalizing. Never use absolute paths or "..".\n'
    "Use runPython for small, temporary snippets that are naturally a few lines at most, such as "
    "quick arithmetic, regex checks, simple list comprehensions, or one-off verification. For "
    "workspace file I/O, reusable file/data analysis, filtering, aggregation, transformation, or "
    "state changes, create or reuse a generated tool and then call it. A generated tool defines "
    "def run(input): and returns the result. Its code runs with the workspace as cwd, so it may "
    "open requested relative files directly.\n"
    "Use only the Python standard library. Do not ask to install packages. If a generated tool "
    "fails or validation says the result is insufficient, update_tool that tool, then call it again. "
    "Do not repeat the same failed call.\n"
    "Ask the user only for information that cannot be read from the workspace or inferred from the "
    "request. Do not ask to confirm already provided file paths, data structure, package installs, "
    "or whether to execute a tool.\n"
    "Write files only when the user explicitly asks to save/write/create/update output. Prefer "
    "returning computed content from a generated tool and using writeFile for the final requested "
    "path so runtime policy can apply.\n"
    "For generated file transformation tools, prefer reusable inputs such as sourcePath/src and "
    "outputPath/dst in inputSchema; do not hardcode the current source or output filename when the "
    "same operation may be reused for another file.\n"
    "Successful tool observations are authoritative. Use runtime observations as state. If an observation gives a successful tool "
    "result for the current request, synthesize the final answer from that result. If observations "
    "reject a tool/action, choose a different action instead of repeating it; do not call the same "
    "tool again with the same input.\n"
    "If the user asks for specific fields or aggregates, answer only those requested values instead "
    "of dumping full records.\n"
    "When the answer is ready, respond(final:true) or finish. Keep tool names kebab-case."
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
    validation_failed_tool_messages: dict[str, str] = field(default_factory=dict)
    blocked_action_counts: dict[str, int] = field(default_factory=dict)
    last_tool_call_index: int | None = None
    last_tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_workspace_paths: set[str] = field(default_factory=set)
    blocked_tool_actions: dict[str, str] = field(default_factory=dict)
    last_blocked_ask_observation: str | None = None
    blocked_ask_repeats: int = 0


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
        self.generated = generated
        self.conv = ConversationStore(system=self._system_prompt())
        self.ctx = ContextManager(
            token_threshold=deps.compaction_token_threshold,
            summarize=summarize_messages,
        )
        self.skills = skills
        self.policy = policy
        self._session_tools: list[str] = []
        self._tool_result_cache: dict[str, ToolResult] = {}
        self._confirmed_write_paths: set[str] = set()
        self._recent_workspace_paths: list[str] = []
        if self.skills is not None and self.generated is not None:
            for digest in self.skills.load_digests():
                spec = self.skills.load_spec(digest.name)
                self.deps.registry.register(self.generated.create(spec))

    def _workspace_root(self) -> str | None:
        if self.generated is None:
            return None
        return str(self.generated.sandbox.workspace.resolve())

    def _system_prompt(self) -> str:
        root = self._workspace_root()
        if root is None:
            return _SYSTEM
        return (
            f"{_SYSTEM}\n"
            f"Current configured workspace root: {root}\n"
            "When the user mentions a filename without a directory, assume it is workspace-relative "
            "unless the user clearly says otherwise."
        )

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

    def _blocked_ask_user_observation(
        self,
        question: str,
        request: str = "",
        missing_workspace_paths: set[str] | None = None,
    ) -> str | None:
        """Convert invalid ask_user package prompts into a runtime observation."""
        repeated_request = self._repeated_actionable_request_observation(question, request)
        if repeated_request is not None:
            return repeated_request
        redundant_confirmation = self._redundant_request_confirmation_observation(
            question, request
        )
        if redundant_confirmation is not None:
            return redundant_confirmation

        lowered = question.lower()
        known_paths = self._mentioned_workspace_paths(request)
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
        asks_to_run_tool = (
            any(term in lowered for term in ("execute", "run", "call", "실행", "호출"))
            and any(term in lowered or term in question for term in ("tool", "도구"))
        )
        if asks_to_run_tool and self._mentioned_workspace_paths(request):
            paths = self._mentioned_workspace_paths(request)
            return (
                "생성/수정한 도구를 실행할지 사용자에게 묻지 않습니다. 현재 요청을 완료하려면 "
                "방금 만든 generated tool을 call_tool로 실행하세요. 필요한 입력 파일 경로는 "
                f"요청에 나온 상대 경로를 그대로 사용하세요: {', '.join(paths)}."
            )
        question_paths = self._mentioned_workspace_paths(question)
        if len(question_paths) == 1 and question_paths[0] in known_paths:
            structure_hint = self._workspace_structure_hint(question)
            if structure_hint:
                return (
                    "질문에 나온 작업 영역 파일은 사용자에게 내용을 확인해 달라고 묻지 않습니다. "
                    "readFile 또는 runPython으로 직접 열어 필요한 값을 확인하고 계속 진행하세요."
                    f"\n{structure_hint}"
                )
        asks_about_output_creation = (
            question_paths
            and self._request_mentions_file_write(request)
            and any(path in known_paths for path in question_paths)
            and any(
                term in lowered or term in question
                for term in (
                    "create",
                    "exist",
                    "make",
                    "생성",
                    "만들",
                    "존재",
                    "파일이 없",
                    "파일이없",
                )
            )
            and not any(
                not self._workspace_path_exists(path) for path in self._request_source_paths(request)
            )
        )
        if asks_about_output_creation:
            return (
                "요청한 출력 파일은 아직 없어도 됩니다. 출력 파일 존재 여부나 생성 여부를 "
                "사용자에게 묻지 말고, 현재 요청이 저장을 요구하면 writeFile 또는 생성 도구로 "
                "작업 영역 내부 상대 경로에 새 파일을 만드세요."
            )
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
        known_missing_paths = missing_workspace_paths or set()
        has_missing_source_path = self._request_has_missing_source_path(request) or any(
            path in known_missing_paths for path in self._request_source_paths(request)
        )
        if (
            question_paths
            and self._request_contains_inline_structured_data(request)
            and any(
                path in known_missing_paths or not self._workspace_path_exists(path)
                for path in question_paths
            )
        ):
            missing = ", ".join(
                path
                for path in question_paths
                if path in known_missing_paths or not self._workspace_path_exists(path)
            )
            return (
                "현재 요청에는 workspace 파일 경로가 아니라 inline structured data가 포함되어 "
                f"있습니다. 존재하지 않는 파일({missing})을 사용자에게 요청하지 말고, 원래 "
                "사용자 요청 본문의 inline 데이터를 직접 처리하도록 generated tool을 만들거나 "
                "수정한 뒤 실행하세요."
            )
        if asks_for_known_paths and not has_missing_source_path:
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
            if question_paths and not all(path in known_paths for path in question_paths):
                return None
            structure_target = f"{request}\n{question}"
            structure_hint = self._workspace_structure_hint(structure_target)
            if not structure_hint:
                return None
            return (
                "작업 영역 파일의 구조는 사용자에게 묻지 않습니다. readFile 또는 runPython으로 "
                "파일을 직접 열어 최상위 타입과 필요한 필드를 확인한 뒤, 수정한 코드로 계속 "
                "진행하세요." + (f"\n{structure_hint}" if structure_hint else "")
            )
        return None

    def _redundant_request_confirmation_observation(
        self, question: str, request: str
    ) -> str | None:
        if not self._mentioned_workspace_paths(request):
            return None
        lowered = question.lower()
        confirmation_terms = (
            "맞는지",
            "확인",
            "confirm",
            "correct",
            "right",
            "이 요청",
            "this request",
        )
        if not any(term in lowered or term in question for term in confirmation_terms):
            return None
        request_paths = set(self._mentioned_workspace_paths(request))
        question_paths = set(self._mentioned_workspace_paths(question))
        if request_paths and not request_paths <= question_paths:
            return None
        request_intents = self._tool_intents(request)
        question_intents = self._tool_intents(question)
        if request_intents and not (request_intents & question_intents):
            return None
        request_terms = self._match_terms(request)
        question_terms = self._match_terms(question)
        if request_terms:
            overlap = len(request_terms & question_terms) / len(request_terms)
            if overlap < 0.5:
                return None
        return (
            "현재 요청은 이미 파일 경로와 작업 기준을 포함하므로 사용자에게 재확인하지 않습니다. "
            "요청을 다시 묻지 말고, 사용 가능한 도구가 현재 작업과 맞으면 호출하고, 맞지 않으면 "
            "현재 요청에 맞는 generated tool을 create_tool로 만든 뒤 call_tool로 실행하세요."
        )

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
        return source_contracts.mentioned_workspace_paths(text)

    def _remember_workspace_paths(self, paths: list[str]) -> None:
        for path in paths:
            if self._path_escapes_workspace(path):
                continue
            if path in self._recent_workspace_paths:
                self._recent_workspace_paths.remove(path)
            self._recent_workspace_paths.append(path)
        self._recent_workspace_paths = self._recent_workspace_paths[-5:]

    def _recent_source_paths(self, suffixes: set[str] | None = None) -> list[str]:
        paths = [path for path in self._recent_workspace_paths if not self._path_escapes_workspace(path)]
        if suffixes is not None:
            paths = [path for path in paths if Path(path).suffix.lower() in suffixes]
        return list(reversed(paths))

    def _contextual_source_observation(self, request: str) -> str | None:
        if not self._request_is_contextual_followup(request):
            return None
        current_source_paths = [
            path
            for path in self._mentioned_workspace_paths(request)
            if Path(path).suffix.lower() in {".json", ".csv"}
        ]
        if current_source_paths:
            return None
        paths = self._recent_source_paths({".json", ".csv"})
        if not paths:
            return None
        source_hint = self._source_content_hint(paths[:1])
        return (
            "이 요청은 이전 결과를 바탕으로 한 후속 작업으로 보입니다. 이전 작업의 source 파일은 "
            f"{', '.join(paths)}입니다. 이전 최종 답변이나 도구 결과에 필요한 필드가 없으면 "
            "값을 추정하거나 평균값을 재사용하지 말고 source 파일을 다시 읽어 필요한 필드를 "
            f"확인한 뒤 진행하세요.{source_hint}"
        )

    def _source_content_hint(self, paths: list[str], max_bytes: int = 4096) -> str:
        hints: list[str] = []
        for path in paths:
            content = self._read_workspace_text(path, max_bytes=max_bytes)
            if content is None:
                continue
            hints.append(f"\n{path} content preview:\n{content[:max_bytes]}")
        return "".join(hints)

    def _workspace_path_exists(self, path: str) -> bool:
        if self._path_escapes_workspace(path):
            return False
        if self.deps.registry.get("readFile") is None:
            if self.generated is not None:
                return (self.generated.sandbox.workspace / path).exists()
            return True
        res = self.deps.registry.call("readFile", {"path": path, "maxBytes": 1})
        return res.ok

    def _request_source_paths(self, request: str) -> list[str]:
        paths = self._mentioned_workspace_paths(request)
        if len(paths) >= 2 and self._request_mentions_file_write(request):
            return paths[:-1]
        if (
            len(paths) == 1
            and self._request_mentions_file_write(request)
            and self._path_is_likely_output_only(request, paths[0])
        ):
            return []
        return paths

    def _path_is_likely_output_only(self, request: str, path: str) -> bool:
        if not self._request_mentions_file_write(request):
            return False
        suffix = Path(path).suffix.lower()
        stem = Path(path).stem.lower()
        if suffix in {".md", ".txt"}:
            return True
        if self._request_is_contextual_followup(request):
            return True
        return any(term in stem for term in ("clean", "sorted", "output", "result", "table", "out"))

    def _request_has_missing_source_path(self, request: str) -> bool:
        return any(not self._workspace_path_exists(path) for path in self._request_source_paths(request))

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

    def _likely_output_path(self, request: str, suffix: str) -> str | None:
        paths = [path for path in self._mentioned_workspace_paths(request) if path.endswith(suffix)]
        if len(paths) >= 2:
            return paths[-1]
        if len(paths) == 1 and self._request_mentions_file_write(request):
            return paths[0]
        return None

    def _validate_completed_work(
        self,
        request: str,
        action: CallTool | None = None,
        result: ToolResult | None = None,
    ) -> tuple[bool, str] | None:
        if action is not None and action.name == "writeFile":
            path = str(action.input.get("path", ""))
            content = str(action.input.get("content", ""))
            return self._generic_file_content_validation(path, content)

        if result is not None and result.ok and self._request_mentions_file_write(request):
            output_path = self._generic_completed_output_path(result)
            if output_path is None:
                requested_output_path = self._likely_output_path(request, ".csv")
                if requested_output_path is None:
                    return None
                output_path = requested_output_path
            requested_output_path = self._likely_output_path(request, Path(output_path).suffix)
            if requested_output_path is not None and output_path != requested_output_path:
                return (
                    False,
                    f"{output_path} 검증 실패: 요청한 출력 경로는 {requested_output_path}입니다. "
                    "도구가 요청한 출력 파일을 만들거나 반환하도록 코드를 수정하세요.",
                )
            output_content = self._read_workspace_text(output_path)
            if output_content is None:
                return (
                    False,
                    f"{output_path} 검증 실패: 파일이 아직 없거나 읽을 수 없습니다.",
                )
            return self._generic_file_content_validation(output_path, output_content)

        if (
            action is None
            and result is None
            and self._request_mentions_file_write(request)
            and self.deps.registry.get("readFile") is not None
            and all(self._workspace_path_exists(path) for path in self._request_source_paths(request))
        ):
            output_path = self._likely_output_path(request, ".csv")
            if output_path is not None:
                output_content = self._read_workspace_text(output_path)
                if output_content is None:
                    return False, f"{output_path} 검증 실패: 파일이 아직 없거나 읽을 수 없습니다."
                return self._generic_file_content_validation(output_path, output_content)

        return None

    def _generic_completed_output_path(self, result: ToolResult) -> str | None:
        if isinstance(result.output, dict):
            for key in ("path", "output", "outputPath", "output_file", "outputFile"):
                value = result.output.get(key)
                if isinstance(value, str) and self._looks_like_file_path(value):
                    return value
        return None

    def _request_mentions_dedupe(self, request: str) -> bool:
        intents = self._tool_intents(request)
        return "dedupe" in intents or "duplicate" in request.lower() or "중복" in request

    def _generic_file_content_validation(self, path: str, content: str) -> tuple[bool, str] | None:
        if not path:
            return None
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return False, f"{path} 검증 실패: JSON으로 파싱할 수 없습니다: {exc.msg}."
            return True, f"{path} 저장 검증 완료: JSON 파일을 읽을 수 있습니다."
        if suffix == ".csv":
            try:
                rows = list(csv.reader(io.StringIO(content)))
            except csv.Error as exc:
                return False, f"{path} 검증 실패: CSV로 파싱할 수 없습니다: {exc}."
            if not rows:
                return False, f"{path} 검증 실패: CSV 내용이 비어 있습니다."
            return True, f"{path} 저장 검증 완료: CSV header={rows[0]}, rows={len(rows) - 1}."
        if suffix in {".md", ".txt"}:
            if not content.strip():
                return False, f"{path} 검증 실패: 파일 내용이 비어 있습니다."
            return True, f"{path} 저장 검증 완료: 텍스트 파일을 읽을 수 있습니다."
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
        file_match = re.search(r"No such file or directory: ['\"]([^'\"]+)['\"]", error)
        if is_generated_tool and file_match is not None:
            missing_path = file_match.group(1)
            paths = self._mentioned_workspace_paths(request)
            path_hint = f" 요청 파일: {', '.join(paths)}." if paths else ""
            return (
                f"생성 도구가 workspace에서 파일을 열지 못했습니다: {missing_path}. "
                "같은 call_tool을 반복하지 마세요. 요청 파일이 실제로 없으면 ask_user로 확인하고, "
                "코드가 잘못된 경로를 하드코딩했거나 call_tool input의 path를 무시했다면 "
                "update_tool로 `input.get('path', '파일명')` 또는 올바른 상대 경로를 열도록 "
                f"수정한 뒤 다시 실행하세요.{path_hint}"
            )
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
        if is_generated_tool and "TypeError: unhashable type: 'dict'" in error:
            if self._request_mentions_dedupe(request) or Path(
                (self._mentioned_workspace_paths(request) or [""])[0]
            ).suffix.lower() == ".csv":
                return (
                    "CSV DictReader의 row는 dict라서 set, dict key, OrderedDict.fromkeys에 "
                    "그대로 넣을 수 없습니다. 같은 실패 코드를 반복하지 말고 update_tool로 "
                    "`key = tuple(row.items())` 같은 hashable key를 만들고, seen set에는 key만 "
                    "넣으세요. 출력할 때는 원래 row dict를 보존해서 writer.writerows(rows)에 "
                    "넘기고, 정렬 요청이 있으면 dedupe가 끝난 rows를 date 등 요청 컬럼으로 "
                    "sorted(...) 하세요."
                )
        if tool_name == "runPython" and "runPython 안에서 파일을 직접 쓰려고 했습니다" in error:
            if not self._request_mentions_file_write(request):
                return (
                    "현재 요청은 저장/수정 요청이 아닌 읽기 전용 계산입니다. output_path 인자, "
                    "open(..., 'w'), writerow/writerows, writeFile 호출, cleaned CSV 생성 코드를 "
                    "모두 제거하고 입력 파일을 읽어서 최종 계산 결과만 stdout으로 출력하세요."
                )
            return (
                "runPython에서는 파일을 직접 쓰지 않습니다. 변환 결과 content를 stdout으로 출력한 뒤 "
                "별도 writeFile 호출로 저장하세요."
            )
        return None

    def _missing_path_from_tool_error(self, error: str | None) -> str | None:
        if not error:
            return None
        file_match = re.search(r"No such file or directory: ['\"]([^'\"]+)['\"]", error)
        if file_match is None:
            return None
        missing_path = file_match.group(1)
        display_path = Path(missing_path).name if Path(missing_path).is_absolute() else missing_path
        if self._path_escapes_workspace(display_path):
            return None
        return display_path

    def _request_mentions_file_write(self, request: str) -> bool:
        if self._request_forbids_file_write(request):
            return False
        lowered = request.lower()
        compact = lowered.replace(" ", "")
        mentioned_paths = self._mentioned_workspace_paths(request)
        file_context = bool(mentioned_paths) or "file" in lowered or "파일" in request
        contextual_write_terms = (
            "create",
            "make",
            "generate",
            "update",
            "생성",
            "만들",
            "수정",
            "업데이트",
        )
        return (
            any(term in lowered for term in FILE_WRITE_INTENT_TERMS)
            or re.search(r"파일에(?!서)", request) is not None
            or (file_context and any(term in lowered or term in request for term in contextual_write_terms))
            or any(
                term in compact
                for term in (
                    "asafile",
                    "outputfile",
                    "파일로",
                    "출력파일",
                )
            )
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
        has_data_task = bool(self._tool_intents(request)) or any(
            term in lowered or term in compact for term in DATA_TOOL_TASK_TERMS
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
            "runPython은 짧은 임시 Python snippet용입니다. 간단한 산술, regex 확인, 작은 "
            "list/filter 검증처럼 몇 줄 안에 끝나는 작업에는 사용할 수 있지만, workspace 파일을 "
            "읽어 처리하거나 재사용 가능한 필터링, 변환, 집계, 평균 계산을 해야 하면 generated "
            "tool 경로를 사용하세요. 요청에 맞는 generated tool이 이미 있으면 call_tool로 "
            "재사용하고, 없으면 create_tool로 관리형 도구를 만든 뒤 그 도구를 call_tool로 "
            f"실행하세요.{existing_hint}"
        )

    def _blocks_managed_tool_default(self, state: _TurnLoopState) -> bool:
        if state.completed_output_paths:
            return False
        return self._request_prefers_managed_tool(
            state.effective_request
        ) and not self._has_called_generated_tool(state)

    def _payload_mentions_any_path(self, payload: Any, paths: list[str]) -> bool:
        return source_contracts.payload_mentions_any_path(payload, paths)

    def _paths_from_payload(self, payload: Any) -> list[str]:
        return source_contracts.paths_from_payload(payload)

    def _payload_contains_inline_dataset(self, payload: Any) -> bool:
        return source_contracts.payload_contains_inline_dataset(payload)

    def _request_contains_inline_structured_data(self, request: str) -> bool:
        return source_contracts.request_contains_inline_structured_data(request)

    def _generated_tool_inline_data_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        request_has_inline_data = self._request_contains_inline_structured_data(
            state.effective_request
        )
        if not self._request_prefers_managed_tool(
            state.effective_request
        ) and not request_has_inline_data:
            return None
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return None
        paths = self._mentioned_workspace_paths(state.effective_request)
        if paths and self._payload_mentions_any_path(action.input, paths):
            return None
        if paths and self._payload_contains_inline_dataset(action.input):
            if (
                self._request_is_contextual_followup(state.effective_request)
                and not self._request_source_paths(state.effective_request)
            ):
                return None
            source_paths = self._request_source_paths(state.effective_request)
            source = source_paths[0] if source_paths else paths[0]
            return (
                "요청은 workspace 파일 기준 처리입니다. call_tool input에 임의 샘플 데이터나 추정 데이터를 "
                "넣지 마세요. 실제 파일 경로를 넘기거나 생성 도구 코드에서 "
                f"`open('{source}')`로 workspace 파일을 직접 읽은 뒤 다시 실행하세요."
            )

        if paths:
            return None

        if not request_has_inline_data:
            return None

        payload_paths = self._paths_from_payload(action.input)
        spec = self._generated_tool_spec(action.name)
        code_paths = self._mentioned_workspace_paths(spec.code) if spec is not None else []
        missing_paths = [
            path
            for path in [*payload_paths, *code_paths]
            if not self._workspace_path_exists(path)
        ]
        if not missing_paths:
            return None
        missing = ", ".join(dict.fromkeys(missing_paths))
        return (
            "현재 요청에는 workspace 파일 경로가 아니라 inline structured data가 포함되어 "
            f"있습니다. 존재하지 않는 workspace 파일({missing})을 입력으로 만들지 마세요. "
            "원래 사용자 요청 본문의 inline 데이터를 처리하도록 generated tool을 만들거나 "
            "수정한 뒤 실행하세요."
        )

    def _generated_tool_file_read_observation(
        self, state: _TurnLoopState, name: str, code: str
    ) -> str | None:
        if not self._request_prefers_managed_tool(state.effective_request):
            return None
        if (
            not self._mentioned_workspace_paths(state.effective_request)
            and self._request_contains_inline_structured_data(state.effective_request)
        ):
            return None
        if self._request_is_contextual_followup(state.effective_request):
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
            return None
        paths = self._request_source_paths(state.effective_request)
        if not paths and self._request_is_contextual_followup(state.effective_request):
            paths = self._recent_source_paths({".json", ".csv"})
        source = paths[0] if paths else "요청 파일"
        return (
            f"생성 도구 {name}은(는) workspace 파일 처리용인데 코드가 실제 파일을 읽지 않습니다. "
            "call_tool input으로 파일 내용을 주입받는 도구를 만들지 말고, 도구 코드 안에서 "
            f"`open('{source}')`, `json.load(open(path))`, `csv.reader(open(path))`처럼 "
            "workspace 파일을 직접 읽도록 create_tool/update_tool을 다시 작성하세요."
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

    def _generated_tool_entrypoint_observation(
        self, name: str, code: str, entrypoint: str = "run"
    ) -> str | None:
        try:
            tree = ast.parse(escape_newlines_in_string_literals(code))
        except SyntaxError as exc:
            return (
                f"생성 도구 {name}의 Python 코드 문법이 올바르지 않습니다: {exc.msg}. "
                f"코드는 반드시 `def {entrypoint}(input):` 함수 안에 작성하고 결과를 return하세요."
            )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
                if not node.args.args:
                    return (
                        f"생성 도구 {name}의 `{entrypoint}` 함수는 input 인자를 받아야 합니다. "
                        f"`def {entrypoint}(input):` 형태로 다시 작성하세요."
                    )
                return None
        return (
            f"생성 도구 {name} 코드에는 반드시 `def {entrypoint}(input):` entrypoint가 있어야 합니다. "
            "module top-level에서 파일을 읽거나 작업을 실행하지 말고, 모든 작업을 이 함수 안에 넣고 "
            "결과를 return하세요."
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
        failure_details = [
            state.validation_failed_tool_messages[tool_name]
            for tool_name in failed_names
            if tool_name in state.validation_failed_tool_messages
        ]
        detail_hint = f" 마지막 실패: {' '.join(failure_details)}" if failure_details else ""
        return (
            f"생성 도구 {', '.join(failed_names)}의 실행 또는 검증이 실패했습니다. "
            "같은 도구를 그대로 다시 call_tool 하거나 finish 하지 말고 update_tool로 코드를 수정하세요. "
            f"실패 메시지의 조건을 코드에 직접 반영한 뒤 다시 실행해 확인하세요."
            f"{detail_hint}{path_hint}"
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
        if normalized_key in AMBIGUOUS_FILE_PATH_FIELD_NAMES:
            return description_suggests_write and self._looks_like_file_path(value)
        return normalized_key == "path" and description_suggests_write

    def _description_suggests_file_write(self, description: str) -> bool:
        lowered = description.lower()
        return any(term in lowered for term in FILE_WRITE_INTENT_TERMS)

    def _generated_tool_read_only_write_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return None
        if self._request_mentions_file_write(state.effective_request):
            return None
        if not self._description_suggests_file_write(tool.description):
            return None
        has_content_payload = any(
            self._normalized_field_name(str(key)) in {"content", "body", "text"}
            for key in action.input
        )
        has_ambiguous_file_target = any(
            self._normalized_field_name(str(key)) in AMBIGUOUS_FILE_PATH_FIELD_NAMES | {"path"}
            and isinstance(value, str)
            and self._looks_like_file_path(value)
            for key, value in action.input.items()
        )
        if not has_content_payload and not has_ambiguous_file_target:
            return None
        return (
            f"현재 요청은 읽기 전용 계산입니다. generated tool {action.name}은(는) 파일 쓰기 "
            "성격이 있어 실행하지 않습니다. 입력 파일을 저장/덮어쓰지 말고, 요청에 맞는 읽기 전용 "
            "generated tool을 create_tool로 만든 뒤 call_tool로 실행하세요."
        )

    def _generated_tool_spec(self, name: str) -> ToolSpec | None:
        if self.generated is not None:
            spec = self.generated.specs().get(name)
            if spec is not None:
                return spec
        if self.skills is not None:
            try:
                return self.skills.load_spec(name)
            except (FileNotFoundError, ValueError):
                return None
        return None

    def _input_field_names_from_generated_tool(self, name: str) -> set[str]:
        fields: set[str] = set()
        tool = self.deps.registry.get(name)
        if tool is not None:
            properties = tool.input_schema.get("properties")
            if isinstance(properties, dict):
                fields.update(str(key) for key in properties)
        spec = self._generated_tool_spec(name)
        if spec is not None:
            properties = spec.input_schema.get("properties")
            if isinstance(properties, dict):
                fields.update(str(key) for key in properties)
            fields.update(
                match.group(1)
                for match in re.finditer(r"input\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", spec.code)
            )
            fields.update(
                match.group(1)
                for match in re.finditer(r"input\.get\(\s*['\"]([^'\"]+)['\"]", spec.code)
            )
        return fields

    def _generated_source_path_field(self, name: str) -> str | None:
        fields = self._input_field_names_from_generated_tool(name)
        for field_name in fields:
            if self._normalized_field_name(field_name) in GENERATED_SOURCE_PATH_FIELD_NAMES:
                return field_name
        return None

    def _generated_output_path_field(self, name: str) -> str | None:
        fields = self._input_field_names_from_generated_tool(name)
        tool = self.deps.registry.get(name)
        description = tool.description if tool is not None else ""
        description_suggests_write = self._description_suggests_file_write(description)
        for field_name in fields:
            normalized = self._normalized_field_name(field_name)
            if self._is_generated_write_path_field(
                normalized, "result.csv", description_suggests_write
            ):
                return field_name
        return None

    def _repair_generated_tool_path_input(
        self, state: _TurnLoopState, action: CallTool
    ) -> bool:
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return False
        paths = self._mentioned_workspace_paths(state.effective_request)
        if not paths or self._payload_mentions_any_path(action.input, paths):
            return False
        changed = False
        source_field = self._generated_source_path_field(action.name)
        if source_field is not None and source_field not in action.input:
            action.input[source_field] = paths[0]
            changed = True
        if len(paths) >= 2 and self._request_mentions_file_write(state.effective_request):
            output_field = self._generated_output_path_field(action.name)
            if output_field is not None and output_field not in action.input:
                action.input[output_field] = paths[-1]
                changed = True
        return changed

    def _generated_tool_missing_requested_source_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return None
        paths = self._mentioned_workspace_paths(state.effective_request)
        if not paths:
            return None
        if self._payload_mentions_any_path(action.input, paths):
            return None
        spec = self._generated_tool_spec(action.name)
        if spec is None:
            return None
        if any(path in spec.code for path in paths):
            return None
        source_field = self._generated_source_path_field(action.name)
        if source_field is not None:
            suggested_input = dict(action.input)
            suggested_input[source_field] = paths[0]
            return (
                f"현재 요청의 workspace 파일({', '.join(paths)})이 generated tool {action.name}의 "
                "입력에 들어가지 않았습니다. 같은 빈 input을 반복하지 말고 다음처럼 요청 파일을 "
                f"넘겨 실행하세요: "
                f'{{"action":"call_tool","name":"{action.name}","input":'
                f"{json.dumps(suggested_input, ensure_ascii=False, sort_keys=True)}}}"
            )
        return (
            f"현재 요청의 workspace 파일({', '.join(paths)})이 generated tool {action.name}의 "
            "입력이나 코드에 연결되어 있지 않습니다. 이전 작업용 도구를 억지로 재사용하지 말고, "
            "요청 파일을 직접 읽는 generated tool을 만들거나 수정한 뒤 실행하세요."
        )

    def _generated_tool_output_as_source_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated":
            return None
        if not self._request_mentions_file_write(state.effective_request):
            return None
        mentioned_paths = self._mentioned_workspace_paths(state.effective_request)
        if not mentioned_paths:
            return None
        source_paths = self._request_source_paths(state.effective_request)
        output_paths = [path for path in mentioned_paths if path not in source_paths]
        if not output_paths:
            return None
        if not any(self._payload_mentions_any_path(action.input, [path]) for path in output_paths):
            return None
        if source_paths and self._payload_mentions_any_path(action.input, source_paths):
            return None
        recent_sources = self._recent_source_paths({".json", ".csv"})
        source_hint = ""
        if recent_sources:
            read_action = {
                "action": "call_tool",
                "name": "readFile",
                "input": {"path": recent_sources[0], "maxBytes": 4096},
            }
            source_hint = (
                f" 이전 source 파일은 {', '.join(recent_sources)}입니다. 다음 액션으로 source를 "
                f"읽어 필요한 필드를 확인하세요: "
                f"{json.dumps(read_action, ensure_ascii=False, sort_keys=True)}"
                f"{self._source_content_hint(recent_sources[:1])}"
            )
        return (
            f"{', '.join(output_paths)}은(는) 현재 요청의 출력 대상입니다. 이 파일을 입력 source로 "
            "읽는 generated tool을 호출하지 마세요. 필요한 값은 source 파일이나 이전 도구 결과에서 "
            f"얻고, 최종 산출물은 writeFile 또는 저장용 generated tool로 만드세요.{source_hint}"
        )

    def _current_turn_tool_mismatch_observation(
        self, state: _TurnLoopState, action: CallTool
    ) -> str | None:
        if action.name in {"readFile", "listFiles", "searchDocs", "askUser"}:
            return None
        tool = self.deps.registry.get(action.name)
        if (
            tool is not None
            and tool.origin != "generated"
            and action.name not in {"runPython", "writeFile"}
        ):
            return None
        if action.name in (state.turn_state.changed_tool_names or []):
            return None
        request = state.effective_request
        if self._request_allows_tool_work(request):
            if tool is None or tool.origin != "generated":
                return None
            paths = self._mentioned_workspace_paths(request)
            tool_has_detectable_intent = bool(self._tool_intents(f"{tool.name} {tool.description}"))
            if self._request_is_contextual_followup(request):
                return None
            if self._tool_covers_request_intent(tool.name, tool.description, request):
                return None
            if (
                paths
                and self._payload_mentions_any_path(action.input, paths)
                and (
                    not tool_has_detectable_intent
                    or self._tool_intent_ratio_satisfied(tool.name, tool.description, request)
                )
            ):
                return None
            sig = self._action_signature(action)
            repeated_block = (
                sig is not None and state.blocked_action_counts.get(sig, 0) > 0
            )
            prefix = (
                f"generated tool {action.name}은(는) 이 턴에서 이미 현재 요청과 맞지 않아 "
                "차단된 동일한 호출입니다. "
                if repeated_block
                else f"generated tool {action.name}은(는) 현재 사용자 요청과 충분히 맞지 않습니다. "
            )
            return (
                prefix
                + "이전 턴의 도구 호출을 계속하지 말고, 현재 요청에 맞는 도구만 사용하세요. "
                "등록된 도구 중 맞는 것이 없으면 현재 요청의 입력·처리·출력 contract를 만족하는 "
                "generated tool을 create_tool로 만든 뒤 call_tool로 실행하세요."
            )
        return (
            f"현재 사용자 요청은 새 도구 실행을 요구하지 않습니다. {action.name}을(를) 호출하지 말고 "
            "현재 요청에 직접 답하세요. 이전 턴의 도구 호출을 이어서 실행하지 마세요."
        )

    def _request_allows_tool_work(self, request: str) -> bool:
        lowered = request.lower()
        compact = re.sub(r"\s+", "", lowered)
        return bool(
            self._mentioned_workspace_paths(request)
            or self._request_mentions_file_write(request)
            or self._request_requires_new_tool(request)
            or self._request_is_contextual_followup(request)
            or self._tool_intents(request)
            or any(term in lowered or term in compact for term in DATA_TOOL_TASK_TERMS)
            or "tool" in lowered
            or "도구" in request
        )

    def _request_is_contextual_followup(self, request: str) -> bool:
        lowered = request.lower()
        return any(term in lowered or term in request for term in CONTEXT_FOLLOWUP_TERMS)

    def _tool_matches_request(self, name: str, description: str, request: str) -> bool:
        request_terms = self._match_terms(request)
        tool_terms = self._match_terms(f"{name} {description}")
        return bool(request_terms & tool_terms)

    def _tool_covers_request_intent(self, name: str, description: str, request: str) -> bool:
        request_intents = self._tool_intents(request)
        tool_intents = self._tool_intents(f"{name} {description}")
        if not request_intents or not tool_intents:
            return self._tool_matches_request(name, description, request)
        specific_request_intents = request_intents - BROAD_TOOL_INTENTS
        required_intents = specific_request_intents or request_intents
        overlap = required_intents & tool_intents
        if not overlap:
            return False
        request_content = self._content_terms(request)
        tool_content = self._content_terms(f"{name} {description}")
        content_overlap = request_content & tool_content
        if request_content and tool_content and not content_overlap:
            return False
        if (len(overlap) / len(required_intents)) >= 0.6:
            return True
        missing_intents = required_intents - overlap
        strong_content_overlap = self._strong_content_overlap(request, content_overlap)
        return (
            bool(strong_content_overlap)
            and missing_intents <= OPTIONAL_TOOL_INTENTS_WITH_CONTENT_MATCH
        )

    def _tool_intent_ratio_satisfied(self, name: str, description: str, request: str) -> bool:
        request_intents = self._tool_intents(request)
        tool_intents = self._tool_intents(f"{name} {description}")
        if not request_intents or not tool_intents:
            return False
        specific_request_intents = request_intents - BROAD_TOOL_INTENTS
        required_intents = specific_request_intents or request_intents
        overlap = required_intents & tool_intents
        return bool(overlap) and (len(overlap) / len(required_intents)) >= 0.6

    def _tool_intents(self, text: str) -> set[str]:
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        intents: set[str] = set()
        for intent, aliases in TOOL_INTENT_ALIASES.items():
            if any(alias in lowered or alias in compact for alias in aliases):
                intents.add(intent)
        return intents

    def _match_terms(self, text: str) -> set[str]:
        terms = {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+|[가-힣]{2,}", text.lower())
            if len(term) >= 3 and term not in TOOL_MATCH_STOPWORDS
        }
        return terms

    def _content_terms(self, text: str) -> set[str]:
        terms = {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*|[가-힣]{2,}", text.lower())
            if (
                (len(term) >= 2 if term.isascii() else len(term) >= 3)
                and term not in TOOL_MATCH_STOPWORDS
                and not self._is_actionish_term(term)
            )
        }
        return terms

    def _strong_content_overlap(self, request: str, overlap: set[str]) -> set[str]:
        path_stems = {Path(path).stem.lower() for path in self._mentioned_workspace_paths(request)}
        weak_terms = path_stems | {"workspace"}
        return {term for term in overlap if term not in weak_terms}

    def _is_actionish_term(self, term: str) -> bool:
        for action_term in TOOL_ACTION_TERMS:
            if term == action_term:
                return True
            if term.isascii() and action_term.isascii():
                if len(action_term) >= 4 and term.startswith(action_term):
                    return True
            elif action_term in term or term in action_term:
                return True
        return False

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
                return f"도구 {name} 결과 파일: {path}"
        return f"도구 {name} 결과: {result.output}"

    def _validated_tool_summary(
        self, name: str, result: ToolResult, validation_message: str
    ) -> str:
        summary = self._cached_tool_summary(name, result)
        if validation_message and validation_message not in summary:
            return f"{summary}\n{validation_message}"
        return summary

    def _cached_tool_observation(self, name: str, result: ToolResult) -> str:
        return f"도구 {name} 캐시 결과: {result.output}"

    def _tool_success_observation(self, name: str, result: ToolResult) -> str:
        observation = f"도구 {name} 결과: {result.output}"
        if name not in CONTEXT_TOOL_NAMES:
            observation = f"{observation}\n{TOOL_RESULT_FINALIZATION_HINT}"
        return observation

    def _read_only_path_only_result_observation(
        self, request: str, action: CallTool, result: ToolResult
    ) -> str | None:
        tool = self.deps.registry.get(action.name)
        if tool is None or tool.origin != "generated" or not result.ok:
            return None
        if self._request_mentions_file_write(request):
            return None
        if not isinstance(result.output, dict) or not result.output:
            return None
        path_fields = {"dest", "destination", "file", "filename", "output", "outputpath", "path"}
        if not all(
            self._normalized_field_name(str(key)) in path_fields
            and isinstance(value, str)
            and self._looks_like_file_path(value)
            for key, value in result.output.items()
        ):
            return None
        return (
            f"도구 {action.name} 결과는 파일 경로만 반환했습니다: {result.output}. "
            "현재 요청은 파일 저장이 아니라 계산 결과 답변이 필요하므로, 이 결과만으로 종료하지 마세요. "
            "요청한 값을 반환하는 generated tool을 만들거나 수정해 실행하세요."
        )

    def _tool_result_satisfies_request(
        self, request: str, action: CallTool, result: ToolResult
    ) -> bool:
        return self._read_only_path_only_result_observation(request, action, result) is None

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
        if not self._request_mentions_file_write(request):
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

    def _plan_raw(self, state: _TurnLoopState) -> str:
        with self.tracer.span():
            digests = self.deps.registry.digests()
            self.tracer.log(kind="llm_call_start", model=getattr(self.deps.llm, "model", None))
            try:
                raw = self._chat_with_deadline(state, digests)
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

    def _planning_messages(self, state: _TurnLoopState) -> list[Message]:
        messages = self.conv.messages()
        return [*messages, self._runtime_state_message(state)]

    def _runtime_state_message(self, state: _TurnLoopState) -> Message:
        recent_observations = state.result.observations[-5:]
        payload: dict[str, Any] = {
            "currentRequest": state.effective_request,
            "workspaceRoot": self._workspace_root(),
            "workspacePathAssumption": (
                "Bare filenames and relative paths in user requests are workspace-relative "
                "unless the user clearly specifies another allowed location."
            ),
            "turnActionIndex": state.turn_state.action_index,
            "requiresExecution": state.turn_state.requires_execution,
            "changedGeneratedTools": state.turn_state.changed_tool_names or [],
            "calledGeneratedTools": sorted(state.called_generated_tools),
            "calledChangedTools": sorted(state.called_changed_tools),
            "lastToolInputs": state.last_tool_inputs,
            "validationFailures": state.validation_failed_tool_messages,
            "missingWorkspacePaths": sorted(state.missing_workspace_paths),
            "blockedToolActions": state.blocked_tool_actions,
            "blockedActionCounts": state.blocked_action_counts,
            "recentObservations": recent_observations,
        }
        if state.turn_state.last_generated_tool_change is not None:
            payload["lastGeneratedToolChangeIndex"] = state.turn_state.last_generated_tool_change
        if state.turn_state.last_successful_work_call is not None:
            payload["lastSuccessfulWorkCallIndex"] = state.turn_state.last_successful_work_call
        preview, _, _ = _preview_value(payload, limit=3000)
        return Message(
            role="tool",
            content=(
                "[runtime-state]\n"
                "Use this structured state to choose the next JSON action for the current turn. "
                "Do not continue stale actions from previous turns. Tool actions listed in "
                "blockedToolActions or blockedActionCounts have already been rejected for this "
                "turn; choose a different tool, update/create a generated tool, or ask only for "
                "missing information that cannot be read from the workspace.\n"
                f"{preview}"
            ),
        )

    def _chat_with_deadline(self, state: _TurnLoopState, digests: list[Any]) -> str:
        timeout = getattr(self.deps.llm, "timeout", None)
        if timeout is None:
            return self.deps.llm.chat(self._planning_messages(state), digests)
        timeout_sec = float(timeout)
        result_queue: queue.Queue[tuple[str, str | Exception]] = queue.Queue(maxsize=1)
        messages = self._planning_messages(state)

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
                    raw = self._plan_raw(state)
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
        self._tool_result_cache.clear()
        self.conv.add_user(request)
        self._remember_workspace_paths(self._mentioned_workspace_paths(request))
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
        if contextual_source := self._contextual_source_observation(request):
            self._append_observation(state, contextual_source)
        return state

    def _append_observation(self, state: _TurnLoopState, observation: str) -> None:
        self.conv.add_observation(observation)
        state.result.observations.append(observation)

    def _record_blocked_tool_action(
        self, state: _TurnLoopState, action: CallTool, reason: str
    ) -> None:
        state.blocked_tool_actions[action.name] = reason
        sig = self._action_signature(action)
        if sig is not None:
            state.blocked_action_counts[sig] = state.blocked_action_counts.get(sig, 0) + 1

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
        ):
            cached = self._tool_result_cache[sig]
            if action.name in CONTEXT_TOOL_NAMES:
                if state.action_repeats > self.deps.max_fix_retries:
                    state.result.stopped_reason = "no_progress"
                    return sig, "break"
                self._append_observation(
                    state,
                    f"{action.name}은(는) 이미 같은 입력으로 실행했습니다. 캐시된 context 결과를 "
                    "다시 최종 답변으로 내보내지 말고, 그 결과를 근거로 현재 요청을 수행할 "
                    "작업 도구를 만들거나 호출하세요.",
                )
                return sig, "continue"
            if not self._tool_result_satisfies_request(state.effective_request, action, cached):
                state.result.stopped_reason = "no_progress"
                return sig, "break"
            if self._cached_tool_call_should_continue(state, action):
                self._append_observation(state, self._cached_tool_observation(action.name, cached))
                if warning := self._read_only_path_only_result_observation(
                    state.effective_request, action, cached
                ):
                    self._append_observation(state, warning)
                return sig, "continue"
            state.result.summary = self._cached_tool_summary(action.name, cached)
            state.result.stopped_reason = "cached_result"
            return sig, "break"

        return self._apply_non_cached_loop_guards(state, action, sig)

    def _cached_tool_call_should_continue(self, state: _TurnLoopState, action: CallTool) -> bool:
        """Allow one cached runPython observation when a following write may use it."""
        return (
            action.name == "runPython"
            and state.action_repeats == 1
            and self._request_mentions_file_write(state.effective_request)
        )

    def _apply_non_cached_loop_guards(
        self, state: _TurnLoopState, action: Any, sig: str | None
    ) -> tuple[str | None, _StepAction]:
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

    def _ungrounded_current_turn_result_observation(
        self, state: _TurnLoopState, text: str
    ) -> str | None:
        if state.turn_state.last_successful_work_call is not None:
            return None
        if not self._request_allows_tool_work(state.effective_request):
            return None
        if self._request_is_contextual_followup(state.effective_request):
            return None
        if self._looks_like_clarification(text):
            return None
        if not self._looks_like_work_result(text):
            return None
        if not self._has_previous_result_like_assistant_message():
            return None
        return (
            "현재 응답은 작업 결과처럼 보이지만, 이번 턴에서 그 결과를 뒷받침하는 도구 실행이 "
            "없습니다. 이전 턴 결과를 재사용하지 말고, 현재 요청에 맞는 도구를 실행하세요. "
            "요청이 모호하면 ask_user로 필요한 데이터와 정리 기준을 물어보세요."
        )

    def _actionable_instruction_response_observation(
        self, state: _TurnLoopState, text: str
    ) -> str | None:
        if state.turn_state.last_successful_work_call is not None:
            return None
        if not self._request_allows_tool_work(state.effective_request):
            return None
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        looks_like_instruction = any(
            term in lowered or term in compact or term in text
            for term in (
                "call_tool",
                "next action",
                "다음 액션",
                "도구를 호출",
                "실행하세요",
                "생성하세요",
                "저장해줘",
                "저장해주세요",
                "정렬해줘",
                "알려줘",
                "해줘",
            )
        )
        if not looks_like_instruction:
            return None
        return (
            "최종 답변이 실행 결과가 아니라 다음 행동 지시나 사용자 요청 반복처럼 보입니다. "
            "사용자에게 다시 지시하지 말고, 필요한 source 파일을 읽고 도구 실행 또는 writeFile로 "
            "현재 요청을 직접 완료한 뒤 실제 결과를 답하세요."
        )

    def _looks_like_clarification(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return stripped.endswith("?")

    def _looks_like_work_result(self, text: str) -> bool:
        lowered = text.lower()
        if self._numbers_in_text(text):
            return True
        return any(
            marker in lowered
            for marker in (
                ":",
                "완료",
                "저장",
                "합계",
                "평균",
                "rows",
                "path",
                "removed",
            )
        )

    def _has_previous_result_like_assistant_message(self) -> bool:
        return any(
            message.role == "assistant" and self._looks_like_work_result(message.content)
            for message in self.conv.body()
        )

    def _numbers_in_text(self, text: str) -> list[float]:
        numbers: list[float] = []
        for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text):
            try:
                numbers.append(float(match.group(0)))
            except ValueError:
                continue
        return numbers

    def _numbers_close(self, left: float, right: float) -> bool:
        return abs(left - right) <= 0.01

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
        if instruction_echo := self._actionable_instruction_response_observation(
            state, action.summary or ""
        ):
            self._append_observation(state, instruction_echo)
            return "continue"
        if ungrounded := self._ungrounded_current_turn_result_observation(
            state, action.summary or ""
        ):
            self._append_observation(state, ungrounded)
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
            if instruction_echo := self._actionable_instruction_response_observation(
                state, action.text
            ):
                self._append_observation(state, instruction_echo)
                return "continue"
            if validation := self._terminal_validation_observation(state):
                self._append_observation(state, validation)
                return "continue"
            if ungrounded := self._ungrounded_current_turn_result_observation(state, action.text):
                self._append_observation(state, ungrounded)
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
        blocked_ask = self._blocked_ask_user_observation(
            action.question, state.effective_request, state.missing_workspace_paths
        )
        if blocked_ask is not None:
            if blocked_ask == state.last_blocked_ask_observation:
                state.blocked_ask_repeats += 1
            else:
                state.last_blocked_ask_observation = blocked_ask
                state.blocked_ask_repeats = 0
            self._append_observation(state, blocked_ask)
            if state.blocked_ask_repeats > self.deps.max_fix_retries:
                self.conv.add_observation("같은 차단된 질문이 진전 없이 반복되어 작업을 중단합니다.")
                state.result.stopped_reason = "no_progress"
                return "break"
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
            self._record_blocked_tool_action(state, action, "managed_tool_required")
            self._append_observation(state, self._managed_tool_observation(state))
            return "continue"

        if validation_failed := self._validation_failed_tool_update_observation(state, action.name):
            self._append_observation(state, validation_failed)
            return "continue"

        inline_data_block = self._generated_tool_inline_data_observation(state, action)
        if inline_data_block is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._record_blocked_tool_action(state, action, "inline_data_not_workspace_file")
            self._append_observation(state, inline_data_block)
            return "continue"

        if self._repair_generated_tool_path_input(state, action):
            sig = self._action_signature(action)

        output_as_source_block = self._generated_tool_output_as_source_observation(state, action)
        if output_as_source_block is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._record_blocked_tool_action(state, action, "output_path_used_as_source")
            self._append_observation(state, output_as_source_block)
            return "continue"

        missing_source_block = self._generated_tool_missing_requested_source_observation(state, action)
        if missing_source_block is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._record_blocked_tool_action(state, action, "missing_requested_source_file")
            self._append_observation(state, missing_source_block)
            return "continue"

        read_only_write_block = self._generated_tool_read_only_write_observation(state, action)
        if read_only_write_block is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._record_blocked_tool_action(state, action, "read_only_request_would_write")
            self._append_observation(state, read_only_write_block)
            return "continue"

        current_turn_mismatch = self._current_turn_tool_mismatch_observation(state, action)
        if current_turn_mismatch is not None:
            state.turn_state.record_tool_call(action.name, False)
            state.last_tool_call_index = state.turn_state.action_index
            self._record_blocked_tool_action(state, action, "tool_intent_mismatch_current_request")
            self._append_observation(state, current_turn_mismatch)
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

        if self.policy is not None:
            allowed, blocked = self._gate(action.name, action.input)
            if not allowed:
                assert blocked is not None
                self._append_observation(state, blocked)
                return "continue"

        called_tool = self.deps.registry.get(action.name)
        state.last_tool_inputs[action.name] = dict(action.input)
        self._remember_workspace_paths(self._paths_from_payload(action.input))
        res = self.deps.registry.call(action.name, action.input)
        self._log_tool_call(action.name, action.input, res)
        if res.ok:
            state.fix_failures = 0
            if action.name == "searchDocs":
                state.used_search_docs = True
            assert sig is not None
            observation = self._tool_success_observation(action.name, res)
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
            if called_tool is not None and called_tool.origin == "generated":
                missing_path = self._missing_path_from_tool_error(res.error)
                if missing_path is not None:
                    state.missing_workspace_paths.add(missing_path)
                state.validation_failed_tool_indices[action.name] = state.turn_state.action_index
                concise_failure = self._concise_failed_observation(observation) or observation
                state.validation_failed_tool_messages[action.name] = concise_failure

        state.turn_state.record_tool_call(action.name, res.ok)
        state.last_tool_call_index = state.turn_state.action_index
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
        if warning := self._read_only_path_only_result_observation(
            state.effective_request, action, res
        ):
            self._append_observation(state, warning)
        if not res.ok and state.fix_failures > self.deps.max_fix_retries:
            self._append_observation(state, "연속 실패가 한계를 넘었습니다. 작업을 중단합니다.")
            state.result.stopped_reason = "consecutive_failures"
            return "break"
        if res.ok and action.name not in {"readFile", "listFiles", "searchDocs"}:
            post_validation = self._validate_completed_work(
                state.effective_request, action, res
            )
            if post_validation is not None:
                ok, message = post_validation
                self._append_observation(state, message)
                if ok:
                    output_path = self._likely_output_path(
                        state.effective_request, Path(message.split()[0]).suffix
                    )
                    if output_path is not None:
                        state.completed_output_paths.add(output_path)
                    if called_tool is not None and called_tool.origin == "generated":
                        state.result.summary = self._validated_tool_summary(
                            action.name, res, message
                        )
                        state.result.stopped_reason = "cached_result"
                        return "break"
                    return "continue"
                if called_tool is not None and called_tool.origin == "generated":
                    state.validation_failed_tool_indices[action.name] = (
                        state.turn_state.action_index
                    )
                    state.validation_failed_tool_messages[action.name] = message
                    update_hint = self._validation_failed_tool_update_observation(
                        state, action.name
                    )
                    if update_hint is not None:
                        self._append_observation(state, update_hint)
        return "continue"

    def _handle_create_tool_action(self, state: _TurnLoopState, action: CreateTool) -> _StepAction:
        entrypoint_block = self._generated_tool_entrypoint_observation(
            action.spec.name, action.spec.code, action.spec.entrypoint
        )
        if entrypoint_block is not None:
            self._append_observation(state, entrypoint_block)
            return "continue"
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
            existing_spec = self.generated.specs().get(action.spec.name) if self.generated else None
            if existing_spec is not None and existing_spec.code != action.spec.code:
                self._tool_result_cache.clear()
                update_action = UpdateTool(
                    action="update_tool", name=action.spec.name, code=action.spec.code
                )
                observation = (
                    f"도구 {action.spec.name}은(는) 이미 이 턴에서 생성했지만 새 코드가 달라 "
                    "수정으로 처리합니다.\n"
                    + self._handle_update(update_action)
                    + self._tool_call_next_step_hint(action, state.effective_request)
                )
                state.turn_state.record_tool_change(action.spec.name)
                state.validation_failed_tool_indices.pop(action.spec.name, None)
                state.validation_failed_tool_messages.pop(action.spec.name, None)
                self._append_observation(state, observation)
                return "continue"
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
        if self.generated is None or action.name not in self.generated.specs():
            self._append_observation(state, self._handle_update(action))
            return "continue"
        entrypoint_block = self._generated_tool_entrypoint_observation(action.name, action.code)
        if entrypoint_block is not None:
            self._append_observation(state, entrypoint_block)
            return "continue"
        file_read_block = self._generated_tool_file_read_observation(
            state, action.name, action.code
        )
        if file_read_block is not None:
            self._append_observation(state, file_read_block)
            return "continue"
        auto_call_input = state.last_tool_inputs.get(action.name)
        if self._updated_tool_needs_execution(state, action.name) and auto_call_input is None:
            self._append_observation(
                state,
                self._updated_tool_call_observation(action.name, state.effective_request),
            )
            return "continue"
        self._tool_result_cache.clear()
        observation = self._handle_update(action)
        state.turn_state.record_tool_change(action.name)
        state.validation_failed_tool_indices.pop(action.name, None)
        state.validation_failed_tool_messages.pop(action.name, None)
        self._append_observation(state, observation)
        if auto_call_input is not None:
            self._append_observation(
                state,
                f"수정한 도구 {action.name}을(를) 이전 입력으로 즉시 실행해 결과를 확인합니다.",
            )
            state.turn_state.advance()
            auto_call = CallTool(action="call_tool", name=action.name, input=auto_call_input)
            return self._handle_call_tool_action(
                state, auto_call, self._action_signature(auto_call)
            )
        return "continue"

    def _finish_turn(self, result: TurnResult) -> None:
        if not result.summary and result.stopped_reason != "finish":
            self._finalize_incomplete(result)
        if result.stopped_reason in {"finish", "cached_result"}:
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
        visible = self._visible_incomplete_observation_from_history(result.observations)
        message = self._incomplete_stop_message(result.stopped_reason)
        label = self._incomplete_detail_label(result.stopped_reason, visible)
        result.summary = message + (f"\n{label}: {visible}" if visible else "")
        self.tracer.log(kind="error", errorKind=result.stopped_reason, message=result.summary)

    def _visible_incomplete_observation_from_history(self, observations: list[str]) -> str:
        """Return the latest user-useful observation, skipping terminal guard text."""
        for observation in reversed(observations):
            visible = self._visible_incomplete_observation(observation)
            if visible:
                return visible
        return ""

    def _incomplete_detail_label(self, stopped_reason: str, visible: str) -> str:
        """Choose a precise label for the extra detail shown after an incomplete stop."""
        if stopped_reason == "consecutive_failures" or " 실패: " in visible:
            return "마지막 실패"
        return "마지막 결과"

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
        if concise := self._concise_failed_observation(observation):
            return concise
        hidden_markers = (
            "작업 영역 파일의 구조는 사용자에게 묻지 않습니다.",
            "패키지 설치 질문은 사용자에게 묻지 않습니다.",
            "같은 동작이 진전 없이 반복되어",
            "계속 진행하세요.",
            "생성·등록 완료",
            "수정 완료",
            "사용자 답변:",
            "이미 이 턴에서 생성했습니다.",
            "연속 실패가 한계를 넘었습니다.",
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

    def _concise_failed_observation(self, observation: str) -> str:
        match = re.match(r"도구 (?P<tool>[\w.-]+) 실패: (?P<error>.*)", observation, re.S)
        if match is None:
            return ""
        tool_name = match.group("tool")
        error = match.group("error").strip()
        if file_match := re.search(r"No such file or directory: ['\"]([^'\"]+)['\"]", error):
            missing_path = file_match.group(1)
            display_path = (
                Path(missing_path).name if Path(missing_path).is_absolute() else missing_path
            )
            return f"도구 {tool_name} 실패: {display_path} 파일이 존재하지 않습니다."
        if "Traceback (most recent call last):" in error:
            traceback_text = error.split("\n파일 처리 코드가 실패했습니다.", 1)[0]
            traceback_text = traceback_text.split(
                "\nCSV DictReader의 row는 dict라서", 1
            )[0]
            last_line = next(
                (line.strip() for line in reversed(traceback_text.splitlines()) if line.strip()),
                "",
            )
            if last_line:
                return f"도구 {tool_name} 실패: {last_line}"
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
