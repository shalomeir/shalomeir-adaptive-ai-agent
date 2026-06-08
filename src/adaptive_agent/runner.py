from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
from .tools.generated import GeneratedToolManager
from .tools.registry import ToolRegistry

_SYSTEM = (
    "You are a task-solving agent that creates and runs small Python tools. You are not a "
    "general chatbot, and you are not any vendor's assistant — if asked who you are, say you are "
    "this tool-building agent and do not claim to be a specific company's model.\n"
    "On EVERY turn reply with EXACTLY ONE JSON object and nothing else — no prose, no code "
    "fences. Pick one action:\n"
    '- {"action":"respond","text":"...","final":true} — give an answer; final:true ends the task\n'
    '- {"action":"ask_user","question":"..."} — ask when the request is ambiguous\n'
    '- {"action":"call_tool","name":"<tool>","input":{...}} — run a built-in or created tool\n'
    '- {"action":"create_tool","spec":{"name":"kebab-name","description":"...",'
    '"code":"def run(input):\\n    return ...","inputSchema":{"type":"object"}}} — make a tool\n'
    '- {"action":"update_tool","name":"<tool>","code":"def run(input):\\n    return ..."} — fix a failed tool\n'
    '- {"action":"finish","summary":"..."} — stop when the task is done\n'
    'To use a tool you MUST use call_tool — never put a tool name in the "action" field. '
    "Example: to read the workspace file monsters.json, reply "
    '{"action":"call_tool","name":"readFile","input":{"path":"monsters.json"}}. '
    'Workspace file paths are RELATIVE (just the file name, e.g. "monsters.json"); do not use '
    'absolute paths or "..".\n'
    'There are TWO ways to run Python. Both are invoked with call_tool — "runPython" and any tool '
    'you create are tool NAMES, never values of the "action" field:\n'
    "(a) The built-in runPython tool. Invoke it as "
    '{"action":"call_tool","name":"runPython","input":{"code":"<script>"}}. The code is a TOP-LEVEL '
    "script: there is NO input variable, do NOT write return — PRINT the result, e.g. "
    "print(json.dumps(result)).\n"
    "(b) A tool you create_tool, then call_tool by its name. Its code defines run(input) where "
    '`input` is the call_tool "input" dict, and you RETURN the result.\n'
    "To process a workspace file, OPEN IT DIRECTLY inside your code by its relative name — your code "
    "runs with the workspace as the working directory. For example, a single call_tool to runPython "
    'with code "import json\\ndata = json.load(open(\\"monsters.json\\"))\\nresult = ...\\n'
    'print(json.dumps(result))" answers a one-off data question in one step. You do NOT need to '
    "readFile first or pass file contents through input; use readFile only to peek at a small file.\n"
    "For file output, prefer returning the computed text/data from runPython or a created tool, then "
    "call writeFile with the final relative path and content. This lets the runtime apply its file "
    "write policy.\n"
    "For CSV dedupe/sort/save requests, prefer the built-in normalizeCsv tool with "
    '{"src":"events.csv","dst":"events-clean.csv","sortBy":"date"} instead of ad hoc code.\n'
    "Use ONLY the Python standard library (json, csv, re, math, etc.) — pandas/numpy are NOT "
    "installed and will fail to import. NEVER ask the user to install packages; for CSV work use "
    "the built-in csv module. Reuse an existing tool instead of recreating it. When a tool fails, "
    "READ the error message carefully and fix it: use update_tool for a created tool, or call the "
    "tool again with corrected input. Do not repeat the same question; if you already have what you "
    "need, act. Keep tool names in kebab-case. When you have the answer, reply with respond "
    "(final:true) or finish — do not keep calling tools."
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


@dataclass
class TurnResult:
    summary: str = ""
    observations: list[str] = field(default_factory=list)
    stopped_reason: str = "finish"


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

        if compact in {"뭐야", "뭐야.", "머야", "머야."}:
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

    def _blocked_ask_user_observation(self, question: str) -> str | None:
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
        return None

    def _parse_direct_csv_normalize(self, request: str) -> dict[str, Any] | None:
        lowered = request.lower()
        if not (
            ".csv" in lowered
            and "date" in lowered
            and any(term in request for term in ("중복", "duplicate"))
            and any(term in request for term in ("정렬", "sort"))
            and any(term in request for term in ("저장", "save"))
        ):
            return None
        csv_paths = re.findall(r"[\w.-]+\.csv", request)
        if not csv_paths:
            return None
        src = csv_paths[0]
        dst = csv_paths[1] if len(csv_paths) >= 2 else f"{Path(src).stem}-clean.csv"
        return {"src": src, "dst": dst, "sortBy": "date"}

    def _plan_raw(self) -> str:
        with self.tracer.span():
            raw = self.deps.llm.chat(self.conv.messages(), self.deps.registry.digests())
            self.tracer.log(kind="llm_call", model=getattr(self.deps.llm, "model", None))
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
        elif name == "normalizeCsv":
            path = str(payload.get("dst", ""))
        else:
            return True, None
        escapes = path.startswith("/") or path.startswith("~") or ".." in Path(path).parts
        action_id = "out_of_workspace" if escapes else "write_file"
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
        return True, None

    def run_turn(self, request: str) -> TurnResult:
        direct_response = self._direct_conversation_response(request)
        if direct_response is not None:
            self.conv.add_user(request)
            self.conv.add_assistant(direct_response)
            return TurnResult(summary=direct_response)

        self.conv.add_user(request)
        direct_csv_payload = self._parse_direct_csv_normalize(request)
        if direct_csv_payload is not None and any(
            digest.name == "normalizeCsv" for digest in self.deps.registry.digests()
        ):
            return self._run_direct_csv_normalize(direct_csv_payload)

        result = TurnResult()
        fix_failures = 0
        parse_failures = 0
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
                    blocked_ask = self._blocked_ask_user_observation(action.question)
                    if blocked_ask is not None:
                        self.conv.add_observation(blocked_ask)
                        result.observations.append(blocked_ask)
                        continue
                    answer = self.deps.ask(action.question, action.choices)
                    obs = f"사용자 답변: {answer}"
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
                if isinstance(action, CallTool):
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
                        obs = f"도구 {action.name} 결과: {res.output}"
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
                    obs = self._handle_create(action)
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
                if isinstance(action, UpdateTool):
                    obs = self._handle_update(action)
                    self.conv.add_observation(obs)
                    result.observations.append(obs)
                    continue
            else:
                result.stopped_reason = "max_iterations"
            if not result.summary and result.stopped_reason != "finish":
                self._finalize_incomplete(result)
            self._offer_persist()
            self.ctx.maybe_compact(self.conv)
        return result

    def _run_direct_csv_normalize(self, payload: dict[str, Any]) -> TurnResult:
        result = TurnResult()
        with self.tracer.trace():
            if self.policy is not None:
                allowed, blocked = self._gate("normalizeCsv", payload)
                if not allowed:
                    assert blocked is not None
                    self.conv.add_observation(blocked)
                    result.observations.append(blocked)
                    result.summary = blocked
                    result.stopped_reason = "policy_denied"
                    return result
            res = self.deps.registry.call("normalizeCsv", payload)
            self.tracer.log(kind="tool_call", toolName="normalizeCsv")
            if not res.ok:
                obs = f"도구 normalizeCsv 실패: {res.error}"
                self.conv.add_observation(obs)
                result.observations.append(obs)
                result.summary = obs
                result.stopped_reason = "direct_tool_failure"
                self.tracer.log(kind="error", errorKind=result.stopped_reason, message=obs)
                return result
            obs = f"도구 normalizeCsv 결과: {res.output}"
            self.conv.add_observation(obs)
            result.observations.append(obs)
            output = res.output or {}
            result.summary = (
                f"{output.get('dst', payload['dst'])}에 중복 제거 및 date 오름차순 정렬 결과를 "
                f"저장했습니다. 고유 행 {output.get('rows')}개, 제거한 중복 "
                f"{output.get('removedDuplicates')}개입니다."
            )
            return result

    def _finalize_incomplete(self, result: TurnResult) -> None:
        """Surface a result when the loop ends without a completion signal.

        A weak model sometimes runs the tools that produce the answer but never
        emits respond(final)/finish, so it burns through the iteration budget. In
        that case an empty summary hides the work entirely; instead we report the
        last observation (which usually holds the answer) and record the stop
        reason as an error event so the run is traceable from the log alone.
        """
        last = result.observations[-1] if result.observations else ""
        result.summary = f"작업을 완결하지 못하고 {result.stopped_reason}(으)로 중단했습니다." + (
            f" 마지막 결과: {last}" if last else ""
        )
        self.tracer.log(kind="error", errorKind=result.stopped_reason, message=result.summary)

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
