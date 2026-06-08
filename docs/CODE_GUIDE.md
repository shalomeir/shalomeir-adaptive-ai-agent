# 코드 읽기 가이드

코드를 처음 여는 사람을 위해, 어디서 무슨 일이 일어나는지를 흐름 순서로 설명한다. 모든
인용은 실제 소스에서 발췌했다. 전체 설계는 `specs/`, 실행법은 `README.md`를 본다.

## 1. 한눈에 보는 구조

요청 하나가 처리되는 길은 이렇다.

```
사용자 입력
  └─ cli.chat            설정을 읽고 도구·러너를 조립한다
       └─ AgentRunner.run_turn   ← 핵심 루프
            ├─ LLMClient.chat          다음 action을 받는다
            ├─ parse_action_text       JSON을 복구·검증한다
            ├─ (분기) respond / ask_user / call_tool / create_tool / update_tool / finish
            │     ├─ ToolRegistry.call          내장·생성 도구 실행
            │     ├─ GeneratedToolManager        생성 도구를 sandbox에서 실행
            │     └─ PolicyManager               쓰기 같은 부수효과를 게이트
            ├─ ConversationStore / ContextManager   대화 누적과 compaction
            ├─ SkillStore               승인된 도구를 저장·재로딩
            └─ Tracer                   매 단계를 JSONL로 기록
```

읽는 순서를 추천하면 `schemas.py` → `parsing.py` → `runner.py` → `tools/` → `skills.py`
순서다. action의 모양을 먼저 익히면 나머지가 쉽게 읽힌다.

## 2. 한 턴이 도는 과정

`runner.py`의 `run_turn`이 전부의 중심이다. 최대 반복 안에서 LLM에게 묻고, 받은 action을
분기 실행하고, 결과를 observation으로 쌓는다.

```python
def run_turn(self, request: str) -> TurnResult:
    direct_response = self._direct_conversation_response(request)
    if direct_response is not None:
        return TurnResult(summary=direct_response) # 0) 짧은 대화는 루프 밖에서 처리
    self.conv.add_user(request)
    result = TurnResult()
    fix_failures = 0
    with self.tracer.trace():
        for _ in range(self.deps.max_iterations):
            raw = self._plan_raw()                 # 1) LLM 호출
            parsed = parse_action_text(raw)        # 2) 복구 + 검증
            if not parsed.ok or parsed.action is None:
                self.conv.add_observation(parsed.error or "알 수 없는 파싱 오류")
                continue                           #    깨지면 오류를 되먹이고 다시
            action = parsed.action
            if isinstance(action, Finish):         # 3) action 종류로 분기
                result.summary = action.summary or ""
                break
            ...
        else:
            result.stopped_reason = "max_iterations"   # 반복 소진 시 종료
        if not result.summary and result.stopped_reason != "finish":
            self._finalize_incomplete(result)      # 3.5) 미완결 종료 폴백
        self._offer_persist()                      # 4) 생성 도구 저장 제안
        self.ctx.maybe_compact(self.conv)          # 5) 길어지면 요약
    return result
```

`_direct_conversation_response`는 인사, 모델명 질문, "그냥 대화" 같은 입력을 도구 계획으로
오인하지 않게 하는 앞단이다. 이 경로는 LLM 호출도, 로그 이벤트도 만들지 않는다. 실제 작업
요청만 아래 ReAct 루프에 들어간다.

종료는 다섯 갈래다: `finish` action, `final`이 참인 `respond`, 최대 반복 도달,
도구 호출이 연속으로 실패해 상한을 넘는 경우, 그리고 LLM이 유효한 JSON action을
연속으로 못 내 파싱 실패 상한을 넘는 경우(`parse_failures`). 마지막 갈래는 약한 모델이
형식을 잃고 헛도는 것을 `max_iterations`까지 가기 전에 끊는다.

뒤의 두 갈래(최대 반복·연속 실패)는 완료 신호 없이 끝나므로 `summary`가 비어 있다.
약한 모델이 답을 내는 도구는 다 돌려놓고 `respond(final)`/`finish`로 끝맺지 못해 반복을
소진하는 경우가 그렇다. `_finalize_incomplete`가 이때 마지막 observation(대개 실제 결과)을
요약으로 노출하고, 종료 사유를 `error` 이벤트로 남겨 로그만으로도 추적되게 한다.

```python
if not res.ok and fix_failures > self.deps.max_fix_retries:
    result.stopped_reason = "consecutive_failures"
    break
```

`call_tool` 성공 시 실패 카운터를 0으로 되돌리고, 실패 시 1씩 올린다. 이 폐루프가 "실패를
관찰해 고치고 다시 실행"하는 자가수정의 뼈대다. 실패 observation을 본 다음 턴에서 LLM이
`update_tool`을 부르면 코드가 교체되고 재실행된다.

## 3. action 프로토콜과 파싱

모델은 매 턴 하나의 JSON 객체를 반환한다. `schemas.py`가 이를 discriminated union으로
정의한다.

```python
class CallTool(BaseModel):
    action: Literal["call_tool"]
    name: str
    input: dict[str, Any]

AgentAction = Annotated[
    Respond | AskUser | CallTool | CreateTool | UpdateTool | Finish,
    Field(discriminator="action"),
]
```

약한 모델은 코드펜스나 trailing 쉼표로 출력을 깨뜨린다. `parsing.py`가 펜스를 벗기고
가장 바깥 `{...}`를 떼어내 trailing 쉼표를 지운 뒤 `json.loads`한다. 그래도 형식이 어긋나면
사람이 읽을 수 있는 오류 문자열을 만들어 다시 묻는다.

```python
def parse_action_text(raw: str) -> ParseResult:
    try:
        data = json_repair(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return ParseResult(ok=False, error=f"출력이 유효한 JSON이 아닙니다: {e} ...")
    try:
        return ParseResult(ok=True, action=parse_agent_action(data))
    except Exception as e:
        return ParseResult(ok=False, error=f"action 형식을 어겼습니다: {e} ...")
```

## 4. 도구 모델

내장 도구와 생성 도구를 같은 `Tool` 추상으로 다룬다. 레지스트리는 실행도 하지만, 프롬프트에는
이름·설명만 담은 digest로 노출한다. 캐시 안정성을 위해 전체 스키마와 코드는 평소에 넣지 않는다.
명확한 JSON hp 질의 요청은 `queryMonsterHp`, CSV 중복 제거·정렬·저장 요청은
`normalizeCsv` 내장 도구가 처리해, 로컬 모델이 불안정한 ad hoc 검증 스크립트를 반복하지
않게 한다.

```python
@dataclass
class Tool:
    name: str
    description: str
    origin: Literal["builtin", "generated", "mcp"]
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    output_schema: dict[str, Any] | None = None
```

`ToolRegistry.call`은 도구가 없으면 오류 결과를 돌려주고, 핸들러가 던진 예외도 결과로
감싸서 루프가 끊기지 않게 한다.

## 5. 생성 도구가 실제로 실행되는 방식

가장 흥미로운 부분이다. 생성 도구는 `def run(input): ...`를 담은 `tool.py`로 디스크에
기록되고, 작은 runner 스크립트가 stdin으로 받은 JSON을 그 함수에 넘겨 결과를 표준 출력에
한 줄로 찍는다.

```python
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
```

이 스크립트를 `ExecutionSandbox`가 workspace를 cwd로 두고 subprocess로 돌린다. 그래서 생성
도구의 `open("events.csv")` 같은 상대 경로는 사용자가 보는 작업 영역 파일을 가리킨다. 같은
프로세스의 `exec`를 쓰지 않고 타임아웃·작업 디렉터리 제한·출력 제한을 건다.

```python
cmd = [sys.executable, "-I", script_path, *(args or [])]   # -I: 격리 모드
proc = subprocess.run(cmd, cwd=self.workspace, env=env,
                      input=stdin, capture_output=True, text=True,
                      timeout=self.timeout_sec)
```

세션 중 생성된 도구는 `workspace/.session/<tool-name>/` 아래에 기록된다. runner는 그
하위 경로의 `_runner.py`만 실행하도록 workspace 밖 스크립트 경로를 거부한다. macOS에서
`sandbox-exec`가 있고 네트워크 정책이 `deny`이면 subprocess 네트워크 접근도 차단한다.

결과는 `__RESULT__` 접두가 붙은 줄에서 읽는데, 마지막 일치 줄을 쓴다. 신뢰할 수 없는 코드가
가짜 `__RESULT__` 줄을 먼저 출력해 반환값을 위조하지 못하게 하기 위해서다.

```python
result_line = None
for line in res.stdout.splitlines():
    if line.startswith("__RESULT__"):
        result_line = line              # 마지막 것이 진짜 결과
```

## 6. 권한 게이트

부수효과가 있는 호출은 실행 전에 정책을 거친다. 판단은 모델이 아니라 런타임이 내린다.
현재는 `writeFile`을 막는다: 작업 영역 밖 경로는 바로 거부, 안쪽 쓰기는 사용자에게 묻는다.

```python
def _gate(self, name, payload):
    if name != "writeFile":
        return True, None
    path = str(payload.get("path", ""))
    escapes = path.startswith("/") or path.startswith("~") or ".." in Path(path).parts
    action_id = "out_of_workspace" if escapes else "write_file"
    decision = self.policy.evaluate(action_id)
    if decision.decision == "DENY":
        return False, f"정책상 거부됨: {action_id}"
    if decision.decision == "ASK_USER" and not self.policy.confirm(action_id):
        return False, "사용자가 작업을 거부했습니다."
    return True, None
```

`PolicyManager`는 `_DENY`, `_ASK` 집합으로 행동을 분류한다. 작업 영역 밖 접근은 `_DENY`에
있어 사용자 승인으로도 우회되지 않는다.

## 7. skill 영속화와 재로딩

작업을 마치면 그 세션에서 만든 도구를 저장할지 묻는다(`_offer_persist`). 승인 시
`SkillStore`가 `skills/<name>/`에 `tool.py`, `manifest.json`, `SKILL.md`를 쓰고, 다시
저장하면 버전을 올리며 생성 시각은 보존한다.

다음 세션에서 러너는 시작할 때 manifest를 읽어 도구를 다시 등록한다. 프롬프트에는 이름·설명만
올라가고, 코드는 디스크에 남아 호출 시 실행된다.

```python
if self.skills is not None and self.generated is not None:
    for digest in self.skills.load_digests():        # 이름·설명만
        spec = self.skills.load_spec(digest.name)    # 코드 본문
        self.deps.registry.register(self.generated.create(spec))
```

## 8. 컨텍스트 관리

`ConversationStore`는 시스템 프롬프트를 본문과 분리해 보관한다. 시스템 접두가 매 턴 그대로라
프롬프트 앞부분이 안정적이고, 변화는 끝에만 덧붙는다. `ContextManager`는 추정 토큰이 임계를
넘으면 오래된 구간을 요약으로 접되, 코드가 강제로 보존하는 핵심 사실을 함께 남긴다.

```python
old, recent = body[: -self.keep_recent], body[-self.keep_recent:]
summary = self.summarize(old)
facts = "\n".join(f"- {f}" for f in self._facts)
carry = Message(role="user", content=f"[요약] {summary}\n[보존된 핵심 사실]\n{facts}")
conv.replace_body([carry, *recent])
```

러너는 도구를 만들거나 고칠 때 그 이름을 `carry_over_fact`로 등록해, compaction 후에도
보유 능력이 사라지지 않게 한다.

## 9. 관측

`Tracer`는 한 요청을 trace로, 그 안의 LLM 호출과 도구 실행을 span으로 묶어 JSONL 한 줄씩
남긴다. span은 스택으로 관리해 각 이벤트가 부모 span(`parentSpanId`)을 기록하므로 중첩
관계가 로그에 드러난다. 남기는 필드는 허용 목록으로 제한해 민감 정보가 새지 않게 한다. 같은 이벤트를 교체
가능한 익스포터로도 전달하므로, 외부 모니터링은 호출부를 바꾸지 않고 붙일 수 있다. 전달 실패는
핵심 경로를 막지 않도록 무시한다.

```python
with self.path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
try:
    self.exporter.export(evt)
except Exception:
    pass
```

## 10. 따라 읽는 순서

1. `schemas.py` — action과 도구 계약의 모양
2. `parsing.py` — 약한 모델 출력 방어
3. `runner.py` — 전체 루프와 분기
4. `tools/base.py`, `tools/registry.py`, `tools/builtins.py` — 도구 모델과 내장 도구
5. `tools/generated.py`, `sandbox.py` — 생성 도구의 격리 실행
6. `skills.py` — 영속화와 재로딩
7. `policy.py`, `context.py`, `observability.py` — 권한·컨텍스트·관측
8. `cli.py` — 조립

테스트는 각 모듈 옆의 `tests/test_*.py`가, 끝에서 끝까지 흐름은
`tests/test_demo_integration.py`가 보여준다.
