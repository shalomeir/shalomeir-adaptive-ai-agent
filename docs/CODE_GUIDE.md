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

읽는 순서를 추천하면 `schemas.py` → `parsing.py` → `source_contracts.py` → `runner.py` → `tools/` → `skills.py`
순서다. action의 모양을 먼저 익히면 나머지가 쉽게 읽힌다.

## 2. 한 턴이 도는 과정

`runner.py`의 `run_turn`은 한 턴의 루프 뼈대만 담당한다. 최대 반복 안에서 LLM에게 묻고,
파싱 결과를 loop guard에 통과시킨 뒤 action dispatcher로 넘긴다. action별 세부 처리는
`_handle_respond_action`, `_handle_call_tool_action` 같은 작은 메서드가 맡고, 턴 종료 정리는
`_finish_turn`이 맡는다.
LLM 호출에는 누적 대화만 그대로 던지지 않는다. `_planning_messages`가 현재 요청, 이번 턴의
action index, 생성·수정한 도구, 호출한 generated tool, 마지막 tool input, 검증 실패, missing
workspace path, 차단된 tool action, 최근 observation을 담은 `[runtime-state]` JSON 메시지를 매번 합성해 붙인다.
이 메시지는 대화 저장소에 누적하지 않고 호출 시점에만 생성하므로, 모델은 현재 루프 상태를 구조적으로
보고 다음 action을 고를 수 있고 히스토리는 불필요하게 부풀지 않는다.
`readFile` 같은 context tool은 같은 입력으로 반복 호출돼도 캐시 결과를 최종 답변으로 접지 않는다.
이미 읽은 preview는 다음 계획의 근거로만 쓰고, 실제 작업 도구 생성·호출로 넘어가야 한다.

```python
def run_turn(self, request: str) -> TurnResult:
    state = self._start_turn(request)
    with self.tracer.trace():
        for _ in range(self.deps.max_iterations):
            raw = self._plan_raw()                 # 1) LLM 호출
            parsed = parse_action_text(raw)        # 2) 복구 + 검증
            if not parsed.ok or parsed.action is None:
                step = self._handle_parse_failure(state, parsed.error)
                if step == "break":
                    break
                continue
            sig, step = self._apply_loop_guards(state, parsed.action)
            if step == "continue":
                continue
            if step == "break":
                break
            step = self._dispatch_action(state, parsed.action, sig)  # 3) action 분기 실행
            if step == "break":
                break
        else:
            state.result.stopped_reason = "max_iterations"
        self._finish_turn(state.result)            # 4) 미완결 폴백, 영속화, compaction
    return state.result
```

인사, 모델명 질문, 파일 작업 같은 입력은 모두 같은 ReAct 루프에 들어간다. 러너는 자연어
문구를 세밀하게 해석하지 않고, 모델이 낸 action을 파싱한 뒤 도구·정책 경계에서 검증한다.
요청 본문과 tool payload의 workspace 경로, inline JSON/CSV 여부처럼 루프와 독립적인 source
판별은 `source_contracts.py`의 순수 helper가 맡는다. runner는 그 결과를 이용해 요청 파일과
무관한 이전 도구 호출, workspace 파일 요청에 임의 inline payload를 주입하는 호출, inline 데이터
요청을 존재하지 않는 workspace 파일로 치환하는 호출을 observation으로 되돌린다.
system prompt와 `[runtime-state]`는 현재 설정된 workspace root를 알려 주고, 사용자가 특별히
다른 위치를 지정하지 않으면 `events.csv` 같은 파일명과 상대 경로를 그 workspace 안의 파일로
가정하도록 명시한다. 파일이 없을 수는 있으므로 최종 답변 전에 read/list/tool 실행으로 확인한다.

종료는 다섯 갈래다: `finish` action, `final`이 참인 `respond`, 최대 반복 도달,
도구 호출이 연속으로 실패해 상한을 넘는 경우, 그리고 LLM이 유효한 JSON action을
연속으로 못 내 파싱 실패 상한을 넘는 경우(`parse_failures`). 마지막 갈래는 약한 모델이
형식을 잃고 헛도는 것을 `max_iterations`까지 가기 전에 끊는다.

뒤의 두 갈래(최대 반복·연속 실패)는 완료 신호 없이 끝나므로 `summary`가 비어 있다.
약한 모델이 답을 내는 도구는 다 돌려놓고 `respond(final)`/`finish`로 끝맺지 못해 반복을
소진하는 경우가 그렇다. `_finalize_incomplete`는 이때 내부 observation을 그대로 노출하지 않고,
가능한 마지막 실행 결과만 추려 사용자용 종료 메시지를 만든다. 종료 사유는 `error` 이벤트로
남겨 로그만으로도 추적되게 한다.

`TurnExecutionState`는 파일 결과가 필요한 턴에서 생성/수정한 도구를 실제로 실행했는지 기록한다.
이 상태 검증 덕분에 생성 도구를 등록만 하고 종료하는 흐름을 문장 패턴이 아니라 실행 증거로 막는다.
같은 턴에서 차단된 tool action도 runtime state에 남긴다. 모델이 이전 턴 도구를 다시 고르거나
현재 요청을 그대로 확인 질문으로 되풀이하면, 러너는 이를 사용자 상호작용으로 넘기지 않고 다른
도구 생성·수정·호출 경로를 선택하라는 observation으로 되돌린다.
또 사용자가 "tool을 만들어서", "도구를 만들어"처럼 도구 생성을 명시하면, 러너는 `runPython`이나
기존 도구 호출로 바로 답하는 액션을 observation으로 되돌리고 `create_tool` 뒤 생성 도구 `call_tool`을
요구한다. 이 검사는 에이전트의 핵심 흐름인 도구 생성·실행을 프롬프트 의존이 아니라 런타임 불변식으로
보호한다.
명시적 도구 생성 요청이 아니더라도 workspace 파일을 줄 단위로 읽고 필터링·변환·집계해야 하는
요청은 생성 도구 경로가 기본이다. `runPython`은 간단한 산술, regex 확인, 작은 list/filter 검증처럼
짧은 임시 snippet에 남겨 두고, workspace 파일 I/O나 재사용 가능한 데이터 처리 작업이면 생성 도구를
만들도록 되먹인다.
이 판단은 `hp`, `date`, `type` 같은 필드명이나 "알려줘" 같은 일반 동사만으로 켜지지 않고,
평균·합계·정렬·필터·변환·저장처럼 실제 작업 동작이 보일 때 강하게 작동한다.
도구를 만든 뒤에도 완료 조건은 "생성"이 아니라 "생성한 도구의 실행"이다. 생성 후 `runPython`,
`writeFile`, 기존 도구로 우회하려는 액션은 다시 observation으로 되먹이고, 방금 만든 generated tool을
실행하게 한다.

```python
if not res.ok and fix_failures > self.deps.max_fix_retries:
    result.stopped_reason = "consecutive_failures"
    break
```

`call_tool`은 `ToolRegistry`에서 도구 존재와 필수 입력 필드를 먼저 확인한 뒤 실행된다.
성공 시 실패 카운터를 0으로 되돌리고, 실패 시 1씩 올린다. 이 폐루프가 "실패를
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
action 없이 결과 JSON만 나온 경우에는 프로토콜 위반으로 본다. 오류 observation에는 직전
응답과 올바른 `respond(final=true)` / `finish` 예시를 넣어, 다음 LLM 호출이 같은 bare JSON을
반복하지 않고 규약에 맞는 최종 응답으로 교정하게 한다.

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
이름·설명과 compact input field hint만 담은 digest로 노출한다. 캐시 안정성을 위해 전체 코드와
큰 스키마 본문은 평소에 넣지 않는다.
동일한 성공 tool call은 캐시해 같은 입력의 재실행과 반복 권한 확인을 피한다. 모델이 같은
호출을 계속 반복하면 캐시된 결과를 최종 답변으로 접는다.

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
`writeFile`은 파일 쓰기 정책을 거친다. 작업 영역 밖 경로는 바로 거부하고, 안쪽 쓰기는 사용자에게
묻는다. 파일 출력 경로를 받는 생성 도구는 작업 영역 밖 경로만 실행 전 거부하고, 작업 영역 내부
상대 경로는 임시/출력 파일로 사용할 수 있게 둔다. 동일한 성공 tool call은 캐시를 재사용해 같은
승인 질문을 계속 띄우지 않는다.

```python
def _gate_file_write_path(self, name, path):
    escapes = path.startswith("/") or path.startswith("~") or ".." in Path(path).parts
    action_id = "out_of_workspace" if escapes else "write_file"
    decision = self.policy.evaluate(action_id)
    if decision.decision == "DENY":
        return False, f"정책상 거부됨: {action_id}"
    if decision.decision == "ASK_USER" and not self.policy.confirm(action_id):
        return False, "사용자가 작업을 거부했습니다."
    return True, None
```

생성 도구는 `output`, `dst`, `destination`, `target` 같은 출력 경로 필드를 payload에 담고 있으면
경로 이탈 후보로 본다. 해당 값이 절대 경로, `~`, 또는 `..`를 포함하면 모델 판단과 무관하게
거부한다.

파일 출력처럼 결과 상태가 중요한 요청은 도구 성공만으로 끝내지 않고, 도구가 반환한 출력 경로나
`writeFile` payload를 기준으로 작업 영역 파일을 다시 읽어 가벼운 sanity check를 한다. JSON은
파싱 가능해야 하고, CSV는 header와 행을 읽을 수 있어야 하며, 텍스트 파일은 비어 있지 않아야 한다.
러너는 CSV 중복 제거·정렬·group sum, JSON/object-tree 삭제 조건, markdown table 값 검산 같은
semantic contract를 독립 재계산하지 않는다. 그런 판단은 생성 도구와 LLM의 책임으로 두고,
런타임은 경로·권한·파일 형식·반복 진행 상태처럼 작업 종류와 독립적인 경계만 검증한다. 생성 도구가
파일 출력 sanity check를 통과하면 루프는 `cached_result`로 닫을 수 있다. 검증된 산출물을 다시
LLM planning에 넘기면 후속 action이 같은 작업을 반복하거나 산출물을 덮어쓸 수 있기 때문이다.
마찬가지로 최종 답변의 숫자를 도구 결과와 다시 대조해 막지 않는다. 조건 임계값, 개수, 반올림 표현처럼
정상적인 숫자가 섞일 수 있고, 이 검산을 런타임에 넣으면 모델이 이미 답을 냈는데도 반복 루프에 빠질 수
있기 때문이다. 숫자 환각 방지는 system prompt와 도구 observation grounding에 맡긴다.
이 guard는 workspace 파일명이 같다는 이유만으로 이전 generated tool을 재사용하지 않는다. action
input이 요청 파일을 명시적으로 연결하거나 tool code가 파일을 직접 읽더라도, 도구 설명과 이름의
작업 의도가 현재 요청과 충분히 맞지 않으면 호출 전에 observation으로 되돌린다. 또한 현재 턴에
작업 실행 근거가 없는데 이전 assistant 결과처럼 보이는 최종 답변을 내면, 현재 요청에 맞는 도구
실행이나 사용자 확인으로 되돌린다.
파일 구조 확인 질문도 현재 요청의 파일에 anchor되어 있을 때만 자동 차단한다. 모델이 이전 문맥의
파일명을 모호한 새 요청에 끌어오면 runtime이 그 파일을 읽도록 강화하지 않고, 사용자 확인으로
흘려 보낸다. `ask_user`가 같은 차단 observation을 반복하면 조기 no-progress로 닫고, `text` 필드로
질문을 내는 흔한 schema 오류는 parser가 `question` alias로 복구한다. 파일이 실제로 없다는
확인 질문은 차단하지 않는다.
후속 요청이 "방금 필터된 결과"처럼 이전 tool result를 변환하는 흐름이면 generated tool이 그
record list를 input으로 받을 수 있다. 반대로 현재 요청에 명시된 source 파일이 있으면 임의 inline
sample payload를 차단한다. 출력 파일명만 있는 contextual follow-up은 그 출력 경로를 source로
오해하지 않는다.
같은 이름의 `create_tool` 반복은 보통 기존 도구를 `call_tool`하라는 observation으로 되돌리지만,
새 code가 기존 code와 다르면 corrected create로 보고 update 경로로 처리한다.
generated tool 실행 실패도 update-required 상태로 기록해 같은 failing call이 반복 실행되지 않게
한다.

`PolicyManager`는 `_DENY`, `_ASK` 집합으로 행동을 분류한다. 작업 영역 밖 접근은 `_DENY`에
있어 사용자 승인으로도 우회되지 않는다.

## 7. skill 영속화와 재로딩

작업을 마치면 그 세션에서 만든 도구를 현재 세션 전용으로 둘지, 다음 세션에서도 재사용하도록
영구 저장할지 묻는다(`_offer_persist`). 거부하면 `.session` 도구로만 남고, 승인 시 `SkillStore`가
`skills/<name>/`에 `tool.py`, `manifest.json`, `SKILL.md`를 쓴다. 다시 저장하면 버전을 올리며
생성 시각은 보존한다.

다음 세션에서 러너는 시작할 때 manifest를 읽어 도구를 다시 등록한다. 프롬프트에는 이름·설명만
올라가고, 코드는 디스크에 남아 호출 시 실행된다.

```python
if self.skills is not None and self.generated is not None:
    for digest in self.skills.load_digests():        # 이름·설명·입력 필드 hint
        spec = self.skills.load_spec(digest.name)    # 코드 본문
        self.deps.registry.register(self.generated.create(spec))
```

## 8. 컨텍스트 관리

`ConversationStore`는 시스템 프롬프트를 본문과 분리해 보관한다. 시스템 접두가 매 턴 그대로라
프롬프트 앞부분이 안정적이고, 변화는 끝에만 덧붙는다. `ContextManager`는 추정 토큰이 설정된
임계를 넘으면 오래된 구간을 짧은 extractive summary로 접되, 코드가 강제로 보존하는 핵심 사실을
함께 남긴다.
`HttpLLMClient`는 provider payload를 만들 때 프로토콜 system prompt와 도구 목록을 하나의
system 메시지로 합친다. 로컬 모델이 여러 system 메시지 중 도구 목록이나 최근 observation에
끌려 action 규약을 잊는 것을 줄이기 위해, strict output contract를 항상 첫 system 메시지의
앞부분에 둔다.

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

`Tracer`는 생성될 때 `sessionId`를 만들고 `logs/session-*.jsonl` 파일 하나에 그 세션의
이벤트를 기록한다. `runner.py`는 LLM 호출 직후 `llm_call` 이벤트에 응답 원문 preview,
전체 글자 수, 잘림 여부를 기록한다. preview는 로컬 진단용이며 무한정 커지지 않도록
4000자로 제한한다. LLM이 `call_tool` 액션을 고른 시점에는 `llm_call` 이벤트에 도구 이름과
입력 preview를 남긴다. 실제 도구 실행 뒤에는 `tool_call` 이벤트에 도구 이름, 성공 여부,
입력 preview, 출력 또는 오류 preview를 같은 방식으로 남겨, LLM이 아닌 런타임 도구의 입출력도
로그만으로 추적할 수 있게 한다. preview 직렬화 전에는 `apiKey`, `token`, `password` 같은
민감 필드명을 마스킹한다.

## 10. 따라 읽는 순서

1. `schemas.py` — action과 도구 계약의 모양
2. `parsing.py` — 약한 모델 출력 방어
3. `source_contracts.py` — 요청·payload의 경로와 inline structured data 판별
4. `runner.py` — 전체 루프와 분기
5. `tools/base.py`, `tools/registry.py`, `tools/builtins.py` — 도구 모델과 내장 도구
6. `tools/generated.py`, `sandbox.py` — 생성 도구의 격리 실행
7. `skills.py` — 영속화와 재로딩
8. `policy.py`, `context.py`, `observability.py` — 권한·컨텍스트·관측
9. `cli.py` — 조립

테스트는 각 모듈 옆의 `tests/test_*.py`가, 끝에서 끝까지 흐름은
`tests/test_demo_integration.py`가 보여준다.
