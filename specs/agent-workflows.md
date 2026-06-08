# Agent 워크플로 — ReAct 루프와 주요 흐름

상태: 지속 관리 문서. 루프나 워크플로가 바뀌면 코드보다 먼저 이 문서를 갱신한다.

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
| --- | --- | --- |
| 0.1 | 2026-06-01 | 최초 작성. ReAct 루프, tool call, HIL, 권한, 실패 fallback, create_tool 워크플로 정의 |

## 0. 용어

- 턴: LLM에게 한 번 다음 행동을 묻고 그 행동을 처리하는 한 사이클.
- observation: 도구 실행 결과나 오류, 검증 결과처럼 다음 턴 입력에 더해지는 관찰값.
- trace: 한 번의 사용자 요청 처리 전체. 그 안의 LLM 호출과 도구 실행이 span이다.

action 스키마와 도구 입출력 스키마는 `schemas.md`를 따른다.

## 1. ReAct 루프의 구체화

이 에이전트의 ReAct는 Reason과 Act를 한 action 객체로 합쳐 표현한다. LLM은 매 턴 `thought`(Reason)와 `action`(Act)을 함께 반환하고, 런타임은 Act를 실행한 뒤 Observe 결과를 컨텍스트에 더한다.

```text
사용자 요청
   │
   ▼
[컨텍스트 조립]  system 규칙 + ToolDigest 목록 + 압축된 과거 + 최근 메시지
   │
   ▼
┌───────────────────────── 루프 (최대 maxIterations) ─────────────────────────┐
│  [Reason+Act]  LLM 호출 → AgentAction(JSON) 반환                              │
│        │                                                                    │
│        ▼                                                                    │
│  [Parse/Validate]  JSON repair → schema 검증                                 │
│        │ 실패 → 오류를 observation으로 넣고 재요청 (재시도 상한 내)           │
│        ▼                                                                    │
│  [Dispatch]  action 종류로 분기 (2~7절)                                       │
│        │                                                                    │
│        ▼                                                                    │
│  [Observe]  실행 결과·검증 결과를 observation으로 누적                          │
│        │                                                                    │
│        ▼                                                                    │
│  [Compaction 검사]  토큰 임계 초과면 과거 구간 요약 + 핵심 사실 carry-over      │
│        │                                                                    │
│        ▼                                                                    │
│  [종료 검사]  finish / 사용자 종료 / 최대 반복 / 연속 실패 상한                  │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[영속화 제안]  세션 중 생성 도구가 있으면 저장 여부를 묻는다
```

### 1.1 메인 루프 의사코드

```python
async def run_turn(request, context):
    context.add_user(request)
    for i in range(config.maxIterations):
        action = await plan(context)            # 2절: 파싱·검증·재요청 포함
        result = await dispatch(action, context)  # 3~7절
        if result.terminal:
            break
        context.add_observation(result.observation)
        if context.estimated_tokens() > config.compactionTokenThreshold:
            await context.compact()             # 핵심 사실 carry-over
    else:
        context.add_observation("최대 반복에 도달해 작업을 중단했습니다.")
    await offer_persist_generated_tools(context)
```

## 2. 계획 단계: 파싱과 재요청

```python
async def plan(context) -> AgentAction:
    for attempt in range(config.maxFixRetries + 1):
        raw = await llm.chat(context.messages(), tools=registry.digests())
        log(kind="llm_call", model=..., inputTokens=..., cacheHit=...)
        try:
            data = json_repair(raw)             # 흔한 깨짐 복구
            action = AgentAction.validate(data)  # pydantic 검증
            log(actionType=action.action, parseOk=True, retries=attempt)
            return action
        except ValidationError as e:
            context.add_observation(f"이전 출력이 형식을 어겼습니다: {e}. 같은 JSON 형식으로 다시 답하세요.")
            log(parseOk=False, retries=attempt)
    raise ProtocolError("형식 복구에 반복 실패")  # 종료로 이어진다
```

핵심: 코드 블록 추출에 의존하지 않는다. 약한 모델과 로컬 모델을 전제로, 깨진 출력을 복구하고 그래도 안 되면 오류를 되먹여 다시 묻는다.

## 3. Dispatch: action 분기

```python
async def dispatch(action, context) -> Result:
    match action.action:
        case "respond":      return respond_flow(action, context)
        case "ask_user":     return await ask_user_flow(action, context)   # 5절
        case "call_tool":    return await call_tool_flow(action, context)  # 4절
        case "create_tool":  return await create_tool_flow(action, context) # 6절
        case "update_tool":  return await update_tool_flow(action, context) # 7절
        case "finish":       return Result(terminal=True, observation=action.summary)
```

`respond`는 텍스트를 사용자에게 보여준다. `final`이 참이면 종료로 본다.

## 4. Tool call 워크플로

`call_tool`은 내장 도구든 생성 도구든 같은 경로로 처리한다.

```text
call_tool(name, input)
   │
   ▼
[레지스트리 조회]  없으면 → observation "그런 도구 없음. 목록 재확인하거나 생성하세요"
   │
   ▼
[전체 스키마 로드]  지연 로딩한 inputSchema로 input 검증 (실패 → observation에 오류)
   │
   ▼
[권한 게이트]  5절. DENY → 중단 observation / ASK_USER → 사용자 확인 / ALLOW → 진행
   │
   ▼
[실행]
   ├─ 내장 도구: 해당 함수 직접 호출
   └─ 생성 도구: ExecutionSandbox에서 subprocess 격리 실행 (timeout·경로·네트워크 제한)
   │
   ▼
[출력 정규화]  outputSchema 있으면 검증, 큰 출력은 잘라 저장
   │
   ▼
[검증]  8절. 결과 상태가 의도를 만족하는지 확인
   │
   ├─ 통과 → observation(결과 요약) + usageCount 증가
   └─ 실패 → 실패 fallback (4.1)
```

### 4.1 도구 실패 fallback (자가수정 폐루프)

실행 오류, timeout, 검증 실패를 모두 같은 폐루프로 흡수한다.

```text
실행/검증 실패
   │
   ▼
[실패 분류]
   ├─ 내장 도구 입력 오류  → observation: 어떤 입력이 왜 틀렸는지 → 다음 턴 LLM이 재호출
   ├─ 생성 도구 코드 오류  → observation: stderr·traceback 요약 → update_tool 유도 (7절)
   ├─ timeout            → observation: 시간 초과. 더 작은 범위나 효율적 접근 제안
   └─ 검증 실패          → observation: 무엇이 기대와 달랐는지(행 수·조건 등)
   │
   ▼
[재시도 상한 검사]  같은 도구 연속 실패가 maxFixRetries 초과면
   ├─ 사용자에게 상황 보고 후 ask_user 또는
   └─ finish(부분 결과 + 한계 설명)
   │
   ▼
다음 턴으로: LLM이 관찰을 보고 수정 행동을 고른다
```

원칙: 실패를 숨기지 않고 구체적 observation으로 바꿔 되먹인다. 무한 재시도를 막기 위해 도구별 연속 실패 카운터와 전체 반복 상한을 둔다.

## 5. Human-in-the-loop와 권한 처리

HIL은 두 경로로 들어온다. 하나는 LLM이 스스로 모호하다고 판단해 `ask_user`를 고르는 경우, 다른 하나는 런타임의 권한 게이트가 위험 행동을 막고 사용자에게 확인을 요구하는 경우다.

### 5.1 모호성 질문 (LLM 주도)

```python
async def ask_user_flow(action, context):
    answer = prompt_user(action.question, choices=action.choices)  # Rich 입력
    log(kind="ask_user")
    return Result(terminal=False, observation=f"사용자 답변: {answer}")
```

요청이 모호하면 도구를 만들거나 실행하기 전에 먼저 묻도록 system 규칙에 명시한다. 대상 데이터가 불명확하거나, 의도가 갈릴 수 있거나, 되돌리기 어려운 작업일 때가 해당한다.

### 5.2 권한 게이트 (런타임 주도)

모든 부수효과 행동은 실행 직전에 PolicyManager를 통과한다.

```python
def evaluate(action_id, ctx) -> PolicyDecision:
    # action_id 예: write_file, persist_tool, network_access, long_run, out_of_workspace
    if action_id in DENY_SET:    return PolicyDecision("DENY", action_id, reason)
    if action_id in ASK_SET:     return PolicyDecision("ASK_USER", action_id, reason)
    return PolicyDecision("ALLOW", action_id, reason)
```

```text
부수효과 행동
   │
   ▼
PolicyManager.evaluate
   ├─ ALLOW     → 즉시 실행
   ├─ DENY      → 실행 거부, observation으로 사유 전달, 루프 계속
   └─ ASK_USER  → 사용자에게 (y/n 또는 선택) 확인
                    ├─ 승인 → 실행
                    └─ 거부 → 실행 안 함, observation으로 사용자 거부 사실 전달
   │
   ▼
[로그]  policy 결정과 사유를 PolicyDecision으로 기록
```

기본 ASK_USER 대상: 작업 영역 내부 파일 쓰기, 도구 영속화, 네트워크 접근, 장시간 실행, 되돌리기 어려운 작업. 기본 DENY 대상: 경로 traversal이나 작업 영역 밖 읽기·쓰기·실행, 명시적으로 금지한 경로·명령. 설정으로 정책을 조정할 수 있지만, 경로 이탈 쓰기는 데모와 테스트에서 DENY 기준으로 고정한다.

원칙: 권한 결정은 LLM이 아니라 런타임 코드가 내린다. LLM은 행동을 제안할 뿐이고, 실행 여부의 최종 판단은 게이트가 갖는다.

## 6. create_tool 워크플로

생성 도구가 이 에이전트의 차별점이다. 흐름은 생성, 등록, 검증을 거쳐 선택적 영속화로 간다.

```text
create_tool(spec)
   │
   ▼
[중복 확인]  같은 목적 도구가 이미 있으면 재사용 안내 (재생성 대신 call_tool 유도)
   │
   ▼
[ToolSpec 검증]  name 규칙, inputSchema 유효성, code 안에 entrypoint 존재 확인
   │
   ▼
[즉시 등록]  세션 디렉터리에 tool.py를 쓰고 곧바로 레지스트리에 ToolDigest로 노출
   │           (trustedStatus = "session". 디스크 영속은 별도 권한 게이트 대상)
   ▼
[실사용]  같은 턴 또는 다음 턴에 call_tool로 실제 작업 수행 (4절)
   │
   ▼
[검증]  결과 상태 확인 (8절)
   │
   ▼
[영속화 제안]  작업 성공 후 또는 종료 직전
       "이 도구를 저장하면 다음 세션에서도 쓸 수 있습니다. 저장할까요? (y/n)"
       ├─ y → 권한 게이트(persist_tool) 통과 후 skills/<name>/ 에 SKILL.md·tool.py·manifest.json 기록
       │        trustedStatus = "persisted", version 부여
       └─ n → 세션 종료 시 임시 디렉터리째 폐기
```

생성 도구는 빈 입력 스모크 실행으로 게이트하지 않고 즉시 등록한다. 입력이 필요하거나 파일을 읽는 도구는 빈 입력 스모크에서 무조건 실패하기 때문이다. 대신 실제 호출에서 난 오류가 observation으로 되먹여져 update_tool을 유도한다(4.1·7절). `GeneratedToolManager.smoke_test`는 선택적 진단으로 남는다.

영속 저장 형태는 `schemas.md`의 ToolManifest를 따른다. 다음 세션 시작 시 manifest와 description만 읽어 ToolDigest로 등록하고, 코드와 전체 스키마는 호출 시점에 지연 로딩한다.

### 6.1 update_tool 워크플로 (7절 상세)

```text
update_tool(name, code, reason)
   │
   ▼
[대상 확인]  세션 도구 또는 영속 도구. 영속 도구 수정은 권한 게이트 대상
   │
   ▼
[코드 교체]  version += 1, updatedAt 갱신
   │
   ▼
[재실행]  ExecutionSandbox에서 다시 실행
   ├─ 성공 → 검증으로 진행
   └─ 실패 → observation(새 오류) → 재시도 상한 내에서 다시 update_tool
```

## 7. 검증 워크플로

검증은 실행 성공이 아니라 결과 상태 확인이다.

```python
def verify(task_intent, result, workspace) -> Verify:
    # 작업 유형별 검사 선택
    # - 파일 생성: 존재 + 내용 일부
    # - 데이터 변환: 행 수·스키마·정렬·필터 조건
    # - 상태 조작: 대상을 다시 읽어 기대 상태와 비교
    ...
    return Verify(passed=bool, reason=str)
```

검증 실패는 오류와 동급의 observation으로 4.1 fallback에 흘려보낸다. 가능한 한 도구 실행과 독립된 재조회로 확인해, "도구가 스스로를 통과시키는" 상황을 피한다.

## 8. 종료 조건

루프는 다음 중 하나에서 멈춘다.

- `finish` action 또는 `final`이 참인 `respond`
- 사용자 종료
- 최대 반복(maxIterations) 도달
- 도구별 또는 전체 연속 실패 상한 도달
- 프로토콜 복구 반복 실패

종료 후 세션 중 만든 생성 도구가 있으면 영속화를 제안한다. 어떤 경로로 끝나든 마지막에 작업 요약을 taskhistory에 남긴다.

## 9. 대표 시나리오 적용: 상태형 객체 트리

1. 사용자가 트리 조작을 자연어로 요청한다.
2. LLM이 `call_tool` searchDocs로 트리 스키마와 허용 연산을 조회한다(grounding).
3. LLM이 `create_tool`로 트리를 읽어 변형하는 도구를 만든다. 즉시 등록한다.
4. `call_tool`로 실제 트리를 변형한다. 권한 게이트(write_file)가 ASK_USER면 사용자 확인.
5. 트리를 다시 읽어 의도한 상태인지 검증한다.
6. 어긋나면 `update_tool`로 고쳐 다시 실행한다.
7. 성공하면 도구 저장 여부를 묻고, 승인 시 skill로 영속화한다.
