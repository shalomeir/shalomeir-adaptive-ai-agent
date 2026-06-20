# 스키마 선언 — 프로토콜과 도구 계약

상태: 지속 관리 문서. 프로토콜이나 도구 계약이 바뀌면 코드보다 먼저 이 문서를 갱신한다.

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
| --- | --- | --- |
| 0.1 | 2026-06-01 | 최초 작성. action 프로토콜, 내장 도구 입출력, ToolSpec, manifest, 로그 이벤트, 설정 스키마 정의 |
| 0.2 | 2026-06-08 | 전용 데이터 데모용 도구 제거. 일반 `runPython`/생성 도구 경로 유지 | API Model 지원 다변화

## 0. 규약

- 모든 스키마는 pydantic v2 모델로 구현하고, 외부 경계(LLM 출력, manifest 파일, 로그 파일)에서 검증한다.
- 아래 JSON Schema는 그 모델에서 생성되는 계약의 권위 있는 표현이다. 필드를 바꾸면 양쪽을 함께 고친다.
- 식별자 규칙: 도구 이름은 kebab-case, 필드 이름은 camelCase, action 이름은 snake_case.
- 시간은 ISO 8601 UTC 문자열.

## 1. Action 프로토콜

매 턴 LLM은 아래 하나의 객체를 반환한다. `action` 필드로 종류를 구분하는 discriminated union이다. JSON repair 후 이 스키마로 검증하며, 실패하면 검증 오류를 붙여 같은 형식으로 재요청한다.

### 1.1 공통 봉투

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgentAction",
  "type": "object",
  "required": ["action"],
  "properties": {
    "action": {
      "type": "string",
      "enum": ["respond", "ask_user", "call_tool", "create_tool", "update_tool", "finish"]
    }
  },
  "oneOf": [
    { "$ref": "#/$defs/respond" },
    { "$ref": "#/$defs/askUser" },
    { "$ref": "#/$defs/callTool" },
    { "$ref": "#/$defs/createTool" },
    { "$ref": "#/$defs/updateTool" },
    { "$ref": "#/$defs/finish" }
  ]
}
```

### 1.2 action 변형

```json
{
  "$defs": {
    "respond": {
      "type": "object",
      "required": ["action", "text"],
      "properties": {
        "action": { "const": "respond" },
        "text": { "type": "string" },
        "final": { "type": "boolean", "description": "생략하면 true처럼 최종 응답으로 처리한다. 중간 상태 메시지만 보낼 때 false를 명시한다." }
      },
      "additionalProperties": false
    },
    "askUser": {
      "type": "object",
      "required": ["action", "question"],
      "properties": {
        "action": { "const": "ask_user" },
        "question": { "type": "string" },
        "choices": { "type": "array", "items": { "type": "string" } },
        "reason": { "type": "string" }
      },
      "additionalProperties": false
    },
    "callTool": {
      "type": "object",
      "required": ["action", "name", "input"],
      "properties": {
        "action": { "const": "call_tool" },
        "name": { "type": "string" },
        "input": { "type": "object" }
      },
      "additionalProperties": false
    },
    "createTool": {
      "type": "object",
      "required": ["action", "spec"],
      "properties": {
        "action": { "const": "create_tool" },
        "spec": { "$ref": "#/$defs/toolSpec" }
      },
      "additionalProperties": false
    },
    "updateTool": {
      "type": "object",
      "required": ["action", "name", "code"],
      "properties": {
        "action": { "const": "update_tool" },
        "name": { "type": "string" },
        "code": { "type": "string" },
        "reason": { "type": "string" }
      },
      "additionalProperties": false
    },
    "finish": {
      "type": "object",
      "required": ["action"],
      "properties": {
        "action": { "const": "finish" },
        "summary": { "type": "string" }
      },
      "additionalProperties": false
    }
  }
}
```

### 1.3 pydantic 형태

```python
class Respond(BaseModel):
    action: Literal["respond"]
    text: str
    final: bool | None = None

class AskUser(BaseModel):
    action: Literal["ask_user"]
    question: str
    choices: list[str] | None = None
    reason: str | None = None

class CallTool(BaseModel):
    action: Literal["call_tool"]
    name: str
    input: dict[str, Any]

class CreateTool(BaseModel):
    action: Literal["create_tool"]
    spec: ToolSpec

class UpdateTool(BaseModel):
    action: Literal["update_tool"]
    name: str
    code: str
    reason: str | None = None

class Finish(BaseModel):
    action: Literal["finish"]
    summary: str | None = None

AgentAction = Annotated[
    Respond | AskUser | CallTool | CreateTool | UpdateTool | Finish,
    Field(discriminator="action"),
]
```

## 2. ToolSpec (공유 도구 정의)

도구 생성은 `create_tool` action으로 표현한다. `ToolSpec`은 생성 도구의 이름, 설명, 코드,
entrypoint, 입출력 스키마를 담는 공유 계약이다.

```json
{
  "title": "ToolSpec",
  "type": "object",
  "required": ["name", "description", "code", "inputSchema"],
  "properties": {
    "name": { "type": "string", "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$" },
    "description": { "type": "string", "minLength": 1 },
    "code": { "type": "string", "description": "entrypoint 함수를 포함한 Python 소스" },
    "entrypoint": { "type": "string", "default": "run", "description": "tool.py 안에서 호출할 함수 이름" },
    "inputSchema": { "type": "object", "description": "도구 입력의 JSON Schema" },
    "outputSchema": { "type": ["object", "null"], "description": "도구 출력의 JSON Schema. 없으면 자유 형식" }
  },
  "additionalProperties": false
}
```

도구 코드 규약: `tool.py`는 `entrypoint`로 지정한 함수를 정의한다. 함수는 `input` dict 하나를 받고 JSON 직렬화 가능한 값을 반환한다. 표준 출력은 로그로 수집되고 반환값이 결과로 쓰인다.

```python
def run(input: dict) -> dict:
    ...
    return {"...": "..."}
```

## 3. 내장 도구 입출력 스키마

모든 내장 도구는 `input` 객체를 받고 `output` 객체를 반환한다. 경로는 허용된 작업 영역 기준 상대 경로다.

### 3.1 readFile

```json
{ "input": { "type": "object", "required": ["path"],
    "properties": { "path": { "type": "string" },
      "maxBytes": { "type": "integer", "minimum": 1, "default": 1048576 } },
    "additionalProperties": false },
  "output": { "type": "object", "required": ["content", "bytes", "truncated"],
    "properties": { "content": { "type": "string" },
      "bytes": { "type": "integer" }, "truncated": { "type": "boolean" } } } }
```

### 3.2 writeFile

```json
{ "input": { "type": "object", "required": ["path", "content"],
    "properties": { "path": { "type": "string" }, "content": { "type": "string" },
      "mode": { "type": "string", "enum": ["overwrite", "append"], "default": "overwrite" } },
    "additionalProperties": false },
  "output": { "type": "object", "required": ["path", "bytesWritten"],
    "properties": { "path": { "type": "string" }, "bytesWritten": { "type": "integer" } } } }
```

### 3.3 listFiles

```json
{ "input": { "type": "object",
    "properties": { "path": { "type": "string", "default": "." },
      "recursive": { "type": "boolean", "default": false },
      "glob": { "type": "string" } },
    "additionalProperties": false },
  "output": { "type": "object", "required": ["entries"],
    "properties": { "entries": { "type": "array", "items": {
      "type": "object", "required": ["path", "type"],
      "properties": { "path": { "type": "string" },
        "type": { "type": "string", "enum": ["file", "dir"] },
        "size": { "type": "integer" } } } } } } }
```

### 3.4 runPython

```json
{ "input": { "type": "object",
    "properties": { "code": { "type": "string" }, "file": { "type": "string" },
      "args": { "type": "array", "items": { "type": "string" } },
      "stdin": { "type": "string" } },
    "oneOf": [ { "required": ["code"] }, { "required": ["file"] } ],
    "additionalProperties": false },
  "output": { "type": "object", "required": ["stdout", "stderr", "exitCode", "timedOut", "truncated"],
    "properties": { "stdout": { "type": "string" }, "stderr": { "type": "string" },
      "exitCode": { "type": "integer" }, "timedOut": { "type": "boolean" },
      "truncated": { "type": "boolean" } } } }
```

### 3.5 searchDocs

```json
{ "input": { "type": "object", "required": ["query"],
    "properties": { "query": { "type": "string" },
      "scope": { "type": "string", "description": "선택. 문서 묶음 식별자" },
      "limit": { "type": "integer", "minimum": 1, "default": 5 } },
    "additionalProperties": false },
  "output": { "type": "object", "required": ["results"],
    "properties": { "results": { "type": "array", "items": {
      "type": "object", "required": ["docId", "snippet"],
      "properties": { "docId": { "type": "string" }, "title": { "type": "string" },
        "snippet": { "type": "string" }, "score": { "type": "number" } } } } } } }
```

### 3.6 askUser

```json
{ "input": { "type": "object", "required": ["question"],
    "properties": { "question": { "type": "string" },
      "choices": { "type": "array", "items": { "type": "string" } },
      "reason": { "type": "string" } },
    "additionalProperties": false },
  "output": { "type": "object", "required": ["answer"],
    "properties": { "answer": { "type": "string" } } } }
```

## 4. 도구 manifest

저장된 skill의 `manifest.json` 스키마다.

```json
{
  "title": "ToolManifest",
  "type": "object",
  "required": ["name", "description", "inputSchema", "entrypoint",
               "runtime", "createdAt", "updatedAt", "usageCount", "trustedStatus", "version"],
  "properties": {
    "name": { "type": "string", "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$" },
    "description": { "type": "string" },
    "inputSchema": { "type": "object" },
    "outputSchema": { "type": ["object", "null"] },
    "entrypoint": { "type": "string", "default": "run" },
    "runtime": { "type": "string", "enum": ["python"], "default": "python" },
    "createdAt": { "type": "string", "format": "date-time" },
    "updatedAt": { "type": "string", "format": "date-time" },
    "usageCount": { "type": "integer", "minimum": 0, "default": 0 },
    "trustedStatus": { "type": "string", "enum": ["untrusted", "session", "persisted"] },
    "version": { "type": "integer", "minimum": 1, "default": 1 },
    "source": { "type": "string", "enum": ["generated", "mcp"], "default": "generated" }
  },
  "additionalProperties": false
}
```

레지스트리 노출용 축약 형태. 캐시 안정성을 위해 평소에는 이 형태만 LLM에 보인다.

```json
{ "title": "ToolDigest", "type": "object",
  "required": ["name", "description", "origin"],
  "properties": { "name": { "type": "string" }, "description": { "type": "string" },
    "origin": { "type": "string", "enum": ["builtin", "generated", "mcp"] },
    "inputSchema": { "type": ["object", "null"], "description": "선택. 프롬프트에는 compact field hint로 렌더링한다." } } }
```

## 5. 로그 이벤트

JSONL 한 줄이 한 이벤트다.

```json
{
  "title": "LogEvent",
  "type": "object",
  "required": ["ts", "traceId", "sessionId", "spanId", "kind"],
  "properties": {
    "ts": { "type": "string", "format": "date-time" },
    "traceId": { "type": "string" },
    "sessionId": { "type": "string" },
    "spanId": { "type": "string" },
    "parentSpanId": { "type": ["string", "null"] },
    "kind": { "type": "string",
      "enum": ["turn_start", "llm_call_start", "llm_call", "tool_call", "tool_create",
               "tool_update", "policy_decision", "error"] },
    "durationMs": { "type": ["number", "null"] },
    "model": { "type": ["string", "null"] },
    "inputTokens": { "type": ["integer", "null"] },
    "outputTokens": { "type": ["integer", "null"] },
    "cacheHit": { "type": ["boolean", "null"] },
    "responsePreview": { "type": ["string", "null"] },
    "responseChars": { "type": ["integer", "null"] },
    "responseTruncated": { "type": ["boolean", "null"] },
    "actionType": { "type": ["string", "null"] },
    "parseOk": { "type": ["boolean", "null"] },
    "retries": { "type": ["integer", "null"] },
    "toolName": { "type": ["string", "null"] },
    "toolOk": { "type": ["boolean", "null"] },
    "toolInput": { "type": ["string", "null"] },
    "toolInputChars": { "type": ["integer", "null"] },
    "toolInputTruncated": { "type": ["boolean", "null"] },
    "toolOutput": { "type": ["string", "null"] },
    "toolOutputChars": { "type": ["integer", "null"] },
    "toolOutputTruncated": { "type": ["boolean", "null"] },
    "toolError": { "type": ["string", "null"] },
    "toolErrorChars": { "type": ["integer", "null"] },
    "toolErrorTruncated": { "type": ["boolean", "null"] },
    "exitCode": { "type": ["integer", "null"] },
    "timedOut": { "type": ["boolean", "null"] },
    "outputBytes": { "type": ["integer", "null"] },
    "truncated": { "type": ["boolean", "null"] },
    "policy": { "type": ["string", "null"], "enum": ["ALLOW", "DENY", "ASK_USER", null] },
    "policyReason": { "type": ["string", "null"] },
    "verifyPassed": { "type": ["boolean", "null"] },
    "verifyReason": { "type": ["string", "null"] },
    "fixIteration": { "type": ["integer", "null"] },
    "errorKind": { "type": ["string", "null"] },
    "message": { "type": ["string", "null"] }
  },
  "additionalProperties": false
}
```

비밀 값과 민감 정보는 어떤 필드에도 넣지 않는다.

## 6. 권한 결정

```json
{ "title": "PolicyDecision", "type": "object",
  "required": ["decision", "action", "reason"],
  "properties": {
    "decision": { "type": "string", "enum": ["ALLOW", "DENY", "ASK_USER"] },
    "action": { "type": "string", "description": "대상 행동 식별자. 예: write_file, persist_tool, network_access" },
    "reason": { "type": "string" } },
  "additionalProperties": false }
```

기본 ASK_USER 대상: 작업 영역 내부 파일 쓰기, 도구 영속화, 네트워크 접근, 장시간 실행, 되돌리기 어려운 작업.

기본 DENY 대상: 경로 traversal이나 작업 영역 밖 읽기·쓰기·실행, 명시적으로 금지한 경로·명령. 경로 이탈 쓰기는 사용자 승인으로 우회하지 않는다.

## 7. 설정

환경 변수와 `.env`로 로드해 pydantic 모델로 검증한다.

```json
{
  "title": "AgentConfig",
  "type": "object",
  "properties": {
    "provider": { "type": "string", "default": "openai-compatible" },
    "baseUrl": { "type": "string", "description": "LLM 엔드포인트. 로컬 OpenAI 호환을 1순위로 안내" },
    "model": { "type": "string" },
    "apiKey": { "type": ["string", "null"], "description": "로컬 무키 실행 시 null 가능" },
    "maxIterations": { "type": "integer", "minimum": 1, "default": 20 },
    "maxFixRetries": { "type": "integer", "minimum": 0, "default": 3 },
    "toolTimeoutSec": { "type": "number", "minimum": 0.1, "default": 20 },
    "llmTimeoutSec": { "type": "number", "minimum": 0.1, "default": 60 },
    "maxOutputBytes": { "type": "integer", "minimum": 1, "default": 65536 },
    "compactionTokenThreshold": { "type": "integer", "minimum": 1, "default": 12000 },
    "workspaceDir": { "type": "string", "default": "./workspace" },
    "skillsDir": { "type": "string", "default": "./skills" },
    "logDir": { "type": "string", "default": "./logs" },
    "monitoring": { "type": "string", "enum": ["off", "langfuse"], "default": "off" },
    "networkDefault": { "type": "string", "enum": ["deny", "allow"], "default": "deny" }
  },
  "additionalProperties": false
}
```

## 8. LLM 메시지 형태

내부 메시지 모델은 provider 중립이며, `LLMClient`가 각 provider 규약으로 변환한다.

```json
{ "title": "Message", "type": "object",
  "required": ["role", "content"],
  "properties": {
    "role": { "type": "string", "enum": ["system", "user", "assistant", "tool"] },
    "content": { "type": "string" },
    "name": { "type": "string", "description": "tool 역할일 때 도구 이름" } },
  "additionalProperties": false }
```

도구는 평소 ToolDigest 목록으로만 프롬프트에 노출한다. ToolDigest에는 이름, 설명, 출처와
compact input field hint를 만들 수 있는 inputSchema만 포함한다. 특정 도구의 전체 코드는 그
도구를 호출하거나 수정하기로 한 시점에만 런타임에서 사용한다.
