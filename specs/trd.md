# TRD — Adaptive Tool-Building CLI Agent

상태: 지속 관리 문서. 기술 결정이 바뀌면 이 문서를 먼저 갱신한다.

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
| --- | --- | --- |
| 0.1 | 2026-06-01 | 최초 작성. 언어, 의존성 정책, 핵심 라이브러리, MCP·비동기·관측 방침 정의 |
| 0.2 | 2026-06-08 | 현재 구현에 맞춰 sandbox 실행 경계와 환경 변수 설정 범위를 갱신 |

## 1. 문서 관리 원칙

이 문서는 코드와 함께 계속 관리한다. 라이브러리를 추가하거나 빼고, 버전을 올리고, 기술 방향을 바꿀 때는 코드보다 먼저 이 문서를 고친다. 각 의존성은 왜 필요한지와 어떤 대안을 제쳤는지를 함께 적는다. 변경은 위 이력 표에 한 줄로 남긴다.

## 2. 기술적 목표

- 제어 루프를 읽기 쉬운 순수 Python으로 구현한다. 에이전트 프레임워크에 기대지 않는다.
- 의존성을 최소로 둔다. 핵심 동작은 표준 라이브러리와 소수의 검증된 패키지로 충분해야 한다.
- LLM 접근을 provider에 묶지 않는다. 키 없이 로컬에서도 돌릴 수 있어야 한다.
- LLM 호출을 인터페이스 뒤에 두어 결정적 테스트가 가능해야 한다.
- 모델 입력의 앞부분을 안정적으로 유지해 캐시 적중을 높인다.
- 신뢰할 수 없는 생성 코드를 격리해 실행한다.
- 관측을 처음부터 내장한다. 로컬 로그는 항상 켜지고, 외부 모니터링은 선택으로 붙인다.
- 외부 연동(모니터링, MCP 등)은 모두 선택이며, 꺼도 에이전트가 정상 동작한다.

## 3. 언어와 런타임

- 언어: Python 3.11 이상.
- 동시성: 핵심 루프는 동기 순차 실행이다. LLM timeout 감시와 CLI loading 표시는 표준
  `threading`을 사용한다.
- 패키징: `pyproject.toml` 기반. 개발 환경은 `uv`로 잡는 것을 권장한다.

## 4. 의존성 정책

순수 구현을 우선한다. 라이브러리는 다음 기준을 모두 만족할 때만 추가한다.

- 직접 구현하면 분명히 더 약하거나 위험해지는 부분일 것.
- 에이전트 루프나 도구 오케스트레이션을 대신해 주는 추상화가 아닐 것.
- 핵심 경로의 필수 의존성과 선택 의존성을 분리할 수 있을 것.

선택 의존성은 `pyproject.toml`의 extras로 분리한다. 설치하지 않아도 핵심 기능은 돌아간다.

## 5. 핵심 라이브러리

### 5.1 필수 (core)

| 라이브러리 | 용도 | 선택 이유 / 대안 |
| --- | --- | --- |
| typer | CLI 프레임 | 타입 힌트 기반으로 명령과 인자를 간결하게 정의. 대안 argparse는 보일러플레이트가 많다 |
| rich | 사람이 읽는 진행 출력 | 패널, 테이블, 스트리밍 로그, progress. CLI 가독성의 핵심 |
| pydantic v2 | 데이터 모델과 검증 | action, 도구 manifest, 설정, 로그 이벤트를 모델로 정의하고 검증. JSON schema도 여기서 뽑는다 |
| httpx | LLM HTTP 호출 | OpenAI 호환 엔드포인트를 동기 `POST`로 직접 호출한다. 특정 벤더 SDK에 묶이지 않아 provider 비종속과 순수 구현 목표에 맞는다 |
| python-dotenv | `.env` 로딩 | CLI 진입점에서 로컬 `.env`를 읽어 환경 변수 기반 설정을 쉽게 시작할 수 있게 한다 |

표준 라이브러리로 처리하는 부분: `subprocess`, `json`, `pathlib`, `logging`, `dataclasses` 보조.

벤더 SDK(openai, anthropic 등)는 쓰지 않는다. 금지 제약 때문이 아니라, httpx로 OpenAI 호환 규약만 맞추면 로컬 모델과 호스팅을 한 경로로 다룰 수 있어 더 순수하고 이식성이 좋기 때문이다.

### 5.2 선택 (optional extras)

| 라이브러리 | extra | 용도 |
| --- | --- | --- |
| mcp | `mcp` | 향후 MCP 도구 연동을 위한 선택 의존성. 현재 MVP 핵심 경로에는 연결하지 않는다 |
| langfuse | `monitoring` | 외부 관측 익스포터. trace, span, generation, score로 내보낸다 |
| structlog | 기본 포함 검토 | 구조화 로그를 JSONL로 남긴다. 표준 logging + 커스텀 JSON formatter로 대체 가능 |
| pydantic-settings | 기본 포함 검토 | 현재는 `os.environ`과 pydantic 모델로 충분하다. 설정이 복잡해질 때 검토한다 |

### 5.3 개발 (dev)

| 라이브러리 | 용도 |
| --- | --- |
| pytest, pytest-asyncio | 단위·통합 테스트. 현재 핵심 테스트는 동기 경로 중심이며, async 플러그인은 개발 의존성으로 남아 있다 |
| ruff | 린트와 포맷 |
| mypy 또는 pyright | 정적 타입 검사 |

## 6. 실행 방침

핵심 제어 루프는 순차적으로 읽히는 동기 코드로 둔다.

- LLM 호출은 `httpx.post`를 사용하고, runner는 별도 thread에서 호출을 감시해 설정된 timeout을 넘으면 중단한다.
- 생성 도구와 `runPython`은 `subprocess.run`으로 실행하고 timeout, cwd, 환경 변수 allowlist, 출력 제한을 적용한다.
- 관측 익스포터 실패는 삼켜 핵심 경로를 막지 않는다.
- 향후 스트리밍 출력, 병렬 도구 실행, 작업 목록 위임이 필요해지면 그때 `asyncio` 도입을 재검토한다.

비동기를 미리 깔지 않는 이유는 runner의 action dispatch와 실패 복구 흐름을 코드에서 바로 읽히게 하기 위해서다.

## 7. MCP 통합

MCP는 도구를 주고받는 프로토콜로만 쓴다. 제어 루프를 대신하지 않는다. 현재 구현의 핵심
루프는 MCP 없이 완결하며, `mcp` extra는 향후 확장용으로만 둔다.

두 방향을 둔다.

- 도구 제공자로서: 내장 도구와 저장된 skill을 MCP 서버로 노출해, 외부 클라이언트가 이 에이전트의 도구를 쓸 수 있게 한다.
- 도구 소비자로서: 외부 MCP 서버를 마운트해 그 서버의 도구를 레지스트리에 추가 도구로 등록한다.

레지스트리는 도구의 출처(내장, 생성, MCP)를 구분할 수 있는 필드를 갖지만, 현재 테스트된
경로는 내장 도구와 생성 도구다. MCP 소비자/제공자 연결은 stretch로 둔다.

## 8. 데이터 모델 (pydantic 적용 지점)

- LLM action: respond, ask_user, call_tool, create_tool, update_tool, finish를 모델로 정의하고 검증한다. JSON repair 후 이 모델로 파싱해 실패하면 오류를 만들어 재요청한다.
- 도구 manifest: name, description, inputSchema, outputSchema, createdAt, updatedAt, usageCount, trustedStatus.
- 설정: provider, base_url, model, 제한값(timeout, 최대 반복, 출력 크기), 로그 경로, 모니터링 on/off.
- 로그 이벤트: 9절의 필드를 모델로 정의해 직렬화한다.

입력 스키마 검증과 JSON schema 생성을 pydantic 한 곳으로 모아, 약한 모델 출력 방어와 도구 계약을 같은 도구로 처리한다.

## 9. 실행 격리

- 생성 코드는 subprocess로 분리해 실행한다. 같은 프로세스의 직접 실행은 쓰지 않는다.
- 생성 도구와 `runPython`은 workspace를 cwd로 실행한다. 실행할 스크립트 경로는 workspace 안으로 제한하고, 세션 도구 코드는 `workspace/.session/` 아래에 둔다.
- timeout, 환경 변수 allowlist, 표준 출력·오류 크기 제한, 쓰기 경로 정책, 네트워크 기본 차단을 둔다.
- macOS에서는 `sandbox-exec`가 있으면 `networkDefault=deny`일 때 subprocess 네트워크 접근을 차단한다. 다른 환경에서는 subprocess 격리와 timeout·출력 제한이 기본선이다.
- 더 강한 파일시스템 격리와 CPU·메모리 자원 제한(컨테이너, OS sandbox, cgroup 등)은 운영 환경 옵션으로 문서화하되 MVP 필수는 아니다. 한계는 README에 명시한다.

## 10. 로그와 관측

- 로컬 구조화 로그는 항상 켜진다. 한 줄이 한 이벤트인 JSONL로 남긴다. 한 요청을 trace로, 그 안의 LLM 호출과 도구 실행을 span으로 묶는다.
- 화면 출력은 rich로 사람이 읽기 좋게 보여준다.
- 외부 모니터링은 하나의 익스포터 인터페이스 뒤에 둔다. 기본은 no-op이고, 환경 변수로 켜면 Langfuse로 내보낸다. trace, span, generation, score 개념에 로컬 모델을 대응시킨다.
- 익스포터는 비차단으로 동작하고, 실패해도 핵심 경로를 막지 않는다.
- 도구 실행 로그는 도구 이름, 성공 여부, 입력 preview, 출력 또는 오류 preview를 남긴다.
  큰 값은 잘라 저장하고 전체 글자 수와 잘림 여부를 함께 기록한다.
- 민감 정보와 비밀 값은 로그에 남기지 않는다. `apiKey`, `token`, `password` 같은 민감
  필드명은 preview 직렬화 전에 마스킹한다.

세부 필드와 튜닝 활용은 PRD 7절을 따른다.

## 11. 설정 관리

- 환경 변수와 `.env`로 설정한다. `.env.example`을 제공하고 키 값은 저장소에 두지 않는다.
- 설정은 pydantic 모델로 로드해 타입과 기본값, 제한값을 한곳에서 관리한다.
- LLM provider는 base_url과 model 조합으로 선택한다. 로컬 OpenAI 호환 엔드포인트를 1순위로 안내한다.

## 12. 품질 기준

- 정적 타입 검사를 통과시킨다.
- 핵심 모듈은 단위 테스트로 덮고, 통합 테스트로 대표 시나리오를 확인한다.
- LLM 호출은 fake 구현으로 대체해 결정적으로 테스트한다.
- 린트와 포맷을 ruff로 강제한다.

## 13. 추가로 고려할 만한 것 (제안)

확정 전 검토 대상이다. 채택하면 위 표와 이력에 반영한다.

- json-repair: 약한 모델의 깨진 JSON 복구. 작고 목적이 분명하다. 다만 의존성을 줄이려면 최소 복구 로직을 직접 구현하는 선택도 있다. 우선 직접 구현하고, 한계가 보이면 도입을 검토한다.
- tenacity: 재시도와 백오프. 편하지만 흐름을 가린다. 재시도 상한이 단순하므로 직접 구현을 우선한다.
- prompt_toolkit: 대화형 입력 경험 개선. typer 기본 입력으로 충분하면 도입하지 않는다.
- jsonschema: pydantic이 스키마 생성과 검증을 함께 하므로 별도 도입은 보류한다.
- platformdirs: 로그와 skill 저장 위치를 OS 규약에 맞춰 잡고 싶을 때. 초기에는 프로젝트 상대 경로로 충분하다.
- tiktoken 등 토크나이저: 캐시·compaction 임계 판단을 위한 토큰 추정. provider 비종속을 해치므로, 우선 문자 길이 근사로 시작하고 필요 시 도입한다.

## 14. 의존성 요약

- 필수: typer, rich, pydantic, httpx, python-dotenv
- 선택: mcp(`mcp`), langfuse(`monitoring`), structlog, pydantic-settings
- 개발: pytest, pytest-asyncio, ruff, mypy 또는 pyright
- 표준 라이브러리 중심: subprocess, threading, json, pathlib, dataclasses
