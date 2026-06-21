# Adaptive Tool-Building CLI Agent

자연어 작업을 받아 **필요한 Python 도구를 직접 만들고**, 격리된 subprocess에서 실행한 뒤,
결과 상태를 다시 읽어 확인하는 CLI 에이전트다. 실행이 실패하면 오류를 관찰해 코드를 고치고
다시 돌린다. 사용자가 승인한 도구는 저장해 다음 세션에서 재사용한다. 제어 루프는 외부 에이전트
프레임워크 없이 직접 구현했다.

- 설계 문서: `specs/`
- 코드 읽기 가이드: `docs/CODE_GUIDE.md`
- 구현 과정 요약: `docs/IMPLEMENTATION.md`

---

## 빠른 시작

Python 3.11 이상이 필요하다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

adaptive-agent version   # 0.1.1 이 찍히면 설치 성공
```

---

## 직접 대화하며 시연하기

기본 실행 방식은 키 없이 로컬 [Ollama](https://ollama.com)의 `qwen2.5-coder:7b` 모델을
사용하는 것이다. OpenAI API, Claude API, Vercel AI Gateway, OpenRouter API 키를 쓰는
방식도 지원한다.

### 1) Model 설정

#### a. 로컬 Ollama 기본 설정

```bash
cp .env.example .env
ollama pull qwen2.5-coder:7b
```

기본값 그대로 쓴다면 사실상 설정은 여기서 끝난다. `.env.example`을 복사한 `.env`에는 이미
Ollama의 OpenAI 호환 엔드포인트와 `qwen2.5-coder:7b` 모델명이 들어 있다.

다른 Ollama 로컬 모델을 쓰고 싶다면 `.env`의 `AGENT_MODEL`만 바꾼다.

#### b. API key 설정

OpenAI API, Claude API, Vercel AI Gateway, OpenRouter를 쓰고 싶다면 `.env` 하단의 해당
provider 블록을 수정한다. OpenAI, Claude, Vercel AI Gateway, OpenRouter 블록 중 하나에서
"uncomment the four lines below" 안내가 붙은 네 줄을 주석 해제하고, `AGENT_API_KEY`와
`AGENT_MODEL`을 본인 계정에 맞게 바꾼다.

### 2) 작업 영역에 데모 데이터 넣기

에이전트의 파일 도구는 작업 영역(`./workspace`) 안에서만 동작한다. 데모 입력을 그 안으로
복사한다.

```bash
mkdir -p workspace
cp demorsc/data/monsters.json  workspace/
cp demorsc/data/events.csv     workspace/
cp demorsc/data/events2.csv    workspace/
cp demorsc/world/world.json    workspace/
```

### 3) 세션 시작

```bash
adaptive-agent chat
```

`you:` 프롬프트가 뜨면 아래 데모 문장을 입력한다. 작업 중 파일 쓰기나 도구 저장처럼 되돌리기
어려운 단계에서는 에이전트가 `(y/n)`으로 확인을 묻는다. `exit`, `/exit`, `quit`, `/quit`으로 종료한다.

### 4) 데모별 입력 문장과 기대 동작

- **D1 · 데이터 질의**
  입력: `workspace의 monsters.json에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘.`
  기대: Orc, Dragon, Wolf와 평균 186.67. 읽기만 하므로 승인 없음.

- **D2 · CSV 정리와 도구 저장**
  입력: `events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 events-clean.csv로 저장해줘.`
  기대: 고유 5행, 날짜순. 파일 쓰기에서 `(y/n)` 확인을 묻는다.

- **D3 · 실패와 자가수정**
  입력: `events.csv에서 완전히 중복된 행은 한 번만 세고, amount 합계를 type별로 구해줘.`
  기대: 최종 합계는 purchase 2500, signup 0, refund -200. 읽기 전용 집계라 파일 쓰기
  확인을 묻지 않는다.

- **D4 · 모호한 요청**
  입력: `데이터 좀 정리해줘.`
  기대: 에이전트가 바로 실행하지 않고 어떤 데이터를 어떻게 정리할지 되묻는다. 이어서
  `events.csv에서 중복 제거하고 date로 정렬해줘.`로 답하면 진행한다.

- **D5 · 저장한 도구 재사용** (D2를 먼저 한 다음, 세션을 새로 시작)
  입력: `events2.csv도 똑같이 중복 제거하고 date로 정렬해서 events2-clean.csv로 저장해줘.`
  기대: 새 도구를 만들지 않고 D2에서 저장한 도구를 다시 불러 쓴다. 결과는 고유 3행(a, b, c).

- **D6 · 상태형 객체 트리**
  입력: `world.json에서 health가 100 미만인 Entity를 제외하고, 남은 Entity의 평균 health를 알려줘.`
  기대: 에이전트가 먼저 트리 스키마와 허용 연산을 문서에서 조회하고, 트리를 고친 뒤 다시 읽어
  검증한다. 남는 Entity는 셋, 평균 health 190.

- **D7 · 작업 영역 밖 쓰기 차단**
  입력: `events.csv를 정렬해서 ../events-sorted.csv에 저장해줘.`
  기대: 작업 영역 밖 경로라 정책이 거부한다. 파일이 만들어지지 않고 거부 사유를 알려준다.

- **D8 · 다중 턴 컨텍스트**
  입력 1: D1과 동일. 입력 2: `방금 필터된 결과를 hp 내림차순 마크다운 표로 table.md에 저장해줘.`
  기대: 이전 턴의 결과를 이어받아 표를 만들고, 파일 쓰기에서 `(y/n)` 확인 후 저장한다.

- **D9 · 외부 패키지 회피**
  입력: D2와 동일.
  기대: 모델이 pandas 같은 외부 패키지나 설치 질문으로 빠져도 사용자에게 묻지 않고, 표준
  라이브러리 `csv` 기반 도/구로 고쳐 저장한다.

> 로컬 모델의 실제 결과는 모델 성능에 따라 달라질 수 있다. 정확한 기대값을 결정적으로 보고
> 싶다면 위 통합 테스트를 사용한다.

---

## CLI 사용법

명령은 셋이다.

```bash
adaptive-agent version        # 버전 출력
adaptive-agent chat           # 대화형 세션 시작
adaptive-agent chat --no-loading   # 처리 중 loading 표시 없이 시작
adaptive-agent chat --docs-dir demorsc/docs   # 근거 조회용 문서 폴더 지정(기본값)

# 작업 한 건을 비대화형으로 실행(스크립트·CI·반복 테스트용)
adaptive-agent run "monsters.json에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘."
adaptive-agent run "events.csv에서 중복을 지우고 date로 정렬해 events-clean.csv로 저장해줘." --yes
adaptive-agent run "..." --max-iterations 30   # 반복 상한 override
```

`run`은 한 번 실행하고 결과를 출력한 뒤 종료한다. 비대화형이라 `(y/n)` 확인을 직접 받을 수
없으므로, 파일 쓰기·도구 저장 같은 부수효과를 진행하려면 `--yes`로 모두 자동 승인한다.
`--yes` 없이는 안전하게 거절한다. 자유 형식 되묻기(ask_user)는 비대화형에서 답할 수 없어
한계가 있다.

대화형 `chat` 세션 동작은 이렇다.

- 시작할 때 현재 CLI 버전, 적용된 모델명, 실행 디렉터리, workspace 경로를 보여준다.
- 이어서 `세션을 시작합니다. 'exit' 또는 '/exit'로 종료.` 안내를 보여준다.
- `you:` 프롬프트에 자연어로 요청을 적는다.
- 요청을 받은 뒤 처리 중에는 `agent: loading.`, `loading..`, `loading...`을 순환 표시한다.
  이 표시는 답변이나 사용자 확인 prompt 전에 지워지고, 답을 입력하면 같은 턴의 남은 처리 동안
  다시 표시된다. `--no-loading`으로 끌 수 있다.
- 인사, 모델명 질문, 짧은 잡담도 같은 JSON action 루프를 거쳐 답한다.
- 요청이 모호하면 에이전트가 되묻는다. 답을 입력하면 이어 진행한다.
- 파일 쓰기, 도구 저장처럼 부수효과가 있는 단계에서 `(y/n)`을 묻는다. `y`로 승인, `n`으로 거절.
- 작업이 끝나면 `agent:` 줄에 요약을 보여준다.
- 작업이 실패 한도에 걸려 중단되면 raw traceback 대신 마지막 실패 원인을 짧게 보여준다.
- `exit`, `/exit`, `quit`, `/quit`으로 종료한다.

세션이 남기는 것들:

| 위치 | 내용 |
| --- | --- |
| `./workspace/` | 에이전트가 읽고 쓰는 작업 영역(파일 도구는 이 안에서만 동작) |
| `./workspace/.session/` | 세션 중 만든 생성 도구의 코드 |
| `./skills/<이름>/` | 저장을 승인한 도구(`tool.py`, `manifest.json`, `SKILL.md`) |
| `./logs/session-*.jsonl` | 세션별 JSONL 실행 로그(LLM 호출, 도구 실행, 권한 결정, 검증) |

실행 흐름을 들여다보려면 로그를 본다.

```bash
ls -t logs/session-*.jsonl | head -1
cat "$(ls -t logs/session-*.jsonl | head -1)" | python -m json.tool   # 또는 jq
```

`tool_call` 이벤트에는 실행한 도구 이름, 성공 여부, 입력 preview, 출력 또는 오류 preview가 함께
남는다. 큰 값은 4000자로 잘리고 전체 글자 수와 잘림 여부가 같이 기록된다. `apiKey`,
`token`, `password` 같은 민감 필드명은 로그에 쓰기 전에 마스킹된다.
LLM이 `call_tool`을 선택한 시점의 `llm_call` 이벤트에도 도구 이름과 입력 preview가 남고,
실제 실행 결과는 이어지는 `tool_call` 이벤트에서 확인한다.

저장된 도구를 확인하려면 skill 폴더를 본다.

```bash
ls skills/
cat skills/*/SKILL.md
```

---

## LLM 없이 데모 9종 바로 확인하기

가장 빠르고 확실한 검증 경로다. 통합 테스트가 가짜 LLM으로 에이전트 루프를 그대로 돌리고,
실제 생성 도구를 sandbox에서 실행해 결과 상태까지 검증한다. 모델 설치 없이 즉시 돌아간다.

```bash
pytest tests/test_demo_integration.py -v
```

전체 테스트와 품질 점검은 이렇게 돌린다.

```bash
pytest -q
mypy src
ruff check src tests
ruff format --check .
```

---

## 설정

복사해서 시작할 `.env.example`이 있다. 기본값은 로컬 Ollama이며, OpenAI 호환 Chat
Completions 엔드포인트라면 `AGENT_BASE_URL`, `AGENT_MODEL`, `AGENT_API_KEY` 조합으로 바꿔
쓸 수 있다. Vercel AI Gateway는 `https://ai-gateway.vercel.sh/v1`, OpenRouter는
`https://openrouter.ai/api/v1` base URL을 사용한다. `AGENT_PROVIDER=anthropic`이면 같은
설정값으로 Anthropic native Messages API를 호출한다. Anthropic의 Opus 4.8 모델 ID는
`claude-opus-4-8`처럼 hyphen 표기를 쓴다. Anthropic native 호출은 Opus 계열의 sampling
파라미터 제한을 피하기 위해 `temperature`를 처음부터 보내지 않는다.

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `AGENT_PROVIDER` | `openai-compatible` | provider 표식. 실제 호출 방식은 base URL과 model 조합이 정한다 |
| `AGENT_BASE_URL` | `http://localhost:11434/v1` | chat completions 엔드포인트 |
| `AGENT_MODEL` | `qwen2.5-coder:7b` | 모델 이름 |
| `AGENT_API_KEY` | (없음) | 호스팅 provider에서만 필요 |
| `AGENT_MONITORING` | `off` | `off`, 또는 `langfuse`로 외부 모니터 연결 |
| `AGENT_MAX_ITERATIONS` | `20` | 한 작업에서 허용하는 최대 루프 횟수 |
| `AGENT_MAX_FIX_RETRIES` | `3` | 파싱 실패·도구 연속 실패·동일 동작 무진전 반복 허용 횟수 |
| `AGENT_TOOL_TIMEOUT_SEC` | `20` | 생성 도구와 runPython 실행 타임아웃 |
| `AGENT_LLM_TIMEOUT_SEC` | `60` | LLM HTTP 호출 타임아웃 |
| `AGENT_MAX_OUTPUT_BYTES` | `65536` | 도구 stdout/stderr 보관 상한 |
| `AGENT_COMPACTION_TOKEN_THRESHOLD` | `12000` | 대화 compaction 임계값 |
| `AGENT_WORKSPACE_DIR` | `./workspace` | 파일 도구와 생성 도구 실행 작업 영역 |
| `AGENT_SKILLS_DIR` | `./skills` | 승인된 도구 저장 위치 |
| `AGENT_LOG_DIR` | `./logs` | JSONL 실행 로그 위치 |
| `AGENT_NETWORK_DEFAULT` | `deny` | sandbox 네트워크 기본 정책 |

---

## 설계 결정 (요약)

- 프레임워크 없이 `AgentRunner` 중심의 제어 루프를 직접 구현했다. 매 턴 LLM이 하나의 JSON
  action을 내고, runner가 파싱·검증·실행·observation 누적을 반복하는 ReAct 루프를 실행한다.
- JSON action 프로토콜을 1차 프로토콜로 두고, code fence/trailing comma/direct tool alias 같은
  흔한 출력은 parsing 과정에서 복구를 지원하고 이후 pydantic schema로 검증한다.
- loop guard는 동일 action 반복, 동일 tool call 재실행, 부수효과 도구 반복을 감지해
  no-progress 루프와 중복 실행을 끊는다.
- 도구는 `ToolRegistry`에 내장 도구와 생성 도구를 같은 인터페이스로 등록하고, 프롬프트에는
  tool compact digest를 매회 다시 만들어 prompt system message에 붙여서 보낸다.
- 생성 도구는 메타데이터를 가진 skill 라이브러리로 저장하고, 프롬프트에는 이름과 설명만 올린다.
- 신뢰할 수 없는 코드는 workspace cwd의 subprocess로 격리하고, 세션 도구 코드는
  `workspace/.session/` 아래에 둔다. `runPython`과 생성 도구 실행은 sandbox를 통해 timeout,
  출력 제한, workspace 경로 제한을 적용한다.
- 권한 판단은 모델이 아닌 런타임으로 강제한다. 쓰기는 묻고, 작업 영역 밖은 거부한다.
- 컨텍스트는 누적, 요약 compaction, 영속 skill 세 층으로 둔다.

자세한 근거는 `docs/IMPLEMENTATION.md`와 `specs/`에 있다.

## 한계와 향후 작업

- sandbox는 subprocess, timeout, 출력 제한, workspace 내부 스크립트 실행 제한을 제공한다.
  macOS에서는 `sandbox-exec`가 있으면 `AGENT_NETWORK_DEFAULT=deny`일 때 네트워크도 차단한다.
  더 강한 파일시스템 격리는 컨테이너나 OS sandbox 정책으로 보강해야 한다.
- 생성 도구 품질은 모델과 검증 루프에 의존되며 일부 low 모델에서 잘못된 결과도 검증이 통과되어 잘못된
  결과를 제공하기도 한다.
- 한 번에 한 세션만 실행한다. 동시 병렬 실행이나 세션 사이의 skill 충돌·버전 관리는 다루지 않고 있다.
  같은 이름으로 도구를 다시 저장하면 이전 것을 덮어쓴다.
- 긴 작업을 미리 여러 단계로 쪼개고, 정해 둔 순서대로 실행하는 구조는 없다. 지금은 매 턴 모델이 다음
  action을 고르는 ReAct 루프로 구현되어 있다. 단계가 많은 작업에서는 모델이 중간에 다른 길로
  새기 쉽고, 복구 루프에 의존하게 된다. 모델 성능에 따라 복구 루프 턴 횟수를 초과하고 중단되기도 하며
  그래서 긴 작업은 복구를 포함한 여러 사전 단계를 미리 plan 하고 task step 으로 관리하여 의도하지 않은
  방향으로 새는 일이 없도록 개선이 필요하다.
- 툴 결과 테스트 검증 로직이 불완전해서 가끔 툴이 잘못된 결과를 만들어 내도 패스할 수 있다. 검증 로직을 툴
  생성후 제대로 검증하도록 강제해야 한다.
- 캐시를 고려하지 않아 skill inventory 가 바뀌면 system message 가 바뀌어 캐시가 쉽게 무효화 될 수 있다.
  캐시를 고려해서 고정 builtin tool 외에는 분리하고 listTools 등 필요시 도구 조회 전용 tool 도 추가로 제공한다.

## 라이선스

MIT. `LICENSE` 참고.
