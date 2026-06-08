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

adaptive-agent version   # 0.1.0 이 찍히면 설치 성공
```

---

## 직접 대화하며 시연하기

OpenAI API 키를 쓰는 호스팅 모델 방식과, 키 없이 로컬 [Ollama](https://ollama.com)로 돌리는
방식 둘 다 가능하다. 처음 시연은 OpenAI 키 방식이 가장 간단하다.

### 1) 모델 준비

```bash
# 방법 A: OpenAI 호스팅 모델 사용
cp .env.example .env
# .env 아래쪽의 OpenAI 예시 블록을 주석 해제하고 AGENT_API_KEY만 본인 키로 바꾼다.
# AGENT_BASE_URL=https://api.openai.com/v1
# AGENT_MODEL=gpt-4o-mini

# 방법 B: Ollama 로컬 모델 사용
ollama pull qwen2.5-coder:7b
```

### 2) 에이전트가 바라볼 엔드포인트 설정

OpenAI를 쓸 때는 `.env.example`에서 복사한 `.env`의 OpenAI 설정을 그대로 쓰면 된다.

```bash
# Ollama를 쓸 때만 아래처럼 로컬 OpenAI 호환 엔드포인트로 바꾼다
export AGENT_BASE_URL=http://localhost:11434/v1
export AGENT_MODEL=qwen2.5-coder:7b
unset AGENT_API_KEY
```

### 3) 작업 영역에 데모 데이터 넣기

에이전트의 파일 도구는 작업 영역(`./workspace`) 안에서만 동작한다. 데모 입력을 그 안으로
복사한다.

```bash
mkdir -p workspace
cp demorsc/data/monsters.json  workspace/
cp demorsc/data/events.csv     workspace/
cp demorsc/data/events2.csv    workspace/
cp demorsc/world/world.json    workspace/
```

### 4) 세션 시작

```bash
adaptive-agent chat
```

`you:` 프롬프트가 뜨면 아래 데모 문장을 입력한다. 작업 중 파일 쓰기나 도구 저장처럼 되돌리기
어려운 단계에서는 에이전트가 `(y/n)`으로 확인을 묻는다. `exit` 또는 `quit`으로 종료한다.

### 5) 데모별 입력 문장과 기대 동작1

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
  라이브러리 `csv` 기반 도구로 고쳐 저장한다.

> 로컬 모델의 실제 결과는 모델 성능에 따라 달라질 수 있다. 정확한 기대값을 결정적으로 보고
> 싶다면 위 통합 테스트를 사용한다.

---

## CLI 사용법

명령은 셋이다.

```bash
adaptive-agent version        # 버전 출력
adaptive-agent chat           # 대화형 세션 시작
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

- `you:` 프롬프트에 자연어로 요청을 적는다.
- 인사, 모델명 질문, 짧은 잡담은 도구 계획 없이 바로 답한다.
- 요청이 모호하면 에이전트가 되묻는다. 답을 입력하면 이어 진행한다.
- 파일 쓰기, 도구 저장처럼 부수효과가 있는 단계에서 `(y/n)`을 묻는다. `y`로 승인, `n`으로 거절.
- 작업이 끝나면 `agent:` 줄에 요약을 보여준다.
- `exit` 또는 `quit`으로 종료한다.

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

복사해서 시작할 `.env.example`이 있다. 환경 변수로 바꾼다.

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `AGENT_PROVIDER` | `openai-compatible` | provider 표식 |
| `AGENT_BASE_URL` | `http://localhost:11434/v1` | chat completions 엔드포인트 |
| `AGENT_MODEL` | `qwen2.5-coder:7b` | 모델 이름 |
| `AGENT_API_KEY` | (없음) | 호스팅 provider에서만 필요 |
| `AGENT_MONITORING` | `off` | `off`, 또는 `langfuse`로 외부 모니터 연결 |
| `AGENT_MAX_ITERATIONS` | `20` | 한 작업에서 허용하는 최대 루프 횟수 |
| `AGENT_MAX_FIX_RETRIES` | `3` | 파싱 실패·도구 연속 실패·동일 동작 무진전 반복 허용 횟수 |
| `AGENT_TOOL_TIMEOUT_SEC` | `20` | 생성 도구와 runPython 실행 타임아웃 |
| `AGENT_LLM_TIMEOUT_SEC` | `180` | LLM HTTP 호출 타임아웃 |
| `AGENT_MAX_OUTPUT_BYTES` | `65536` | 도구 stdout/stderr 보관 상한 |
| `AGENT_COMPACTION_TOKEN_THRESHOLD` | `12000` | 대화 compaction 임계값 |
| `AGENT_WORKSPACE_DIR` | `./workspace` | 파일 도구와 생성 도구 실행 작업 영역 |
| `AGENT_SKILLS_DIR` | `./skills` | 승인된 도구 저장 위치 |
| `AGENT_LOG_DIR` | `./logs` | JSONL 실행 로그 위치 |
| `AGENT_NETWORK_DEFAULT` | `deny` | sandbox 네트워크 기본 정책 |

---

## 설계 결정 (요약)

- 프레임워크 없이 제어 루프를 직접 구현해 흐름을 코드에 드러낸다.
- JSON action 프로토콜을 1차 채널로 두고, 복구·검증·재요청으로 약한 모델을 방어한다.
- 생성 도구는 메타데이터를 가진 skill 라이브러리로 저장하고, 프롬프트에는 이름과 설명만 올린다.
- 신뢰할 수 없는 코드는 workspace cwd의 subprocess로 격리하고, 세션 도구 코드는
  `workspace/.session/` 아래에 둔다.
- 권한 판단은 모델이 아니라 런타임이 내린다. 쓰기는 묻고, 작업 영역 밖은 거부한다.
- 컨텍스트는 누적, 요약 compaction, 영속 skill의 세 층으로 둔다.

자세한 근거는 `docs/IMPLEMENTATION.md`와 `specs/`에 있다.

## 한계와 향후 작업

- sandbox는 subprocess, timeout, 출력 제한, workspace 내부 스크립트 실행 제한을 제공한다.
  macOS에서는 `sandbox-exec`가 있으면 `AGENT_NETWORK_DEFAULT=deny`일 때 네트워크도 차단한다.
  더 강한 파일시스템 격리는 컨테이너나 OS sandbox 정책으로 보강해야 한다.
- 생성 도구 품질은 모델과 검증 루프에 의존한다.
- 한 번에 한 세션만 가정한다. 동시 실행이나 세션 사이의 skill 충돌·버전 관리는 다루지 않는다.
  같은 이름으로 도구를 다시 저장하면 이전 것을 덮어쓴다.
- 긴 작업을 안정적으로 끌고 가는 계획-실행 분리가 없다. 지금은 매 턴 모델이 다음 한 수를
  정하는 ReAct 루프라, 단계를 미리 계획으로 고정한 뒤 그대로 실행하거나 하위 작업으로 나눠
  위임하는 결정적 오케스트레이션은 향후 작업으로 남겼다. 단계가 많은 작업일수록 중간 이탈을
  복구로 메우는 비용이 커진다.

## 라이선스

MIT. `LICENSE` 참고.
