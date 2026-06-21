# 구현 요약

설계부터 완성까지의 과정을 한곳에 정리한 문서다. 무엇을 만들었고, 어떤 순서로 쌓았으며,
각 단계에서 무엇을 검증하고 어떤 문제를 고쳤는지를 담는다. 설계 근거는 `specs/`,
실행과 사용법은 `README.md`를 함께 본다.

## 1. 개요

자연어 작업을 받아 필요한 Python 도구를 직접 생성하고, 격리된 subprocess에서 실행한 뒤,
결과 상태를 다시 읽어 검증하는 CLI 에이전트다. 실행이 실패하면 오류를 관찰해 도구를 고치고
다시 돌린다. 사용자가 승인한 도구는 skill로 저장해 다음 세션에서 재사용한다. 제어 루프는
외부 에이전트 프레임워크 없이 손으로 구현했다.

## 2. 아키텍처

소스는 `src/adaptive_agent/` 아래 단일 책임 모듈로 나뉜다.

| 모듈 | 역할 |
| --- | --- |
| `config.py` | 환경 변수 기반 `AgentConfig` 로딩 |
| `schemas.py` | action 유니온, ToolSpec, ToolManifest, Message 등 pydantic 모델 |
| `parsing.py` | LLM 출력의 JSON 복구와 action 파싱, 실패 시 오류 피드백 |
| `observability.py` | 세션별 JSONL trace/span 로그, Rich 출력, bounded LLM·도구 입출력 preview, 이벤트를 익스포터로 전달 |
| `monitoring.py` | 외부 모니터링 익스포터 인터페이스와 no-op 기본 구현 |
| `sandbox.py` | subprocess 격리 실행(타임아웃, 작업 디렉터리 제한, 네트워크 정책, 출력 제한) |
| `policy.py` | ALLOW / DENY / ASK_USER 권한 게이트 |
| `tools/base.py`, `tools/registry.py` | 도구 추상화와 레지스트리(이름·설명 digest 노출) |
| `tools/builtins.py` | 내장 도구: 파일 읽기·쓰기·목록, 제한된 Python 실행, 문서 조회, 사용자 질문 |
| `tools/generated.py` | 생성 도구의 코드 기록과 sandbox 실행 |
| `skills.py` | 승인된 도구의 영속 저장과 재로딩 |
| `conversation.py`, `context.py` | 대화 누적과 compaction, 핵심 사실 carry-over |
| `llm.py` | OpenAI 호환·Anthropic 클라이언트와 테스트용 fake |
| `verify.py` | 결과 상태 검증 helper |
| `runner.py` | 핵심 ReAct 루프 |
| `cli.py` | Typer 진입점, 설정·도구·러너 배선 |


## 3. 핵심 동작

매 턴 LLM은 하나의 JSON action을 반환한다: `respond`, `ask_user`, `call_tool`,
`create_tool`, `update_tool`, `finish`. 러너는 이를 파싱·검증하고 분기 실행한 뒤 결과를
observation으로 누적한다. 약한 모델이나 로컬 모델에서 출력이 깨지면 복구하고, 그래도 안 되면
오류를 붙여 다시 묻는다. 생성 코드는 같은 프로세스에서 돌리지 않고 subprocess로 분리한다.
검증은 실행 성공 여부가 아니라 결과 상태를 다시 읽어 판단하며, 실패는 observation으로
되먹여져 `update_tool`을 부른다. 루프는 `finish`, 사용자 종료, 최대 반복, 연속 실패 상한,
그리고 같은 동작이 진전 없이 반복될 때(무진전)에서 멈춘다.

## 4. 단계별 구현 과정

테스트 우선(TDD)으로, 의존성이 없는 하위 모듈부터 쌓고 마지막에 러너와 CLI로 합쳤다. 각
단계는 구현 후 스펙 준수와 코드 품질을 독립적으로 검토하고, 지적 사항을 고친 뒤 커밋했다.

1. **스캐폴딩과 설정** — 패키지 골격, `pyproject.toml`, pytest·ruff 설정, `AgentConfig`.
   검토에서 기본값 중복(DRY)을 잡아 단일화하고 설정 테스트를 보강했다.
2. **스키마와 파싱** — action 유니온과 도구 계약 모델, JSON 복구와 파싱. 검토에서 테스트의
   죽은 import를 `# noqa`로 가리는 대신 제거했다.
3. **관측과 모니터링** — JSONL 트레이서와 익스포터 인터페이스. 불필요한 suppression과 쓰지
   않는 타이밍 scaffold를 제거했다.
4. **격리 실행과 권한** — subprocess sandbox와 권한 게이트. 검토에서 작업 영역 밖 접근을
   ASK_USER가 아니라 DENY로 바로잡고, 타임아웃 분기의 bytes/str 타입 버그를 고쳤다.
5. **도구와 내장 도구** — 레지스트리와 내장 도구 일습. 검토가 macOS 심볼릭 링크 작업 영역에서
   `listFiles`가 죽는 실제 버그를 잡아 해결 경로를 해결 후 경로 비교로 바꿨고, 쓰기 바이트 수
   계산과 남은 타입 부채를 함께 정리했다. 이후 레지스트리에 최소 입력 검증을 넣어 필수 필드가
   빠진 tool call은 핸들러 예외가 아니라 도구 오류 observation으로 되돌아가게 했다.
6. **생성 도구와 영속화** — 생성 도구 관리자와 skill 저장소. 검토가 신뢰할 수 없는 코드가
   가짜 결과 줄을 먼저 출력해 반환값을 위조할 수 있는 지점을 찾아, 결과 파싱을 마지막 줄
   기준으로 바꾸고 경로 이탈 가드를 더했다.
7. **대화와 컨텍스트** — 대화 저장과 설정 기반 compaction, carry-over.
8. **LLM 클라이언트와 검증** — OpenAI 호환·Anthropic 클라이언트, 결정적 테스트용 fake,
   검증 helper. 클라이언트는 provider별 payload 차이를 좁게 처리한다. capability를 아는
   hosted 모델에는 처음부터 호환되는 payload를 보내고, 로컬 OpenAI 호환 서버처럼 미리 알 수
   없는 경우에만 제한적으로 옵션을 낮춰 재시도한다. 이어서 트레이서가 같은 이벤트를
   익스포터로도 전달하도록 보강해, 나중에 외부 모니터링을 붙일 자리를 열어 두었다.
9. **러너** — ReAct 루프와 도구 생성·수정·영속화 연결. 검토를 받아 연속 실패 상한 종료를
   실제로 구현하고(설계의 종료 조건), 수정한 도구도 영속 제안 대상에 넣어 조용한 손실을 막고,
   생성·수정한 도구 이름을 carry-over로 보존했다. 이어서 `call_tool` 경로에 권한 게이트를
   더해, 쓰기는 사용자에게 묻고 작업 영역 밖 쓰기는 거부하도록 했다.
10. **CLI** — 설정·도구·러너를 묶는 Typer 진입점. 일반 clarification과 권한 확인은
   내부 prompt 문자열이 아니라 `agent:` / `you:` 대화 형태로 렌더링한다.
11. **데모 통합 테스트** — 아래 9개 시나리오를 fake 모델과 실제 sandbox 도구로 결정적으로 검증.
12. **문서와 마감** — README, 라이선스, 위생 검사 스크립트, 구현과 어긋난 워크플로 문서 동기화.

마지막 전체 검토에서 설정의 모니터링 모드가 트레이서에 연결되지 않은 점을 찾아, 익스포터를
러너 의존성으로 넘겨 배선을 닫았다. 이후 생성 도구 실행 cwd를 workspace로 고정해 프롬프트의
상대 경로 안내와 실제 실행을 일치시켰고, pytest가 현재 `src/`를 우선 보도록 설정했으며,
runtime 제한값과 로컬 경로를 환경 변수로 조정할 수 있게 했다.

## 5. 핵심 설계 결정

- 제어 루프를 직접 구현해 흐름을 코드에 드러낸다.
- JSON action 프로토콜을 1차 채널로 두고 복구·검증·재요청으로 약한 모델을 방어한다.
- 생성 도구는 메타데이터를 가진 skill 라이브러리로 저장하고, 프롬프트에는 이름과 설명만 상시
  노출한다. 전체 코드는 디스크에 두어 프롬프트 앞부분을 안정적으로 유지한다.
- 신뢰할 수 없는 코드는 workspace cwd의 subprocess로 격리하고, 세션 도구 코드는
  `workspace/.session/` 아래에 둔다.
- 권한 판단은 모델이 아니라 런타임이 내린다. 쓰기는 묻고, 작업 영역 밖은 거부한다.
- 컨텍스트는 누적, extractive compaction, 영속 skill의 세 층으로 둔다.
- 관측을 처음부터 내장하고, 외부 모니터링은 교체 가능한 익스포터로 분리한다.

## 6. 테스트

LLM 호출은 fake로 대체해 모든 테스트를 결정적으로 만들었다. 단위 테스트가 파서·정책·sandbox·
skill 라운드트립·러너 분기를 덮고, 데모 통합 테스트가 다음 시나리오를 끝에서 끝까지 확인한다.

- D1 JSON 데이터 질의
- D2 CSV 중복 제거·정렬과 도구 영속화
- D3 CSV 읽기 전용 그룹 집계와 생성 도구 자가수정
- D5 저장된 도구의 다음 세션 재사용
- D6 상태형 객체 트리 조작과 문서 근거 조회, 결과 상태 검증
- D7 작업 영역 밖 쓰기 거부
- D8 다중 턴과 권한 확인을 거친 쓰기
- D9 외부 패키지 사용·설치 질문 차단 후 표준 라이브러리 재시도

실제 CLI는 특정 데모 전용 내장 도구로 우회하지 않고, 생성 도구와 저장된 도구의 생성·수정·호출
루프로 파일 처리와 집계를 수행하는 쪽을 기본값으로 둔다. `runPython`은 `print(3+4)` 같은
한 줄 scalar 계산에만 쓰고, 파일을 읽어 줄 단위 확인·필터링·변환·집계·평균 계산을 해야 하면
생성 도구를 만들거나 재사용한다. 파일 저장은 직접 쓰기
우회를 감지해 권한 정책 경계로 올리고, 생성 도구가 `output`, `dst` 같은 출력 경로 입력을 받으면
작업 영역 밖 경로만 실행 전에 거부한다. 읽기 전용 요청에서 `store/write/save` 성격의 persisted
generated tool이 `name/path`와 `content` payload로 입력 파일을 덮어쓸 수 있으면 실행 전에
차단한다. 생성 도구 코드는 등록·수정 전에 `def run(input):` entrypoint를 정적으로 확인해,
module top-level에서 파일을 읽거나 쓰는 코드가 tool로 등록되지 않게 한다. 동일한 성공 tool call은
캐시해 재실행과 반복 권한 확인을 피하고, 같은 호출이 계속 반복되면 캐시된 결과를 최종 답변으로 접는다.
생성 도구가 workspace 파일 처리 요청에서 파일을 직접 읽지 않는 코드로 만들어지거나, `call_tool`
입력에 임의 샘플 데이터를 주입하려 하면 observation으로 되돌린다. 도구 코드는 실제 workspace
파일을 `open`, `json.load`, `csv.reader` 등으로 읽어야 한다.
workspace 파일이 명시된 요청에서 이전 generated tool을 재사용하려면, 그 파일이 call input에
들어가 있거나 tool code가 직접 그 파일을 읽어야 한다. 파일 연결만 맞아도 작업 의도(필터·집계·정렬·
저장 등)가 현재 요청과 충분히 겹치지 않으면 이전 도구 호출을 observation으로 되돌린다. read-only
계산 요청에서 generated tool이 파일 경로만 반환하면 완료로 보지 않고, 요청한 값을 반환하도록 새
도구 생성이나 수정을 유도한다.
파일 출력 결과는 러너가 작업 영역 파일을 다시 읽어 일반 sanity check로 검증한다. 도구가 반환한
출력 경로나 `writeFile` payload를 기준으로 JSON 파싱 가능 여부, CSV header/행 존재 여부,
텍스트 파일의 비어 있지 않음을 확인한다. 요청에 출력 파일명이 명시되어 있으면 도구가 반환한
출력 경로도 그 파일명과 일치해야 한다. CSV 정렬 규칙이나 object tree 상태 같은 데모별 semantic
검증은 러너에 박지 않고, 실패한 도구는 실행 오류와 observation을 통해 수정 루프로 되돌린다.

약한 모델이 현재 요청에 명시된 파일의 구조를 사용자에게 묻는 경우는 런타임 observation으로 되먹여
다시 계획하게 한다. 반대로 현재 요청에 없는 파일명을 이전 문맥에서 끌어온 질문은 파일 읽기 루프로
강제하지 않고 사용자 확인 흐름으로 둔다. 요청 source 파일이 실제로 작업 영역에 없을 때는 사용자 확인을
허용하지만, 이 판단은 "없다" 같은 문구가 아니라 `readFile`로 확인한 파일 존재 여부를 기준으로 한다.
JSON dict에 `root.children` 같은 중첩 트리 구조가 보이면
파일 힌트로 재귀 순회를 유도하고, 후속 요청은 대화 컨텍스트와 이전 소스 파일을 활용하도록 planner
prompt에서 안내한다. 반복·파싱 실패처럼 완료 신호 없이 끝난 경우에도 내부 observation을 그대로
사용자에게 내보내지 않고, 사용자용 종료 메시지로 정리한다.
모델이 `respond(final:true)`나 `ask_user`로 사용자 요청을 그대로 되풀이하는 경우도 완료나
질문으로 인정하지 않고, 파일을 직접 열어 실제 계산·변환을 수행하라는 observation을 넣어
다시 계획하게 한다. 파일 결과가 필요한 턴에서 생성 도구를 만들거나 수정했다면, 그 뒤 성공한
작업 도구 호출이 있어야만 `respond(final:true)`나 `finish`를 종료로 인정한다. 이 검사는
응답 문장 표현이 아니라 현재 턴의 실행 상태를 기준으로 한다. 또한 "write/update 하지 말고"
같은 명시적 금지 문구가 있으면 "제거" 같은 논리적 필터링 단어가 있어도 파일 쓰기 의도로 보지
않는다.
차단된 `ask_user` observation이 반복되면 같은 문제를 max iteration까지 끌고 가지 않고 no-progress로
조기 종료한다. `ask_user`가 `question` 대신 `text`를 쓴 경우는 parser가 일반 alias로 복구한다.
파일이 실제로 없다는 질문은 구조 확인 질문으로 막지 않고 사용자 확인 흐름으로 둔다.
같은 이름의 `create_tool`이 반복되면 새 도구를 계속 덮어쓰지 않고, 이미 생성된 도구를
`call_tool`로 실행하라는 구체적인 payload 힌트를 observation에 넣는다. 단, 같은 이름이라도 새
코드가 기존 코드와 다르면 모델이 수정 의도를 `create_tool`로 잘못 표현한 것으로 보고 `update_tool`
경로로 처리한다.
이미 호출된 generated tool을 `update_tool`로 수정하면, 러너는 이전 입력으로 수정된 도구를
즉시 실행해 실제 실패 또는 검증 결과를 observation으로 만든다. 같은 update만 반복하며
실행을 미루는 loop를 LLM 설득만으로 풀지 않기 위해서다.
generated tool 실행 자체가 실패한 경우도 같은 도구 재호출을 허용하지 않고 update 또는 파일 존재
확인으로 돌아가게 한다.
사용자가 도구 생성을 명시한 요청은 `create_tool`만으로 끝낼 수 없고, 방금 만든 generated tool의
성공한 `call_tool`이 있어야 종료할 수 있다.
`respond`의 `final` 필드는 생략하면 최종 응답으로 처리하고, 중간 상태 메시지만 보낼 때
`final:false`를 명시하게 했다. 약한 모델이 짧은 대화 응답에서 `final`을 자주 생략하기 때문이다.
CSV 정리, JSON 변형, markdown 표 생성 같은 파일 작업은 러너가 직접 수행하지 않고 생성 도구와
`writeFile` 호출의 observation 루프로 처리하며, `runPython`은 한 줄 scalar 계산 외에는 쓰지
않는다. 현재 요청과 맞지 않는 generated tool 호출은 이전 턴의 도구 실행을 이어가지 않도록
observation으로 차단한다. 현재 턴에 도구 실행 근거가 없는데 결과형 최종 답변이 이전 assistant
결과처럼 보이면, 이전 결과를 재사용하지 말고 현재 요청에 맞는 도구 실행이나 사용자 확인으로
돌아가게 한다.
generated tool 재사용은 요청 파일명이 `call_tool` 입력에 있다는 이유만으로 허용하지 않고,
도구 이름·설명의 작업 의도와 현재 요청의 작업 의도가 충분히 겹쳐야 한다. 단, 의도 키워드가 거의
없는 범용 도구는 파일 연결을 기준으로 재사용할 수 있다. 이전에 만든 generated tool이 `input['path']`,
`input.get('src')`처럼 파일 경로 입력을 기대하는데 모델이 빈 input으로 호출하면, 런타임이 현재 요청의
workspace 파일 경로를 해당 입력 필드에 보정한 뒤 실행한다. inline 샘플 데이터 주입은 이 보정보다 먼저
차단해, 임의 데이터가 실제 파일 처리로 둔갑하지 않게 한다.
LLM 요청 payload는 프로토콜 system prompt와 도구 목록을 하나의 system 메시지로 합쳐 보낸다.
여러 system 메시지로 나누면 약한 로컬 모델이 도구 목록이나 직전 결과 JSON에 끌려 action 규약을
잊는 경향이 있어, strict output contract가 항상 최상위 지시로 남도록 했다.
각 LLM 호출에는 누적 대화와 별도로 `[runtime-state]` JSON 메시지를 합성해 붙인다. 여기에는
현재 요청, 이번 턴 action index, 생성·수정된 도구, 호출된 generated tool, 마지막 tool input,
검증 실패, missing workspace path, 차단된 tool action, 최근 observation이 들어간다. 이 상태판은
대화 히스토리에 저장하지 않고 호출 시점에만 만들기 때문에, 모델이 현재 루프 상태를 보면서 판단하되
컨텍스트가 불필요하게 누적되지는 않는다. `readFile` 같은 context tool의 캐시 결과는 작업 완료로
접지 않고, 이미 읽은 정보를 근거로 실제 작업 도구를 만들거나 호출하도록 루프를 계속한다.
`call_tool`에서 모델이 `output_path` 같은 인자를 `input` 밖 top-level에 둔 경우는 parser가 `input`
으로 병합해 도구 호출에서 사라지지 않게 한다.
`create_tool`에서 모델이 `inputSchema` 대신 `inputs: ["path"]`처럼 간략한 입력 목록을 낸 경우도
JSON Schema 형태로 복구한 뒤 엄격 검증한다.

데모 입력은 `demorsc/`에, 시나리오 명세는 `specs/demo_case.md`에 있다. 상태를 바꾸는 데모는
원본을 보존하기 위해 작업 영역으로 복사한 사본에 작용한다.

## 7. 품질과 출처 위생

모든 단계에서 `ruff`, `mypy src`, `pytest`를 통과시켰다. 공개 저장소에는 작업 배경이나
출처를 드러내는 표현을 두지 않는다. 금지어 목록은 저장소 밖에서 관리하고
`scripts/hygiene_check.sh`가 `HYGIENE_PATTERNS` 경로를 받아 검사한다.

## 8. 한계와 향후 작업

- sandbox는 subprocess, timeout, 출력 제한, workspace 내부 스크립트 실행 제한을 제공한다.
  macOS에서는 `sandbox-exec`가 있으면 네트워크 기본 차단도 적용한다. 더 강한 파일시스템
  격리는 컨테이너나 OS sandbox 정책으로 보강해야 한다.
- 생성 도구 품질은 모델과 검증 루프에 의존한다. 스모크 결과는 참고용이고, 실제 호출 오류가
  수정을 이끈다.
- 런타임 설정은 `.env`와 환경 변수로 조정할 수 있다.
- 복잡한 작업을 task list로 분해해 위임하는 동적 워크플로는 설계만 두고 향후 작업으로 남겼다.
