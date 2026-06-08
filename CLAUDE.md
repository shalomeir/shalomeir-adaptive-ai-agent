# 프로젝트 작업 지침

이 저장소에서 작업할 때 따른다. `AGENTS.md`는 이 파일의 symlink다.

## 현재 상태

핵심 구현은 완료된 상태다. `src/adaptive_agent/`에 전 모듈이 있고 테스트(`tests/`)·타입·린트가 통과하며, 로컬 모델과 OpenAI 호환 엔드포인트 양쪽에서 동작을 확인했다. 이 폴더가 공개 산출물이며, 여기서 정리·보완을 이어간다.

## 시작 시 읽을 것

작업 전에 다음을 먼저 읽어 현재 설계와 구현을 복원한다.

- `docs/IMPLEMENTATION.md` — 구현 단계별 요약
- `docs/CODE_GUIDE.md` — 코드 흐름 읽기 가이드
- `specs/prd.md`, `specs/trd.md`, `specs/schemas.md`, `specs/agent-workflows.md`, `specs/demo_case.md`, `specs/project-brief.md` — 확정 설계와 데모 명세

## 무엇을 만드는가

자연어 작업을 받아 필요한 Python 도구를 스스로 생성·격리 실행·자가수정하고, 사용자가 승인하면 재사용 가능한 skill로 영속화하는 CLI 에이전트. 에이전트 추상화 라이브러리(LangChain, 벤더 SDK 등) 없이 제어 루프를 직접 구현한다.

## 코딩 규약

- Python 식별자는 snake_case(PEP 8). ruff가 강제한다.
- JSON 와이어 포맷(action, manifest, config 직렬화)은 `specs/schemas.md`대로 camelCase. pydantic alias로 매핑한다.
- 도구 이름은 kebab-case.
- 테스트는 TDD로: 실패 테스트 → 최소 구현 → 통과 → 커밋. LLM 호출은 fake로 대체해 결정적으로 검증한다.

## 명령

- 설치: `pip install -e ".[dev]"`
- 테스트: `pytest -v`
- 타입: `mypy src`
- 린트: `ruff check src tests`

## 작성 원칙

저장소의 모든 이름, 문서, 주석, 커밋 메시지는 제품·회사 중립으로 유지한다. 특정 제품이나 회사명을 넣지 않고, 문제와 예제는 일반 용어로만 기술한다. 데모 환경은 "object tree world" 같은 일반 명칭만 쓴다.
