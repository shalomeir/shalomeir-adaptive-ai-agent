# 데모 리소스

이 폴더는 에이전트 데모와 통합 테스트가 쓰는 입력 데이터를 담는다. 데모 선언과 기대 결과는 `specs/demo_case.md`를 본다.

```text
demorsc/
  data/
    monsters.json     JSON 데이터 질의용
    events.csv        중복 제거와 날짜 정렬용 (중복 행 포함)
    events2.csv       저장된 도구 재사용 확인용 (다른 스키마의 CSV)
  world/
    world.json        상태를 가진 계층적 객체 트리
  docs/
    object-tree-schema.md      객체 트리 스키마 (searchDocs grounding 대상)
    object-tree-operations.md  객체 트리 허용 연산과 검증 규칙
```

## 사용 원칙

- 데모는 이 파일들을 입력으로 읽는다. 상태를 바꾸는 데모(객체 트리)는 원본을 직접 고치지 않도록, 실행 전 작업 영역으로 복사한 사본에 작용하는 것을 기본으로 한다.
- `docs/`는 `searchDocs`가 조회하는 로컬 문서 묶음이다. 에이전트는 작업 전에 여기서 스키마와 허용 연산을 근거로 확보한다.
- 모든 데이터는 일반적인 게임·씬 용어만 쓴다.
- 정책 거부와 다중 턴 컨텍스트 데모는 별도 입력 파일을 추가하지 않고 위 데이터를 재사용한다.
