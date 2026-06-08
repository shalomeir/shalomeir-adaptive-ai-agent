# Object Tree World — 허용 연산

`world.json`을 다루는 연산은 다음으로 한정한다. 도구는 이 목록 안에서 조합해 작업을 수행한다.

| 연산 | 설명 |
| --- | --- |
| find_by_type(type) | 주어진 type의 모든 노드를 찾는다 |
| find_by_prop(key, op, value) | props의 key를 기준으로 비교(op: >=, <=, ==, >, <)해 노드를 찾는다 |
| get_prop(id, key) | 특정 노드의 속성값을 읽는다 |
| set_prop(id, key, value) | 특정 노드의 속성값을 바꾼다 |
| remove_node(id) | 노드와 그 자손을 제거한다 |
| add_child(parentId, node) | 부모 아래에 노드를 추가한다 |
| aggregate(type, key, fn) | 특정 type 노드의 key 값을 모아 fn(평균, 합, 개수, 최소, 최대)으로 집계한다 |

## 검증 규칙

연산 후에는 트리를 다시 읽어 의도한 상태인지 확인한다.

- 제거 작업이면 대상이 더 이상 존재하지 않는지 확인한다.
- 속성 변경이면 해당 노드의 값이 바뀌었는지 확인한다.
- 집계 작업이면 같은 조건으로 다시 계산해 값이 일치하는지 확인한다.
- 모든 작업 후 불변식(루트 type, id 유일성)이 유지되는지 확인한다.
