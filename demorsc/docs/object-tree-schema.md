# Object Tree World — 스키마

`world.json`은 하나의 루트에서 뻗는 계층적 객체 트리다. 노드 하나는 다음 형태를 가진다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| id | string | 노드 고유 식별자 |
| type | string | 노드 종류. Scene, Container, Entity, Light, Camera, Sound 중 하나 |
| name | string | 사람이 읽는 이름 |
| props | object | 노드 속성. type에 따라 키가 다르다 |
| children | array | 자식 노드 목록 |

## type별 props

- Scene: 루트 한 개. props 비어 있음.
- Container: 그룹용. props 비어 있음. 자식을 가진다.
- Entity: 월드의 실체. props에 `health`(number)와 `solid`(boolean)를 가진다.
- Light: props에 `intensity`(0~1 number).
- Camera: props에 `fov`(number).
- Sound: props에 `volume`(0~1 number).

## 불변식

- 루트는 항상 type이 Scene이다.
- id는 트리 전체에서 유일하다.
- Entity의 health는 0 이상의 정수다.
- 노드를 제거하면 그 자손도 함께 제거된다.
