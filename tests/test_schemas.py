import pytest
from pydantic import ValidationError

from adaptive_agent.schemas import parse_agent_action, ToolManifest


def test_call_tool_action():
    a = parse_agent_action({"action": "call_tool", "name": "readFile", "input": {"path": "x.txt"}})
    assert a.action == "call_tool"
    assert a.name == "readFile"


def test_create_tool_action_with_spec():
    a = parse_agent_action(
        {
            "action": "create_tool",
            "spec": {
                "name": "normalize-csv",
                "description": "dedup+sort",
                "code": "def run(input):\n    return {}",
                "inputSchema": {"type": "object"},
            },
        }
    )
    assert a.spec.name == "normalize-csv"
    assert a.spec.entrypoint == "run"


def test_invalid_action_rejected():
    with pytest.raises(ValidationError):
        parse_agent_action({"action": "nope"})


def test_manifest_roundtrip():
    m = ToolManifest(
        name="t",
        description="d",
        inputSchema={"type": "object"},
        entrypoint="run",
        runtime="python",
        createdAt="2026-06-01T00:00:00Z",
        updatedAt="2026-06-01T00:00:00Z",
        usageCount=0,
        trustedStatus="persisted",
        version=1,
    )
    dumped = m.model_dump(by_alias=True)
    assert dumped["createdAt"] == "2026-06-01T00:00:00Z"
    assert ToolManifest.model_validate(dumped).name == "t"


def test_tool_name_normalized_to_kebab():
    # 약한 모델이 흔히 내는 camelCase/snake_case/공백 이름을 거부하지 않고 kebab으로 복구한다.
    from adaptive_agent.schemas import ToolSpec

    def mk(name: str) -> str:
        return ToolSpec(name=name, description="d", code="x", inputSchema={}).name

    assert mk("filterMonsters") == "filter-monsters"
    assert mk("filter_monsters") == "filter-monsters"
    assert mk("Monster Filter") == "monster-filter"
    assert mk("already-kebab") == "already-kebab"
