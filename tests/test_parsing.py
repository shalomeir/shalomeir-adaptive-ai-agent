from adaptive_agent.parsing import json_repair, parse_action_text


def test_repair_trailing_text_and_fence():
    raw = 'Here:\n```json\n{"action":"finish","summary":"done",}\n```\nthanks'
    data = json_repair(raw)
    assert data["action"] == "finish"


def test_parse_action_ok():
    res = parse_action_text('{"action":"respond","text":"hi"}')
    assert res.ok
    assert res.action.action == "respond"


def test_parse_action_fail_returns_error():
    res = parse_action_text("not json at all")
    assert not res.ok
    assert res.error
