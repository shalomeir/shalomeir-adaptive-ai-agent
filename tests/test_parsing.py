from adaptive_agent.parsing import json_repair, parse_action_text


def test_repair_trailing_text_and_fence():
    raw = 'Here:\n```json\n{"action":"finish","summary":"done",}\n```\nthanks'
    data = json_repair(raw)
    assert data["action"] == "finish"


def test_repair_uses_first_balanced_json_object():
    raw = 'Here {"action":"finish","summary":"done"} trailing } text'
    data = json_repair(raw)
    assert data == {"action": "finish", "summary": "done"}


def test_parse_action_ok():
    res = parse_action_text('{"action":"respond","text":"hi"}')
    assert res.ok
    assert res.action.action == "respond"


def test_parse_direct_tool_action_as_call_tool():
    res = parse_action_text('{"action":"runPython","input":{"code":"print(1)"}}')
    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "runPython"
    assert res.action.input == {"code": "print(1)"}


def test_parse_triple_quoted_code_field():
    raw = '''{
      "action": "runPython",
      "input": {
        "code": """
print("hello")
"""
      }
    }'''

    res = parse_action_text(raw)

    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "runPython"
    assert res.action.input["code"].strip() == 'print("hello")'


def test_parse_camel_case_call_tool_alias():
    res = parse_action_text('{"action":"callTool","name":"readFile","input":{"path":"a.txt"}}')
    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "readFile"


def test_parse_respond_summary_alias():
    res = parse_action_text('{"action":"respond","summary":"done","final":true}')
    assert res.ok
    assert res.action.action == "respond"
    assert res.action.text == "done"
    assert res.action.final is True


def test_parse_action_fail_returns_error():
    res = parse_action_text("not json at all")
    assert not res.ok
    assert res.error
