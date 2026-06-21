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
    assert res.action.final is None


def test_parse_direct_tool_action_as_call_tool():
    res = parse_action_text('{"action":"runPython","input":{"code":"print(1)"}}')
    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "runPython"
    assert res.action.input == {"code": "print(1)"}


def test_parse_direct_tool_action_uses_top_level_fields_as_input():
    res = parse_action_text(
        '{"action":"writeFile","path":"world.md","content":"graph TD\\nscene --> ground\\n"}'
    )
    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "writeFile"
    assert res.action.input == {"path": "world.md", "content": "graph TD\nscene --> ground\n"}


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


def test_parse_call_tool_top_level_extra_fields_into_input():
    res = parse_action_text(
        '{"action":"call_tool","name":"csv-cleaner","input":{"path":"events2.csv"},'
        '"output_path":"events2-clean.csv"}'
    )
    assert res.ok
    assert res.action.action == "call_tool"
    assert res.action.name == "csv-cleaner"
    assert res.action.input == {"path": "events2.csv", "output_path": "events2-clean.csv"}


def test_parse_respond_summary_alias():
    res = parse_action_text('{"action":"respond","summary":"done","final":true}')
    assert res.ok
    assert res.action.action == "respond"
    assert res.action.text == "done"
    assert res.action.final is True


def test_parse_ask_user_text_alias():
    res = parse_action_text('{"action":"ask_user","text":"어떤 파일을 정리할까요?","final":false}')
    assert res.ok
    assert res.action.action == "ask_user"
    assert res.action.question == "어떤 파일을 정리할까요?"


def test_parse_create_tool_inputs_alias_as_input_schema():
    res = parse_action_text(
        '{"action":"create_tool","spec":{"name":"clean-csv","description":"clean",'
        '"code":"def run(input):\\n    return input[\\"path\\"]","inputs":["path"]}}'
    )

    assert res.ok
    assert res.action.action == "create_tool"
    assert res.action.spec.input_schema == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def test_parse_bare_json_without_action_returns_protocol_error():
    res = parse_action_text('{"path":"events-clean.csv","rows":5,"removed":2}')
    assert not res.ok
    assert res.error
    assert "action 필드가 없습니다" in res.error
    assert '{"action":"respond","text":"...","final":true}' in res.error


def test_parse_action_fail_returns_error():
    res = parse_action_text("not json at all")
    assert not res.ok
    assert res.error
