from adaptive_agent.tools.base import Tool, ToolResult
from adaptive_agent.tools.registry import ToolRegistry


def make_echo() -> Tool:
    return Tool(
        name="echo",
        description="echo input",
        origin="builtin",
        input_schema={"type": "object"},
        handler=lambda inp: ToolResult(ok=True, output={"echo": inp}),
    )


def test_register_and_digests():
    reg = ToolRegistry()
    reg.register(make_echo())
    digests = reg.digests()
    assert digests[0].name == "echo"
    assert digests[0].origin == "builtin"


def test_call_runs_handler():
    reg = ToolRegistry()
    reg.register(make_echo())
    res = reg.call("echo", {"a": 1})
    assert res.ok
    assert res.output == {"echo": {"a": 1}}


def test_missing_tool_returns_error():
    reg = ToolRegistry()
    res = reg.call("nope", {})
    assert not res.ok
    assert "nope" in res.error


def test_call_is_tool_name_format_insensitive():
    # 모델이 등록명과 다른 형식(camelCase)으로 불러도 정규화 비교로 도구를 찾는다.
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="filter-monsters",
            description="d",
            origin="generated",
            input_schema={"type": "object"},
            handler=lambda i: ToolResult(ok=True, output="hit"),
        )
    )
    res = reg.call("filterMonsters", {})
    assert res.ok and res.output == "hit"


def test_call_validates_required_fields_before_handler():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="writeFile",
            description="write",
            origin="builtin",
            input_schema={"type": "object", "required": ["path", "content"]},
            handler=lambda i: ToolResult(ok=True, output="should not run"),
        )
    )

    res = reg.call("writeFile", {"path": "out.txt"})

    assert not res.ok
    assert "필수 필드" in (res.error or "")
    assert "content" in (res.error or "")


def test_generated_tool_properties_are_required_when_required_is_omitted():
    called = False

    def handler(inp):
        nonlocal called
        called = True
        return ToolResult(ok=True, output="should not run")

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="csv-dedupe-sort",
            description="dedupe sort",
            origin="generated",
            input_schema={
                "type": "object",
                "properties": {"source": {"type": "string"}, "output": {"type": "string"}},
            },
            handler=handler,
        )
    )

    res = reg.call("csv-dedupe-sort", {"inputPath": "events.csv", "outputPath": "out.csv"})

    assert not res.ok
    assert called is False
    assert "source" in (res.error or "")
    assert "output" in (res.error or "")
    assert "inputPath" not in (res.error or "")


def test_call_validates_property_types_before_handler():
    called = False

    def handler(inp):
        nonlocal called
        called = True
        return ToolResult(ok=True, output="should not run")

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="document-repository",
            description="Stores documents",
            origin="generated",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
            handler=handler,
        )
    )

    res = reg.call("document-repository", {"name": "monsters.json", "content": None})

    assert not res.ok
    assert called is False
    assert "content" in (res.error or "")
    assert "string" in (res.error or "")
