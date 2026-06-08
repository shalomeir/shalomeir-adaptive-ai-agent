from adaptive_agent.tools.base import Tool, ToolResult
from adaptive_agent.tools.registry import ToolRegistry


def make_echo() -> Tool:
    return Tool(
        name="echo", description="echo input", origin="builtin",
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
