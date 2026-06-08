from adaptive_agent.tools.generated import GeneratedToolManager
from adaptive_agent.sandbox import ExecutionSandbox
from adaptive_agent.schemas import ToolSpec


def make_spec(code: str) -> ToolSpec:
    return ToolSpec(name="adder", description="adds a and b",
                    code=code, inputSchema={"type": "object"})


def test_create_and_call(tmp_path):
    sandbox = ExecutionSandbox(tmp_path / "ws", timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=tmp_path / "session", sandbox=sandbox)
    spec = make_spec("def run(input):\n    return {'sum': input['a'] + input['b']}")
    tool = mgr.create(spec)
    res = tool.handler({"a": 2, "b": 3})
    assert res.ok
    assert res.output["sum"] == 5


def test_create_with_smoke_failure_reports_error(tmp_path):
    sandbox = ExecutionSandbox(tmp_path / "ws", timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=tmp_path / "session", sandbox=sandbox)
    spec = make_spec("def run(input):\n    raise ValueError('boom')")
    res = mgr.smoke_test(spec)
    assert not res.ok
    assert "boom" in res.error


def test_update_replaces_code(tmp_path):
    sandbox = ExecutionSandbox(tmp_path / "ws", timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=tmp_path / "session", sandbox=sandbox)
    mgr.create(make_spec("def run(input):\n    return {'sum': 0}"))
    tool = mgr.update("adder", "def run(input):\n    return {'sum': input['a'] + input['b']}")
    assert tool.handler({"a": 1, "b": 1}).output["sum"] == 2
