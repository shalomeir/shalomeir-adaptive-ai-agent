from adaptive_agent.tools.generated import GeneratedToolManager
from adaptive_agent.sandbox import ExecutionSandbox
from adaptive_agent.schemas import ToolSpec


def make_spec(code: str) -> ToolSpec:
    return ToolSpec(
        name="adder", description="adds a and b", code=code, inputSchema={"type": "object"}
    )


def test_create_and_call(tmp_path):
    ws = tmp_path / "ws"
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=ws / ".session", sandbox=sandbox)
    spec = make_spec("def run(input):\n    return {'sum': input['a'] + input['b']}")
    tool = mgr.create(spec)
    res = tool.handler({"a": 2, "b": 3})
    assert res.ok
    assert res.output["sum"] == 5


def test_create_with_smoke_failure_reports_error(tmp_path):
    ws = tmp_path / "ws"
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=ws / ".session", sandbox=sandbox)
    spec = make_spec("def run(input):\n    raise ValueError('boom')")
    res = mgr.smoke_test(spec)
    assert not res.ok
    assert "boom" in res.error


def test_update_replaces_code(tmp_path):
    ws = tmp_path / "ws"
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=ws / ".session", sandbox=sandbox)
    mgr.create(make_spec("def run(input):\n    return {'sum': 0}"))
    tool = mgr.update("adder", "def run(input):\n    return {'sum': input['a'] + input['b']}")
    assert tool.handler({"a": 1, "b": 1}).output["sum"] == 2


def test_generated_tool_runs_with_workspace_cwd(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "data.txt").write_text("hello", encoding="utf-8")
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=ws / ".session", sandbox=sandbox)
    tool = mgr.create(make_spec("def run(input):\n    return {'content': open('data.txt').read()}"))

    res = tool.handler({})

    assert res.ok
    assert res.output == {"content": "hello"}


def test_generated_tool_rejects_session_outside_workspace(tmp_path):
    sandbox = ExecutionSandbox(tmp_path / "ws", timeout_sec=5, max_output_bytes=4096)
    mgr = GeneratedToolManager(session_dir=tmp_path / "session", sandbox=sandbox)
    tool = mgr.create(make_spec("def run(input):\n    return {'ok': True}"))

    res = tool.handler({})

    assert not res.ok
    assert "workspace 안" in res.error
