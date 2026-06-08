from adaptive_agent.runner import AgentRunner, RunnerDeps
from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.tools.registry import ToolRegistry
from adaptive_agent.tools.base import Tool, ToolResult


def build_runner(tmp_path, replies, ask="n"):
    reg = ToolRegistry()
    reg.register(Tool("echo", "echo", "builtin", {"type": "object"},
                      lambda inp: ToolResult(ok=True, output=inp)))
    return AgentRunner(RunnerDeps(
        llm=FakeLLMClient(replies=replies),
        registry=reg,
        ask=lambda *a: ask,
        log_dir=tmp_path,
        max_iterations=10,
        max_fix_retries=2,
    ))


def test_respond_then_finish(tmp_path):
    runner = build_runner(tmp_path, [
        '{"action":"respond","text":"working"}',
        '{"action":"finish","summary":"done"}',
    ])
    result = runner.run_turn("hello")
    assert "done" in result.summary


def test_call_tool_observation(tmp_path):
    runner = build_runner(tmp_path, [
        '{"action":"call_tool","name":"echo","input":{"x":1}}',
        '{"action":"finish","summary":"ok"}',
    ])
    result = runner.run_turn("use echo")
    assert any("x" in o for o in result.observations)


def test_bad_json_then_recovers(tmp_path):
    runner = build_runner(tmp_path, [
        'totally not json',
        '{"action":"finish","summary":"recovered"}',
    ])
    result = runner.run_turn("go")
    assert "recovered" in result.summary


def test_max_iterations_guard(tmp_path):
    runner = build_runner(tmp_path, ['{"action":"respond","text":"loop"}'] * 50)
    result = runner.run_turn("go")
    assert result.stopped_reason == "max_iterations"


def test_deps_exporter_receives_events(tmp_path):
    captured = []

    class RecordingExporter:
        def export(self, event):
            captured.append(event)

    reg = ToolRegistry()
    runner = AgentRunner(RunnerDeps(
        llm=FakeLLMClient(replies=['{"action":"finish","summary":"done"}']),
        registry=reg,
        ask=lambda *a: "n",
        log_dir=tmp_path,
        exporter=RecordingExporter(),
    ))
    runner.run_turn("go")
    assert captured  # the tracer forwarded at least one event to the exporter


def test_ask_user_flow(tmp_path):
    runner = build_runner(tmp_path, [
        '{"action":"ask_user","question":"which?"}',
        '{"action":"finish","summary":"ok"}',
    ], ask="events.csv")
    result = runner.run_turn("vague")
    assert any("events.csv" in o for o in result.observations)


def test_consecutive_failures_stop(tmp_path):
    reg = ToolRegistry()
    reg.register(Tool("boom", "always fails", "builtin", {"type": "object"},
                      lambda inp: ToolResult(ok=False, error="boom")))
    runner = AgentRunner(RunnerDeps(
        llm=FakeLLMClient(replies=['{"action":"call_tool","name":"boom","input":{}}'] * 10),
        registry=reg, ask=lambda *a: "n", log_dir=tmp_path,
        max_iterations=20, max_fix_retries=2))
    result = runner.run_turn("go")
    assert result.stopped_reason == "consecutive_failures"
