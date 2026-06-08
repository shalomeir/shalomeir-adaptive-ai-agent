import json

from adaptive_agent.observability import Tracer


def test_jsonl_event_written(tmp_path):
    tracer = Tracer(log_dir=tmp_path)
    with tracer.trace() as trace_id:
        tracer.log(kind="llm_call", model="m", inputTokens=10, parseOk=True)
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["kind"] == "llm_call"
    assert evt["traceId"] == trace_id
    assert evt["model"] == "m"


def test_event_forwarded_to_exporter(tmp_path):
    captured = []

    class RecordingExporter:
        def export(self, event):
            captured.append(event)

    tracer = Tracer(log_dir=tmp_path, exporter=RecordingExporter())
    tracer.log(kind="tool_call", toolName="echo")
    assert len(captured) == 1
    assert captured[0]["kind"] == "tool_call"


def test_exporter_failure_does_not_break_logging(tmp_path):
    class BrokenExporter:
        def export(self, event):
            raise RuntimeError("monitor down")

    tracer = Tracer(log_dir=tmp_path, exporter=BrokenExporter())
    tracer.log(kind="tool_call", toolName="echo")  # must not raise
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
