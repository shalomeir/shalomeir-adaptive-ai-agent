import json

from adaptive_agent.observability import Tracer


def test_jsonl_event_written(tmp_path):
    tracer = Tracer(log_dir=tmp_path)
    with tracer.trace() as trace_id:
        tracer.log(
            kind="llm_call",
            model="m",
            inputTokens=10,
            parseOk=True,
            responsePreview='{"action":"finish"}',
            responseChars=19,
            responseTruncated=False,
        )
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["kind"] == "llm_call"
    assert evt["traceId"] == trace_id
    assert evt["model"] == "m"
    assert evt["responsePreview"] == '{"action":"finish"}'
    assert evt["responseChars"] == 19
    assert evt["responseTruncated"] is False


def test_parent_span_id_propagated(tmp_path):
    # 스키마(LogEvent.parentSpanId)대로 중첩 span의 부모를 실제로 기록해야 한다.
    tracer = Tracer(log_dir=tmp_path)
    with tracer.trace():
        with tracer.span():
            tracer.log(kind="llm_call")
            with tracer.span():
                tracer.log(kind="tool_call")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    outer, inner = events[0], events[1]
    assert outer["parentSpanId"] is None  # 최상위 span은 부모가 없다
    assert inner["parentSpanId"] == outer["spanId"]  # 중첩 span은 부모 spanId를 가리킨다


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
