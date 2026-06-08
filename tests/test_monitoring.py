from adaptive_agent.monitoring import get_exporter, NoopExporter


def test_noop_default_never_raises():
    exp = get_exporter("off")
    assert isinstance(exp, NoopExporter)
    exp.export({"kind": "llm_call"})  # must not raise


def test_unknown_falls_back_to_noop():
    assert isinstance(get_exporter("unknown"), NoopExporter)
