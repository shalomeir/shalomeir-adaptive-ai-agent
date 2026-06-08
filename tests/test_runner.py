from adaptive_agent.runner import AgentRunner, RunnerDeps
from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.tools.registry import ToolRegistry
from adaptive_agent.tools.base import Tool, ToolResult


def build_runner(tmp_path, replies, ask="n"):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "echo",
            "echo",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(ok=True, output=inp),
        )
    )
    return AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(replies=replies),
            registry=reg,
            ask=lambda *a: ask,
            log_dir=tmp_path,
            max_iterations=10,
            max_fix_retries=2,
        )
    )


def test_respond_then_finish(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"respond","text":"working"}',
            '{"action":"finish","summary":"done"}',
        ],
    )
    result = runner.run_turn("start task")
    assert "done" in result.summary


def test_call_tool_observation(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"call_tool","name":"echo","input":{"x":1}}',
            '{"action":"finish","summary":"ok"}',
        ],
    )
    result = runner.run_turn("use echo")
    assert any("x" in o for o in result.observations)


def test_bad_json_then_recovers(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            "totally not json",
            '{"action":"finish","summary":"recovered"}',
        ],
    )
    result = runner.run_turn("go")
    assert "recovered" in result.summary


def test_max_iterations_guard(tmp_path):
    # Distinct actions each turn keep advancing (no-progress guard does not fire),
    # so the run is bounded only by max_iterations.
    runner = build_runner(
        tmp_path, [f'{{"action":"respond","text":"loop {i}"}}' for i in range(50)]
    )
    result = runner.run_turn("go")
    assert result.stopped_reason == "max_iterations"


def test_repeated_parse_failures_stop_early(tmp_path):
    # 약한 모델이 JSON action을 계속 깨뜨리면, max_iterations까지 헛돌지 말고
    # 연속 파싱 실패 상한에서 일찍 멈춰야 한다("무한 반복" 체감 방지).
    runner = build_runner(tmp_path, ["totally not json"] * 50)  # max_iterations=10
    result = runner.run_turn("go")
    assert result.stopped_reason == "parse_failures"


def test_model_question_answers_without_llm_call(tmp_path):
    llm = FakeLLMClient(replies=[])
    llm.model = "qwen2.5-coder:7b"
    llm.base_url = "http://localhost:11434/v1"
    runner = AgentRunner(
        RunnerDeps(
            llm=llm,
            registry=ToolRegistry(),
            ask=lambda *a: "n",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("너 무슨 모델?")

    assert "qwen2.5-coder:7b" in result.summary
    assert "http://localhost:11434/v1" in result.summary
    assert llm.calls == 0
    assert not (tmp_path / "events.jsonl").exists()


def test_model_word_in_task_still_enters_agent_loop(tmp_path):
    runner = build_runner(
        tmp_path,
        ['{"action":"finish","summary":"loop used"}'],
    )

    result = runner.run_turn("model metrics 파일을 읽어서 요약해줘")

    assert result.summary == "loop used"


def test_small_talk_answers_without_ask_user_boilerplate(tmp_path):
    llm = FakeLLMClient(replies=[])
    llm.model = "qwen2.5-coder:7b"
    runner = AgentRunner(
        RunnerDeps(
            llm=llm,
            registry=ToolRegistry(),
            ask=lambda *a: "n",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("그냥 대화.")

    assert "도구 실행 말고 그냥 얘기" in result.summary
    assert llm.calls == 0
    assert not (tmp_path / "events.jsonl").exists()


def test_incomplete_loop_reports_last_result(tmp_path):
    # 모델이 도구는 돌렸지만 finish/respond(final)로 끝맺지 못하고 같은 호출만 반복하면,
    # no-progress로 중단하되 빈 요약 대신 마지막 관찰(실제 결과)을 돌려줘야 한다.
    runner = build_runner(
        tmp_path,
        ['{"action":"call_tool","name":"echo","input":{"answer":42}}'] * 50,
    )
    result = runner.run_turn("go")
    assert result.stopped_reason == "no_progress"
    assert result.summary
    assert "42" in result.summary


def test_incomplete_loop_logs_error_event(tmp_path):
    # 비정상 종료는 종료 사유를 error 이벤트로 남겨 로그만으로 추적 가능해야 한다.
    import json

    runner = build_runner(
        tmp_path,
        ['{"action":"call_tool","name":"echo","input":{"answer":42}}'] * 50,
    )
    runner.run_turn("go")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    errors = [e for e in events if e["kind"] == "error"]
    assert errors
    assert errors[-1]["errorKind"] == "no_progress"


def test_deps_exporter_receives_events(tmp_path):
    captured = []

    class RecordingExporter:
        def export(self, event):
            captured.append(event)

    reg = ToolRegistry()
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(replies=['{"action":"finish","summary":"done"}']),
            registry=reg,
            ask=lambda *a: "n",
            log_dir=tmp_path,
            exporter=RecordingExporter(),
        )
    )
    runner.run_turn("go")
    assert captured  # the tracer forwarded at least one event to the exporter


def test_ask_user_flow(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"ask_user","question":"which?"}',
            '{"action":"finish","summary":"ok"}',
        ],
        ask="events.csv",
    )
    result = runner.run_turn("vague")
    assert any("events.csv" in o for o in result.observations)


def test_package_install_ask_is_blocked_before_user_prompt(tmp_path):
    asks = []
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"Pandas 모듈이 설치되어 있지 않습니다. 설치하시겠습니까?"}',
                    '{"action":"ask_user","question":"Pandas를 사용할 수 없습니다. 대신 표준 라이브러리를 사용하여 저장해주세요."}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=ToolRegistry(),
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("dedup and sort events.csv")

    assert result.summary == "continued"
    assert asks == []
    assert len([o for o in result.observations if "표준 라이브러리" in o]) == 2


def test_consecutive_failures_stop(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "boom",
            "always fails",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(ok=False, error="boom"),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(replies=['{"action":"call_tool","name":"boom","input":{}}'] * 10),
            registry=reg,
            ask=lambda *a: "n",
            log_dir=tmp_path,
            max_iterations=20,
            max_fix_retries=2,
        )
    )
    result = runner.run_turn("go")
    assert result.stopped_reason == "consecutive_failures"
