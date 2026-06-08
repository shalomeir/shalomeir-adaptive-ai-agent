from adaptive_agent.runner import NON_INTERACTIVE_ASK, AgentRunner, RunnerDeps, _SYSTEM
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


def read_log_events(runner):
    import json

    return [json.loads(line) for line in runner.tracer.path.read_text().splitlines()]


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


def test_llm_response_preview_is_logged(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"finish","summary":"logged"}',
        ],
    )

    runner.run_turn("go")

    events = read_log_events(runner)
    llm_events = [event for event in events if event["kind"] == "llm_call"]
    assert llm_events[0]["responsePreview"] == '{"action":"finish","summary":"logged"}'
    assert llm_events[0]["responseChars"] == len('{"action":"finish","summary":"logged"}')
    assert llm_events[0]["responseTruncated"] is False


def test_llm_response_preview_is_bounded(tmp_path):
    raw = '{"action":"respond","text":"' + ("x" * 5000) + '"}'
    runner = build_runner(tmp_path, [raw])

    runner.run_turn("go")

    events = read_log_events(runner)
    first_llm_event = next(event for event in events if event["kind"] == "llm_call")
    assert len(first_llm_event["responsePreview"]) == 4000
    assert first_llm_event["responseChars"] == len(raw)
    assert first_llm_event["responseTruncated"] is True


def test_bare_json_response_finishes_instead_of_looping(tmp_path):
    runner = build_runner(
        tmp_path,
        ['{"path":"events-clean.csv","rows":5,"removed":2}'] * 10,
    )

    result = runner.run_turn("events.csv 정리 결과 알려줘")

    assert result.stopped_reason == "finish"
    assert '"events-clean.csv"' in result.summary
    assert runner.deps.llm.calls == 1


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
    assert not runner.tracer.path.exists()


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
    assert not runner.tracer.path.exists()


def test_runtime_identity_complaint_does_not_reuse_previous_task(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"respond","text":"previous data result","final":true}',
        ],
    )

    first = runner.run_turn("analyze data")
    second = runner.run_turn("아니 너 뭐냐.")

    assert first.summary == "previous data result"
    assert "adaptive-agent CLI" in second.summary
    assert "previous data result" not in second.summary


def test_polishes_record_dump_when_user_asked_for_names_and_average(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            (
                '{"action":"respond","final":true,"text":"High HP Monsters: '
                "[{'name': 'Orc', 'hp': 150}, {'name': 'Dragon', 'hp': 300}]"
                '\\nAverage HP: 225.0"}'
            ),
        ],
    )

    result = runner.run_turn("hp가 100 이상인 이름과 평균 hp")

    assert result.summary == "Orc, Dragon의 평균 HP는 225.00입니다."


def test_incomplete_loop_reports_last_result(tmp_path):
    # 모델이 도구는 돌렸지만 finish/respond(final)로 끝맺지 못하고 같은 호출만 반복하면,
    # 캐시된 성공 결과로 중단하되 빈 요약 대신 마지막 결과를 돌려줘야 한다.
    runner = build_runner(
        tmp_path,
        ['{"action":"call_tool","name":"echo","input":{"answer":42}}'] * 50,
    )
    result = runner.run_turn("go")
    assert result.stopped_reason == "cached_result"
    assert result.summary
    assert "42" in result.summary


def test_cached_result_loop_does_not_log_error_event(tmp_path):
    # 이미 성공한 동일 tool call 반복은 실패가 아니라 캐시 종료이므로 error 이벤트로 남기지 않는다.
    runner = build_runner(
        tmp_path,
        ['{"action":"call_tool","name":"echo","input":{"answer":42}}'] * 50,
    )
    runner.run_turn("go")
    events = read_log_events(runner)
    errors = [e for e in events if e["kind"] == "error"]
    assert errors == []


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


def test_non_interactive_file_structure_ask_is_auto_blocked(tmp_path):
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"몬스터 데이터 구조를 확인해 주세요."}',
                    '{"action":"ask_user","question":"파일 내부 root 키의 값이 리스트인 경우 이름을 알려주세요."}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=ToolRegistry(),
            ask=lambda *a: NON_INTERACTIVE_ASK,
            log_dir=tmp_path,
            non_interactive=True,
        )
    )

    result = runner.run_turn("monsters.json 분석")

    assert result.summary == "continued"
    assert sum("파일을 직접 열어" in o for o in result.observations) == 2


def test_non_interactive_general_ask_ends_with_hitl_required(tmp_path):
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"어떤 데이터를 정리할까요?"}',
                ]
            ),
            registry=ToolRegistry(),
            ask=lambda *a: NON_INTERACTIVE_ASK,
            log_dir=tmp_path,
            non_interactive=True,
        )
    )

    result = runner.run_turn("데이터 좀 정리해줘")

    assert result.stopped_reason == "hitl_required"
    assert result.summary == (
        "HITL 처리가 필요합니다: 어떤 데이터를 어떻게 정리할까요? "
        "파일명과 원하는 작업을 같이 알려주세요. "
        "예: events.csv에서 중복 제거하고 date로 정렬해줘."
    )
    assert not any("사용자 답변: n" in o for o in result.observations)


def test_non_interactive_known_file_path_ask_is_auto_blocked(tmp_path):
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"input_file과 output_file의 경로를 입력해주세요."}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=ToolRegistry(),
            ask=lambda *a: NON_INTERACTIVE_ASK,
            log_dir=tmp_path,
            non_interactive=True,
        )
    )

    result = runner.run_turn(
        "events.csv를 정렬해서 events-clean.csv로 저장해줘."
    )

    assert result.summary == "continued"
    assert any("events.csv, events-clean.csv" in o for o in result.observations)


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


def test_file_structure_ask_is_blocked_before_user_prompt(tmp_path):
    asks = []
    reg = ToolRegistry()
    reg.register(
        Tool(
            "readFile",
            "read",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={"content": '{"items":[{"name":"a","value":1}]}', "truncated": False},
            ),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"ask_user","question":"Could you provide more details about '
                        'the structure of the data in data.json?"}'
                    ),
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("analyze data.json")

    assert result.summary == "continued"
    assert asks == []
    assert any("파일을 직접 열어" in o for o in result.observations)
    assert any("listFields={'items': 1}" in o for o in result.observations)


def test_file_field_count_ask_is_blocked_before_user_prompt(tmp_path):
    asks = []
    reg = ToolRegistry()
    reg.register(
        Tool(
            "readFile",
            "read",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={"content": '{"items":[{"name":"a","value":1}]}', "truncated": False},
            ),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"데이터는 각각 몇 개의 필드를 가지고 있나요?"}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("data.json 분석해줘")

    assert result.summary == "continued"
    assert asks == []
    assert any("파일을 직접 열어" in o for o in result.observations)


def test_file_field_type_ask_is_blocked_before_user_prompt(tmp_path):
    asks = []
    reg = ToolRegistry()
    reg.register(
        Tool(
            "readFile",
            "read",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={"content": '{"items":[{"name":"a","hp":100}]}', "truncated": False},
            ),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"hp가 문자열로 표현되어 있는지 확인해 주세요."}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("data.json에서 hp 평균을 알려줘")

    assert result.summary == "continued"
    assert asks == []
    assert any("파일을 직접 열어" in o for o in result.observations)


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


def test_system_prompt_is_not_demo_case_specific():
    forbidden = (
        "events.csv",
        "monsters.json",
        "query" + "Monster" + "Hp",
        "aggregate" + "Csv",
        "normalize" + "Csv",
    )

    for term in forbidden:
        assert term not in _SYSTEM
