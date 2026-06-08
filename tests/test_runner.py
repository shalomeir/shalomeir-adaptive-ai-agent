from adaptive_agent.runner import NON_INTERACTIVE_ASK, AgentRunner, RunnerDeps, _SYSTEM
from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.schemas import Message, ToolDigest
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


class RecordingLLM:
    def __init__(self, reply: str = '{"action":"finish","summary":"done"}') -> None:
        self.reply = reply
        self.calls = 0
        self.messages: list[Message] = []

    def chat(self, messages: list[Message], digests: list[ToolDigest]) -> str:
        self.calls += 1
        self.messages = messages
        return self.reply


def test_respond_then_finish(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"respond","text":"working","final":false}',
            '{"action":"finish","summary":"done"}',
        ],
    )
    result = runner.run_turn("start task")
    assert "done" in result.summary


def test_respond_without_final_finishes_immediately(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"respond","text":"네, 어떻게 도와드릴 수 있을까요?"}',
            '{"action":"respond","text":"should not be called"}',
        ],
    )

    result = runner.run_turn("어어 안녕")

    assert result.summary == "네, 어떻게 도와드릴 수 있을까요?"
    assert result.stopped_reason == "finish"
    assert runner.deps.llm.calls == 1


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


def test_known_file_is_inspected_before_first_plan(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "readFile",
            "read",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={
                    "content": (
                        '{"root":{"type":"Scene","children":[{"type":"Entity",'
                        '"props":{"health":120},"children":[]}]}}'
                    ),
                    "truncated": False,
                },
            ),
        )
    )
    llm = RecordingLLM()
    runner = AgentRunner(
        RunnerDeps(
            llm=llm,
            registry=reg,
            ask=lambda *a: "n",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json에서 health 평균을 알려줘")

    rendered = "\n".join(message.content for message in llm.messages)
    assert result.summary == "done"
    assert "작업 영역 파일을 미리 확인했습니다" in rendered
    assert "root는 object tree node처럼 보입니다" in rendered
    assert "root.children[].props.health" in rendered


def test_run_python_failure_repeats_file_structure_hint(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "readFile",
            "read",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={
                    "content": (
                        '{"root":{"children":[{"type":"Container","children":[{"type":"Entity",'
                        '"props":{"health":120},"children":[]}]}]}}'
                    ),
                    "truncated": False,
                },
            ),
        )
    )
    reg.register(
        Tool(
            "runPython",
            "run",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(ok=False, error="KeyError: 'health'"),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"call_tool","name":"runPython","input":{"code":"bad"}}',
                    '{"action":"finish","summary":"stopped"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "n",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json에서 health 평균을 알려줘")

    assert result.summary == "stopped"
    assert any("모든 descendants를 재귀 순회" in o for o in result.observations)
    assert any("root.children[].children[].props.health" in o for o in result.observations)


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
    assert any(event["kind"] == "llm_call_start" for event in events)


def test_llm_response_preview_is_bounded(tmp_path):
    raw = '{"action":"respond","text":"' + ("x" * 5000) + '"}'
    runner = build_runner(tmp_path, [raw])

    runner.run_turn("go")

    events = read_log_events(runner)
    first_llm_event = next(event for event in events if event["kind"] == "llm_call")
    assert len(first_llm_event["responsePreview"]) == 4000
    assert first_llm_event["responseChars"] == len(raw)
    assert first_llm_event["responseTruncated"] is True


def test_bare_json_response_gets_protocol_feedback_and_retries(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"path":"events-clean.csv","rows":5,"removed":2}',
            '{"action":"respond","text":"events-clean.csv 저장 완료. rows=5, removed=2","final":true}',
        ],
    )

    result = runner.run_turn("events.csv 정리 결과 알려줘")

    assert result.stopped_reason == "finish"
    assert "events-clean.csv 저장 완료" in result.summary
    assert runner.deps.llm.calls == 2
    assert any("action 필드가 없습니다" in observation for observation in result.observations)


def test_max_iterations_guard(tmp_path):
    # Distinct actions each turn keep advancing (no-progress guard does not fire),
    # so the run is bounded only by max_iterations.
    runner = build_runner(
        tmp_path, [f'{{"action":"respond","text":"loop {i}","final":false}}' for i in range(50)]
    )
    result = runner.run_turn("go")
    assert result.stopped_reason == "max_iterations"


def test_repeated_parse_failures_stop_early(tmp_path):
    # 약한 모델이 JSON action을 계속 깨뜨리면, max_iterations까지 헛돌지 말고
    # 연속 파싱 실패 상한에서 일찍 멈춰야 한다("무한 반복" 체감 방지).
    runner = build_runner(tmp_path, ["totally not json"] * 50)  # max_iterations=10
    result = runner.run_turn("go")
    assert result.stopped_reason == "parse_failures"


def test_model_question_enters_agent_loop(tmp_path):
    llm = FakeLLMClient(replies=['{"action":"finish","summary":"runtime handled"}'])
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

    assert result.summary == "runtime handled"
    assert llm.calls == 1
    events = read_log_events(runner)
    assert any(event["kind"] == "llm_call_start" for event in events)


def test_model_word_in_task_still_enters_agent_loop(tmp_path):
    runner = build_runner(
        tmp_path,
        ['{"action":"finish","summary":"loop used"}'],
    )

    result = runner.run_turn("model metrics 파일을 읽어서 요약해줘")

    assert result.summary == "loop used"


def test_small_talk_enters_agent_loop(tmp_path):
    llm = FakeLLMClient(replies=['{"action":"finish","summary":"chat handled"}'])
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

    assert result.summary == "chat handled"
    assert llm.calls == 1
    events = read_log_events(runner)
    assert any(event["kind"] == "llm_call_start" for event in events)


def test_runtime_identity_complaint_does_not_reuse_previous_task(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            '{"action":"respond","text":"previous data result","final":true}',
            '{"action":"respond","text":"runtime identity answer","final":true}',
        ],
    )

    first = runner.run_turn("analyze data")
    second = runner.run_turn("아니 너 뭐냐.")

    assert first.summary == "previous data result"
    assert second.summary == "runtime identity answer"
    assert "previous data result" not in second.summary


def test_system_prompt_tells_model_not_to_dump_full_records():
    assert "specific fields or aggregates" in _SYSTEM
    assert "instead of dumping full records" in _SYSTEM


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
    assert result.summary == "HITL 처리가 필요합니다: 어떤 데이터를 정리할까요?"
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

    result = runner.run_turn("events.csv를 정렬해서 events-clean.csv로 저장해줘.")

    assert result.summary == "continued"
    assert any("events.csv, events-clean.csv" in o for o in result.observations)


def test_final_response_that_repeats_actionable_request_is_blocked(tmp_path):
    runner = build_runner(
        tmp_path,
        [
            (
                '{"action":"respond","text":"world.json에서 health가 100 미만인 Entity를 '
                '모두 제거하고, 남은 Entity의 평균 health를 알려줘.","final":true}'
            ),
            '{"action":"finish","summary":"average health is 190"}',
        ],
    )

    result = runner.run_turn(
        "world.json에서 health가 100 미만인 Entity를 모두 제거하고, 남은 Entity의 평균 "
        "health를 알려줘. write or update 하지는 말고."
    )

    assert result.summary == "average health is 190"
    assert any("요청을 최종 답변이나 질문으로 되풀이하지" in o for o in result.observations)


def test_ask_user_that_repeats_actionable_request_is_blocked(tmp_path):
    asks = []
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"ask_user","question":"world.json 파일에서 health가 100 '
                        '미만인 Entity를 제거한 후 남은 Entity의 평균 health를 알려줘."}'
                    ),
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=ToolRegistry(),
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn(
        "world.json 파일에서 health가 100 미만인 Entity를 제거한 후 남은 Entity의 평균 "
        "health를 알려줘."
    )

    assert result.summary == "continued"
    assert asks == []
    assert any("요청을 최종 답변이나 질문으로 되풀이하지" in o for o in result.observations)


def test_file_format_ask_is_blocked_before_user_prompt(tmp_path):
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
                output={
                    "content": '{"root":{"props":{},"children":[]}}',
                    "truncated": False,
                },
            ),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"Entity는 어떤 형식으로 표현되어 있나요?"}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json에서 Entity 평균 health를 알려줘")

    assert result.summary == "continued"
    assert asks == []
    assert any("파일을 직접 열어" in o for o in result.observations)


def test_no_write_instruction_blocks_write_file_even_with_remove_word(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "writeFile",
            "write",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(ok=False, error="writeFile should not run"),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"call_tool","name":"writeFile","input":'
                        '{"path":"world-clean.json","content":"{}"}}'
                    ),
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn(
        "world.json에서 health가 100 미만인 Entity를 제거하고 평균을 알려줘. "
        "write or update 하지는 말고."
    )

    assert result.summary == "continued"
    assert any("파일 쓰기를 금지합니다" in o for o in result.observations)
    assert not any("writeFile should not run" in o for o in result.observations)


def test_md_file_request_allows_write_file(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "writeFile",
            "write",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(ok=True, output={"path": inp["path"]}),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"writeFile","path":"world.md",'
                        '"content":"graph TD\\nscene --> ground\\n"}'
                    ),
                    '{"action":"finish","summary":"saved"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert result.summary == "saved"
    assert any("world.md" in o for o in result.observations)


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
    assert any("질문에 나온 작업 영역 파일" in o for o in result.observations)
    assert any("listFields={'items': 1}" in o for o in result.observations)


def test_file_content_ask_with_path_in_question_is_blocked_before_user_prompt(tmp_path):
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
                output={"content": '{"monsters":[{"name":"Orc","hp":150}]}', "truncated": False},
            ),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"ask_user","question":"monsters.json 파일의 내용을 확인해 주실 수 있으신가요?"}',
                    '{"action":"finish","summary":"continued"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: asks.append(a) or "no",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("글자 3개인 경우만 처리하라니까.")

    assert result.summary == "continued"
    assert asks == []
    assert any("질문에 나온 작업 영역 파일" in o for o in result.observations)
    assert any("listFields={'monsters': 1}" in o for o in result.observations)


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
