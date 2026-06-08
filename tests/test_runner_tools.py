import json

from adaptive_agent.runner import AgentRunner, RunnerDeps
from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.tools.registry import ToolRegistry
from adaptive_agent.tools.generated import GeneratedToolManager
from adaptive_agent.tools.builtins import build_file_tools, build_run_python, build_search_docs
from adaptive_agent.skills import SkillStore
from adaptive_agent.sandbox import ExecutionSandbox
from adaptive_agent.policy import PolicyManager
from adaptive_agent.schemas import ToolSpec
from adaptive_agent.tools.base import Tool, ToolResult


def build(tmp_path, replies, ask="y"):
    ws = tmp_path / "ws"
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    deps = RunnerDeps(
        llm=FakeLLMClient(replies=replies),
        registry=ToolRegistry(),
        ask=lambda *a: ask,
        log_dir=tmp_path,
    )
    return AgentRunner(
        deps,
        generated=GeneratedToolManager(ws / ".session", sandbox),
        skills=SkillStore(tmp_path / "skills"),
        policy=PolicyManager(ask=lambda q: ask),
    )


def test_create_tool_registers_and_calls(tmp_path):
    runner = build(
        tmp_path,
        [
            '{"action":"create_tool","spec":{"name":"adder","description":"adds",'
            '"code":"def run(input):\\n    return {\\"sum\\": input[\\"a\\"]+input[\\"b\\"]}",'
            '"inputSchema":{"type":"object"}}}',
            '{"action":"call_tool","name":"adder","input":{"a":2,"b":3}}',
            '{"action":"finish","summary":"5"}',
        ],
    )
    result = runner.run_turn("add 2 and 3")
    assert any("5" in o for o in result.observations)


def test_persist_offer_saves_on_yes(tmp_path):
    runner = build(
        tmp_path,
        [
            '{"action":"create_tool","spec":{"name":"adder","description":"adds",'
            '"code":"def run(input):\\n    return {\\"ok\\": True}",'
            '"inputSchema":{"type":"object"}}}',
            '{"action":"finish","summary":"done"}',
        ],
        ask="y",
    )
    runner.run_turn("make adder")
    assert (tmp_path / "skills" / "adder" / "manifest.json").exists()


def test_update_persisted_tool_is_reoffered(tmp_path):
    skills = SkillStore(tmp_path / "skills")
    skills.persist(
        ToolSpec(
            name="adder",
            description="adds",
            code='def run(input):\n    return {"v": 1}',
            inputSchema={"type": "object"},
        )
    )
    ws = tmp_path / "ws"
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                '{"action":"update_tool","name":"adder",'
                '"code":"def run(input):\\n    return {\\"v\\": 2}"}',
                '{"action":"finish","summary":"done"}',
            ]
        ),
        registry=ToolRegistry(),
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(
        deps,
        generated=GeneratedToolManager(ws / ".session", sandbox),
        skills=skills,
        policy=PolicyManager(ask=lambda q: "y"),
    )
    runner.run_turn("update adder")
    assert '"v": 2' in (tmp_path / "skills" / "adder" / "tool.py").read_text()


def _write_runner(tmp_path, path, ask):
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = ToolRegistry()
    for t in build_file_tools(ws):
        reg.register(t)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                '{"action":"call_tool","name":"writeFile","input":'
                f'{{"path":"{path}","content":"hello"}}}}',
                '{"action":"finish","summary":"done"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: ask,
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: ask))
    return ws, runner


def test_write_out_of_workspace_denied(tmp_path):
    ws, runner = _write_runner(tmp_path, "../escape.txt", ask="n")
    result = runner.run_turn("write outside")
    assert any("거부" in o for o in result.observations)
    assert not (tmp_path / "escape.txt").exists()


def test_write_in_workspace_declined(tmp_path):
    ws, runner = _write_runner(tmp_path, "out.txt", ask="n")
    result = runner.run_turn("write")
    assert any("거부" in o for o in result.observations)
    assert not (ws / "out.txt").exists()


def test_write_in_workspace_approved(tmp_path):
    ws, runner = _write_runner(tmp_path, "out.txt", ask="y")
    runner.run_turn("write")
    assert (ws / "out.txt").read_text() == "hello"


def test_run_python_failure_surfaces_stderr(tmp_path):
    # runPython이 실패하면 stderr를 ToolResult.error로 올려, 모델이 무엇이 틀렸는지
    # 보고 다음 턴에 고칠 수 있어야 한다(자가수정 루프). 비어 있으면 같은 실수를 반복한다.
    from adaptive_agent.tools.builtins import build_run_python

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    tool = build_run_python(sandbox)

    res = tool.handler({"code": "return 1"})  # top-level return → SyntaxError

    assert res.ok is False
    assert res.error  # 실패 사유가 비어 있으면 안 된다
    assert "return" in res.error.lower() or "syntax" in res.error.lower()


def test_repeated_ask_user_stops_with_no_progress(tmp_path):
    # A weak model can spin emitting the same ask_user forever. These parse fine,
    # so they never trip the failure counter — the no-progress guard must stop it
    # instead of prompting the user to max_iterations.
    ask_reply = '{"action":"ask_user","question":"무엇을 정리할까요?"}'
    runner = build(tmp_path, [ask_reply] * 6)
    result = runner.run_turn("데이터 좀 정리해줘")
    assert result.stopped_reason == "no_progress"


def test_repeated_identical_tool_call_reuses_cached_result(tmp_path):
    # Re-calling the same tool with identical input never advances state. Each call
    # may succeed (so fix_failures resets), but the run is going nowhere.
    create = (
        '{"action":"create_tool","spec":{"name":"noop","description":"noop",'
        '"code":"def run(input):\\n    return {\\"ok\\": True}",'
        '"inputSchema":{"type":"object"}}}'
    )
    call = '{"action":"call_tool","name":"noop","input":{"x":1}}'
    runner = build(tmp_path, [create] + [call] * 6)
    result = runner.run_turn("noop forever")
    assert result.stopped_reason == "cached_result"


def test_repeated_identical_write_file_prompts_only_once(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    asks = []
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                '{"action":"call_tool","name":"writeFile","input":{"path":"out.txt","content":"hello"}}'
            ]
            * 6
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: asks.append(q) or "y"))

    result = runner.run_turn("write once")

    assert result.stopped_reason == "cached_result"
    assert asks == ["'write_file' 작업을 진행할까요? (y/n)"]
    assert (ws / "out.txt").read_text() == "hello"


def test_run_python_direct_file_write_can_finish_from_created_file(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    asks = []
    reg = ToolRegistry()
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                (
                    '{"action":"call_tool","name":"runPython","input":'
                    '{"code":"open(\\"out.csv\\", \\"w\\").write(\\"a,b\\\\n\\")"}}'
                )
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: asks.append(q) or "y"))

    result = runner.run_turn("save result to out.csv")

    assert result.stopped_reason == "finish"
    assert result.summary == "out.csv 파일 저장이 완료되었습니다."
    assert (ws / "out.csv").read_text() == "a,b\n"
    assert asks == ["'write_file' 작업을 진행할까요? (y/n)"]


def test_ask_user_answer_extends_file_write_intent(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "input.csv").write_text("id,date\n2,2026-01-02\n1,2026-01-01\n")
    reg = ToolRegistry()
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                '{"action":"ask_user","question":"무슨 데이터를 어떻게 정리할까요?"}',
                (
                    '{"action":"call_tool","name":"runPython","input":'
                    '{"code":"open(\\"out.csv\\", \\"w\\").write(\\"id,date\\\\n1,2026-01-01\\\\n2,2026-01-02\\\\n\\")"}}'
                ),
            ]
        ),
        registry=reg,
        ask=lambda *a: "input.csv를 date 기준 오름차순으로 정렬해서 out.csv로 저장해줘.",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: "y"))

    result = runner.run_turn("데이터 좀 정리해줘.")

    assert result.stopped_reason == "finish"
    assert result.summary == "out.csv 파일 저장이 완료되었습니다."
    assert (ws / "out.csv").read_text() == "id,date\n1,2026-01-01\n2,2026-01-02\n"


def test_run_python_direct_file_write_validates_requested_csv_sort(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = ToolRegistry()
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                (
                    '{"action":"call_tool","name":"runPython","input":'
                    '{"code":"open(\\"out.csv\\", \\"w\\").write(\\"date\\\\n2026-02-01\\\\n2026-01-01\\\\n\\")"}}'
                ),
                '{"action":"finish","summary":"fixed"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: "y"))

    result = runner.run_turn("sort by date ascending and save to out.csv")

    assert result.summary == "fixed"
    assert any("date 기준 오름차순이 아닙니다" in o for o in result.observations)


def test_generated_tool_direct_file_write_is_validated_before_finish(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "input.csv").write_text("id,date\n1,2026-01-01\n2,2026-01-02\n")
    bad_code = (
        "def run(input):\n"
        "    open(input['output'], 'w').write('id,date\\n2,2026-01-02\\n1,2026-01-01\\n')\n"
    )
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            [
                json.dumps(
                    {
                        "action": "create_tool",
                        "spec": {
                            "name": "bad-sort",
                            "description": "bad sort",
                            "code": bad_code,
                            "inputSchema": {"type": "object"},
                        },
                    }
                ),
                '{"action":"call_tool","name":"bad-sort","input":{"output":"out.csv"}}',
                '{"action":"finish","summary":"fixed"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(
        deps,
        generated=GeneratedToolManager(ws / ".session", sandbox),
        skills=SkillStore(tmp_path / "skills"),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn("input.csv duplicate rows remove and sort by date ascending and save to out.csv")

    assert result.stopped_reason == "finish"
    assert result.summary.startswith("검증 실패를 감지해")
    assert (ws / "out.csv").read_text() == "id,date\n1,2026-01-01\n2,2026-01-02\n"
    assert not (tmp_path / "skills" / "bad-sort").exists()
    assert (tmp_path / "skills" / "csv-dedupe-sort" / "manifest.json").exists()


def test_run_python_direct_file_write_validates_dedupe_preserves_unique_rows(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "input.csv").write_text("id,date\n1,2026-01-01\n2,2026-01-02\n1,2026-01-01\n")
    reg = ToolRegistry()
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                (
                    '{"action":"call_tool","name":"runPython","input":'
                    '{"code":"open(\\"out.csv\\", \\"w\\").write(\\"id,date\\\\n1,2026-01-01\\\\n\\")"}}'
                ),
                '{"action":"finish","summary":"fixed"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: "y"))

    result = runner.run_turn("input.csv duplicate rows remove and save to out.csv")

    assert result.summary == "fixed"
    assert any("고유 행 내용" in o for o in result.observations)


def test_final_csv_grouped_amount_summary_recomputes_after_dedupe(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "events.csv").write_text(
        "id,date,type,amount\n"
        "3,2026-03-02,purchase,1200\n"
        "1,2026-01-15,signup,0\n"
        "2,2026-02-20,purchase,800\n"
        "2,2026-02-20,purchase,800\n"
        "4,2026-01-15,refund,-200\n"
        "5,2026-04-10,purchase,500\n"
    )
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=['{"action":"respond","text":"purchase: 3300","final":true}']
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps)

    result = runner.run_turn(
        "events.csv에서 완전히 중복된 행은 한 번만 세고, amount 합계를 type별로 구해줘."
    )

    assert result.summary == "purchase: 2500\nsignup: 0\nrefund: -200"


def test_direct_object_tree_health_task_prunes_and_averages(tmp_path):
    ws = tmp_path / "ws"
    docs = tmp_path / "docs"
    ws.mkdir()
    docs.mkdir()
    docs.joinpath("schema.md").write_text("Entity nodes store health in props.")
    ws.joinpath("world.json").write_text(
        json.dumps(
            {
                "root": {
                    "id": "scene",
                    "type": "Scene",
                    "props": {},
                    "children": [
                        {
                            "id": "low",
                            "type": "Entity",
                            "name": "Low",
                            "props": {"health": 80},
                            "children": [],
                        },
                        {
                            "id": "high",
                            "type": "Entity",
                            "name": "High",
                            "props": {"health": 120},
                            "children": [],
                        },
                    ],
                }
            }
        )
    )
    asks = []
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_search_docs(docs))
    deps = RunnerDeps(
        llm=FakeLLMClient(replies=[]),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: asks.append(q) or "y"))

    result = runner.run_turn(
        "world.json에서 health가 100 미만인 Entity를 모두 제거하고, "
        "남은 Entity의 평균 health를 알려줘."
    )

    assert result.summary == "제거: Low\n남은 Entity 평균 health: 120"
    assert asks == ["'write_file' 작업을 진행할까요? (y/n)"]
    world = json.loads(ws.joinpath("world.json").read_text())
    assert [child["id"] for child in world["root"]["children"]] == ["high"]


def test_direct_context_markdown_table_uses_previous_result(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    ws.joinpath("monsters.json").write_text(
        json.dumps(
            {
                "monsters": [
                    {"name": "Orc", "hp": 150},
                    {"name": "Dragon", "hp": 300},
                    {"name": "Wolf", "hp": 110},
                    {"name": "Slime", "hp": 20},
                ]
            }
        )
    )
    asks = []
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(replies=[]),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: asks.append(q) or "y"),
    )
    runner.conv.add_user("workspace의 monsters.json에서 hp가 100 이상인 몬스터 이름을 알려줘.")
    runner.conv.add_assistant("몬스터 이름: Orc, Dragon, Wolf\n평균 HP: 186.67")

    result = runner.run_turn("방금 필터된 결과를 hp 내림차순 마크다운 표로 table.md에 저장해줘.")

    assert result.summary == "table.md 파일 저장이 완료되었습니다."
    assert ws.joinpath("table.md").read_text() == (
        "| name | hp |\n| --- | ---: |\n| Dragon | 300 |\n| Orc | 150 |\n| Wolf | 110 |\n"
    )
    assert asks == ["'write_file' 작업을 진행할까요? (y/n)"]


def test_outside_workspace_write_request_is_denied_before_tools(tmp_path):
    reg = ToolRegistry()
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(replies=['{"action":"finish","summary":"should-not-run"}']),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn("events.csv를 정렬해서 ../events-sorted.csv에 저장해줘.")

    assert result.summary.startswith("정책상 거부됨: out_of_workspace")
    assert runner.deps.llm.calls == 0


def test_read_only_request_blocks_file_transform_generated_tool(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            "sort-csv",
            "Remove duplicate rows and sort CSV by date.",
            "generated",
            {"type": "object"},
            lambda inp: ToolResult(ok=True, output={"path": "out.csv"}),
        )
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"call_tool","name":"sort-csv","input":{}}',
                    '{"action":"finish","summary":"used runPython instead"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("amount 합계를 type별로 알려줘")

    assert result.summary == "used runPython instead"
    assert any("읽기 전용" in o for o in result.observations)


def test_blocked_generated_tool_is_hidden_for_rest_of_turn(tmp_path):
    seen_digests = []

    class RecordingLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, digests):
            seen_digests.append([digest.name for digest in digests])
            self.calls += 1
            if self.calls == 1:
                return '{"action":"call_tool","name":"sum-amount-by-type","input":{}}'
            if self.calls == 2:
                return '{"action":"call_tool","name":"runPython","input":{"code":"print(\\"purchase: 2500\\")"}}'
            return '{"action":"finish","summary":"purchase: 2500"}'

    reg = ToolRegistry()
    reg.register(
        Tool(
            "sum-amount-by-type",
            "Sums the amount for each type and writes the result to a CSV file.",
            "generated",
            {"type": "object"},
            lambda inp: ToolResult(ok=True, output={"path": "events-clean.csv"}),
        )
    )
    sandbox = ExecutionSandbox(tmp_path / "ws", timeout_sec=5, max_output_bytes=4096)
    reg.register(build_run_python(sandbox))
    runner = AgentRunner(
        RunnerDeps(
            llm=RecordingLLM(),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("events.csv amount 합계를 type별로 알려줘")

    assert result.summary == "purchase: 2500"
    assert "sum-amount-by-type" in seen_digests[0]
    assert "sum-amount-by-type" not in seen_digests[1]


def test_update_unknown_tool_is_graceful(tmp_path):
    # The model may try to update a built-in or nonexistent tool; the runner
    # must not crash, just observe and move on.
    runner = build(
        tmp_path,
        [
            '{"action":"update_tool","name":"runPython","code":"def run(input):\\n    return {}"}',
            '{"action":"finish","summary":"done"}',
        ],
    )
    result = runner.run_turn("update a builtin")
    assert result.stopped_reason == "finish"
    assert any("생성 도구가 아닙니다" in o for o in result.observations)
