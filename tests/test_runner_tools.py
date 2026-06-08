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


def test_file_task_cannot_finish_after_tool_creation_without_execution(tmp_path):
    asks = []
    runner = build(
        tmp_path,
        [
            '{"action":"create_tool","spec":{"name":"csv-dedupe-sort","description":"dedupe sort",'
            '"code":"def run(input):\\n    return {\\"path\\": input[\\"output\\"]}",'
            '"inputSchema":{"type":"object"}}}',
            (
                '{"action":"respond","text":"csv-dedupe-sort 도구가 생성되었습니다. 이제 '
                "events.csv를 처리하여 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 "
                'events-clean.csv로 저장해주세요.","final":true}'
            ),
            (
                '{"action":"call_tool","name":"csv-dedupe-sort","input":'
                '{"input":"events.csv","output":"events-clean.csv"}}'
            ),
            '{"action":"finish","summary":"events-clean.csv 저장 완료"}',
        ],
    )
    runner.policy = PolicyManager(ask=lambda q: asks.append(q) or "y")

    result = runner.run_turn(
        "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 "
        "events-clean.csv로 저장해줘."
    )

    assert result.summary == "events-clean.csv 저장 완료"
    assert any("실제 실행 결과가 아직 없습니다" in o for o in result.observations)
    assert any("csv-dedupe-sort" in o for o in result.observations)
    assert asks == [
        "생성한 도구 'csv-dedupe-sort'을(를) 다음 세션에서도 재사용하도록 저장할까요? (y/n)"
    ]


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
    result = runner.run_turn("작업을 도와줘")
    assert result.stopped_reason == "no_progress"
    assert result.summary == "작업이 진전 없이 반복되어 중단했습니다."


def test_vague_data_cleanup_uses_llm_ask_user_flow(tmp_path):
    answers = []
    runner = build(
        tmp_path,
        [
            '{"action":"ask_user","question":"어떤 데이터를 어떻게 정리할까요?"}',
            '{"action":"finish","summary":"continued"}',
        ],
    )
    runner.deps.ask = lambda *a: answers.append(a[0]) or "events.csv를 중복 제거해줘"

    result = runner.run_turn("데이터 좀 정리해줘.")

    assert result.summary == "continued"
    assert answers == ["어떤 데이터를 어떻게 정리할까요?"]


def test_incomplete_summary_hides_internal_tool_creation_observation(tmp_path):
    create = (
        '{"action":"create_tool","spec":{"name":"read-json","description":"read json",'
        '"code":"def run(input):\\n    return {}",'
        '"inputSchema":{"type":"object"}}}'
    )
    runner = build(tmp_path, [create] * 30)

    result = runner.run_turn("json 파일을 분석해줘")

    assert result.stopped_reason == "max_iterations"
    assert result.summary == "반복 한도 안에 작업을 끝내지 못했습니다."
    assert "생성·등록" not in result.summary


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
    assert asks == ["파일 쓰기가 필요합니다. 진행할까요? (y/n)"]
    assert (ws / "out.txt").read_text() == "hello"


def test_run_python_direct_file_write_is_blocked_by_tool(tmp_path):
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
                ),
                '{"action":"finish","summary":"used writeFile instead"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: asks.append(q) or "y"))

    result = runner.run_turn("save result to out.csv")

    assert result.stopped_reason == "finish"
    assert result.summary == "used writeFile instead"
    assert not (ws / "out.csv").exists()
    assert any(
        "runPython 안에서 파일을 직접 쓰려고 했습니다: out.csv" in o for o in result.observations
    )
    assert asks == []


def test_run_python_external_import_is_blocked_by_tool(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = ToolRegistry()
    reg.register(build_run_python(ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)))
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"call_tool","name":"runPython","input":'
                        '{"code":"from mermaid import MermaidGraph\\nprint(MermaidGraph)"}}'
                    ),
                    '{"action":"finish","summary":"retried with strings"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert result.summary == "retried with strings"
    assert any("외부 모듈을 import하지 마세요: mermaid" in o for o in result.observations)


def test_run_python_outside_direct_write_is_blocked_by_tool(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = ToolRegistry()
    reg.register(build_run_python(ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)))
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"call_tool","name":"runPython","input":'
                        '{"code":"open(\\"../world.md\\", \\"w\\").write(\\"x\\")"}}'
                    ),
                    '{"action":"finish","summary":"used writeFile"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert result.summary == "used writeFile"
    assert any("파일을 직접 쓰려고 했습니다: ../world.md" in o for o in result.observations)


def test_run_python_pathlib_direct_write_is_blocked_by_tool(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    reg = ToolRegistry()
    reg.register(build_run_python(ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)))
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"call_tool","name":"runPython","input":'
                        '{"code":"from pathlib import Path\\nPath(\\"out.md\\").write_text(\\"x\\")"}}'
                    ),
                    '{"action":"finish","summary":"used writeFile"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        )
    )

    result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert result.summary == "used writeFile"
    assert not ws.joinpath("out.md").exists()
    assert any("파일을 직접 쓰려고 했습니다: out.md" in o for o in result.observations)


def test_cached_run_python_result_does_not_infer_write_intent(tmp_path):
    writes = []
    reg = ToolRegistry()
    reg.register(
        Tool(
            "runPython",
            "run",
            "builtin",
            {"type": "object"},
            lambda inp: ToolResult(
                ok=True,
                output={
                    "stdout": "graph TD\nscene --> ground\n",
                    "stderr": "",
                    "exitCode": 0,
                    "timedOut": False,
                    "truncated": False,
                },
            ),
        )
    )
    reg.register(
        Tool(
            "writeFile",
            "write",
            "builtin",
            {"type": "object"},
            lambda inp: writes.append(inp) or ToolResult(ok=True, output={"path": inp["path"]}),
        )
    )
    call = '{"action":"call_tool","name":"runPython","input":{"code":"print(\\"graph TD\\")"}}'
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    call,
                    call,
                    '{"action":"writeFile","path":"world.md","content":"graph TD\\nscene --> ground\\n"}',
                    '{"action":"finish","summary":"saved"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert result.summary == "saved"
    assert writes == [{"path": "world.md", "content": "graph TD\nscene --> ground\n"}]
    assert any("도구 runPython 캐시 결과" in o for o in result.observations)


def test_ask_user_answer_can_drive_write_file_call(tmp_path):
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
                '{"action":"call_tool","name":"runPython","input":{"code":"print(\\"id,date\\\\n1,2026-01-01\\\\n2,2026-01-02\\\\n\\")"}}',
                (
                    '{"action":"call_tool","name":"writeFile","input":'
                    '{"path":"out.csv","content":"id,date\\n1,2026-01-01\\n2,2026-01-02\\n"}}'
                ),
                '{"action":"finish","summary":"out.csv 파일 저장이 완료되었습니다."}',
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


def test_csv_transform_is_done_by_python_and_write_file_tools(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "input.csv").write_text("id,date\n2,2026-01-02\n1,2026-01-01\n2,2026-01-02\n")
    content = "id,date\n1,2026-01-01\n2,2026-01-02\n"
    code = (
        "import csv, io, json\n"
        "rows = list(csv.reader(open('input.csv', newline='')))\n"
        "header, body = rows[0], rows[1:]\n"
        "seen = set(); unique = []\n"
        "for row in body:\n"
        "    key = tuple(row)\n"
        "    if key not in seen:\n"
        "        seen.add(key); unique.append(row)\n"
        "unique.sort(key=lambda row: row[header.index('date')])\n"
        "buf = io.StringIO(); writer = csv.writer(buf); writer.writerow(header); writer.writerows(unique)\n"
        "print(json.dumps({'content': buf.getvalue()}, ensure_ascii=False))\n"
    )
    reg = ToolRegistry()
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                json.dumps({"action": "call_tool", "name": "runPython", "input": {"code": code}}),
                json.dumps(
                    {
                        "action": "call_tool",
                        "name": "writeFile",
                        "input": {"path": "out.csv", "content": content},
                    }
                ),
                '{"action":"finish","summary":"out.csv saved"}',
            ]
        ),
        registry=reg,
        ask=lambda *a: "y",
        log_dir=tmp_path,
    )
    runner = AgentRunner(deps, policy=PolicyManager(ask=lambda q: "y"))

    result = runner.run_turn(
        "input.csv duplicate rows remove and sort by date ascending and save to out.csv"
    )

    assert result.summary == "out.csv saved"
    assert (ws / "out.csv").read_text() == content


def test_object_tree_mutation_uses_python_tool_loop(tmp_path):
    ws = tmp_path / "ws"
    docs = tmp_path / "docs"
    ws.mkdir()
    docs.mkdir()
    docs.joinpath("schema.md").write_text("Actor nodes store mana in props.")
    ws.joinpath("arena.json").write_text(
        json.dumps(
            {
                "root": {
                    "id": "scene",
                    "type": "Scene",
                    "props": {},
                    "children": [
                        {
                            "id": "low",
                            "type": "Actor",
                            "name": "LowMana",
                            "props": {"mana": 20},
                            "children": [],
                        },
                        {
                            "id": "high",
                            "type": "Actor",
                            "name": "HighMana",
                            "props": {"mana": 80},
                            "children": [],
                        },
                    ],
                }
            }
        )
    )
    content = json.dumps(
        {
            "root": {
                "id": "scene",
                "type": "Scene",
                "props": {},
                "children": [
                    {
                        "id": "high",
                        "type": "Actor",
                        "name": "HighMana",
                        "props": {"mana": 80},
                        "children": [],
                    }
                ],
            }
        },
        ensure_ascii=False,
    )
    code = (
        "import json\n"
        "data = json.load(open('arena.json'))\n"
        "data['root']['children'] = [\n"
        "    child for child in data['root']['children']\n"
        "    if not (child['type'] == 'Actor' and child['props'].get('mana', 0) < 50)\n"
        "]\n"
        "print(json.dumps({'content': json.dumps(data, ensure_ascii=False), 'avg': 80}, ensure_ascii=False))\n"
    )
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    reg.register(build_search_docs(docs))
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    reg.register(build_run_python(sandbox))
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    '{"action":"call_tool","name":"searchDocs","input":{"query":"mana","limit":3}}',
                    json.dumps(
                        {"action": "call_tool", "name": "runPython", "input": {"code": code}}
                    ),
                    json.dumps(
                        {
                            "action": "call_tool",
                            "name": "writeFile",
                            "input": {"path": "arena.json", "content": content},
                        }
                    ),
                    '{"action":"finish","summary":"제거: LowMana\\n남은 Actor 평균 mana: 80"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn(
        "arena.json에서 mana가 50 미만인 Actor를 모두 제거해서 arena.json을 업데이트하고, "
        "남은 Actor의 평균 mana를 알려줘."
    )

    assert result.summary == "제거: LowMana\n남은 Actor 평균 mana: 80"
    arena = json.loads(ws.joinpath("arena.json").read_text())
    assert [child["id"] for child in arena["root"]["children"]] == ["high"]
    assert runner.deps.llm.calls == 4


def test_previous_json_filter_table_uses_llm_tool_loop(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    ws.joinpath("players.json").write_text(
        json.dumps(
            {
                "players": [
                    {"name": "A", "score": 10, "team": "red"},
                    {"name": "B", "score": 30, "team": "blue"},
                    {"name": "C", "score": 20, "team": "red"},
                ]
            }
        )
    )
    table = "| name | score |\n| --- | --- |\n| B | 30 |\n| C | 20 |\n"
    code = (
        "import json\n"
        "players = json.load(open('players.json'))['players']\n"
        "selected = sorted([p for p in players if p['score'] >= 15], key=lambda p: p['score'], reverse=True)\n"
        "lines = ['| name | score |', '| --- | --- |']\n"
        "for p in selected:\n"
        "    lines.append(f\"| {p['name']} | {p['score']} |\")\n"
        "print(json.dumps({'content': '\\n'.join(lines) + '\\n'}, ensure_ascii=False))\n"
    )
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    sandbox = ExecutionSandbox(ws, timeout_sec=5, max_output_bytes=4096)
    reg.register(build_run_python(sandbox))
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    json.dumps(
                        {"action": "call_tool", "name": "runPython", "input": {"code": code}}
                    ),
                    json.dumps(
                        {
                            "action": "call_tool",
                            "name": "writeFile",
                            "input": {"path": "out.md", "content": table},
                        }
                    ),
                    '{"action":"finish","summary":"out.md 파일 저장이 완료되었습니다."}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )
    runner.conv.add_user("players.json에서 score가 15 이상인 player 이름과 평균 score를 알려줘.")
    runner.conv.add_assistant("B, C의 평균 score는 25입니다.")

    result = runner.run_turn("방금 필터된 결과를 score 내림차순 마크다운 표로 out.md에 저장해줘.")

    assert result.summary == "out.md 파일 저장이 완료되었습니다."
    assert ws.joinpath("out.md").read_text() == (
        "| name | score |\n| --- | --- |\n| B | 30 |\n| C | 20 |\n"
    )
    assert runner.deps.llm.calls == 3


def test_outside_workspace_write_call_is_denied_by_policy(tmp_path):
    reg = ToolRegistry()
    runner = AgentRunner(
        RunnerDeps(
            llm=FakeLLMClient(
                replies=[
                    (
                        '{"action":"call_tool","name":"writeFile","input":'
                        '{"path":"../events-sorted.csv","content":"x"}}'
                    ),
                    '{"action":"finish","summary":"denied"}',
                ]
            ),
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn("events.csv를 정렬해서 ../events-sorted.csv에 저장해줘.")

    assert result.summary == "denied"
    assert any("정책상 거부됨: out_of_workspace" in o for o in result.observations)
    assert runner.deps.llm.calls == 2


def test_outside_workspace_denial_does_not_stop_next_turn(tmp_path):
    reg = ToolRegistry()
    llm = FakeLLMClient(
        replies=[
            (
                '{"action":"call_tool","name":"writeFile","input":'
                '{"path":"../events-sorted.csv","content":"x"}}'
            ),
            '{"action":"finish","summary":"denied"}',
            '{"action":"finish","summary":"next turn handled"}',
        ]
    )
    runner = AgentRunner(
        RunnerDeps(
            llm=llm,
            registry=reg,
            ask=lambda *a: "y",
            log_dir=tmp_path,
        ),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    denied = runner.run_turn("events.csv를 정렬해서 ../events-sorted.csv에 저장해줘.")
    next_result = runner.run_turn("world.json 을 읽고 md 파일에 mermaid 로 표현해줘.")

    assert denied.summary == "denied"
    assert any("정책상 거부됨: out_of_workspace" in o for o in denied.observations)
    assert next_result.summary == "next turn handled"
    assert llm.calls == 3
    events = [json.loads(line) for line in runner.tracer.path.read_text().splitlines()]
    assert [event["kind"] for event in events if event["kind"] == "turn_start"] == [
        "turn_start",
        "turn_start",
    ]
    assert any(event["kind"] == "llm_call_start" for event in events)


def test_generated_tool_reuse_is_not_blocked_by_request_keywords(tmp_path):
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
    assert any("out.csv" in o for o in result.observations)


def test_generated_tool_digest_remains_visible_after_reuse(tmp_path):
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
    assert "sum-amount-by-type" in seen_digests[1]


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
