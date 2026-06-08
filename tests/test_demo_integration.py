import csv
import json
import shutil
from pathlib import Path

from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.policy import PolicyManager
from adaptive_agent.runner import AgentRunner, RunnerDeps
from adaptive_agent.sandbox import ExecutionSandbox
from adaptive_agent.skills import SkillStore
from adaptive_agent.tools.builtins import (
    build_file_tools,
    build_run_python,
    build_search_docs,
)
from adaptive_agent.tools.generated import GeneratedToolManager
from adaptive_agent.tools.registry import ToolRegistry

DEMORSC = Path(__file__).resolve().parents[1] / "demorsc"


def _make_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    return ws


def _runner(
    tmp_path: Path,
    ws: Path,
    replies: list[str],
    ask: str = "y",
    skills_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> AgentRunner:
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    if docs_dir is not None:
        reg.register(build_search_docs(docs_dir))
    sandbox = ExecutionSandbox(ws, timeout_sec=10, max_output_bytes=16384)
    reg.register(build_run_python(sandbox))
    deps = RunnerDeps(
        llm=FakeLLMClient(replies=replies),
        registry=reg,
        ask=lambda *a: ask,
        log_dir=tmp_path / "logs",
    )
    return AgentRunner(
        deps,
        generated=GeneratedToolManager(ws / ".session", sandbox),
        skills=SkillStore(skills_dir or (tmp_path / "skills")),
        policy=PolicyManager(ask=lambda q: ask),
    )


def _create(name: str, code: str) -> str:
    return json.dumps(
        {
            "action": "create_tool",
            "spec": {
                "name": name,
                "description": name,
                "code": code,
                "inputSchema": {"type": "object"},
            },
        }
    )


def _call(name: str, payload: dict) -> str:  # type: ignore[type-arg]
    return json.dumps({"action": "call_tool", "name": name, "input": payload})


def _finish(summary: str = "done") -> str:
    return json.dumps({"action": "finish", "summary": summary})


def _update(name: str, code: str) -> str:
    return json.dumps({"action": "update_tool", "name": name, "code": code})


def _ask(question: str) -> str:
    return json.dumps({"action": "ask_user", "question": question})


def _log_kinds(tmp_path: Path) -> list[str]:
    lines = (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["kind"] for line in lines]


# ---------- D1: JSON query (read-only) ----------
def test_d1_json_query(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "monsters.json", ws / "monsters.json")
    code = (
        "import json\n"
        f"def run(input):\n"
        f"    data = json.load(open(r'{ws}/monsters.json'))['monsters']\n"
        "    sel = [m for m in data if m['hp'] >= 100]\n"
        "    avg = round(sum(m['hp'] for m in sel) / len(sel), 2)\n"
        "    return {'names': [m['name'] for m in sel], 'avg': avg}\n"
    )
    runner = _runner(
        tmp_path,
        ws,
        [
            _create("hp-filter", code),
            _call("hp-filter", {}),
            _finish("5"),
        ],
        ask="n",
    )
    result = runner.run_turn("hp>=100 names and average")
    blob = " ".join(result.observations)
    assert "Orc" in blob and "Dragon" in blob and "Wolf" in blob
    assert "186.67" in blob


def test_d1_live_prompt_uses_general_python_tool_path(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "monsters.json", ws / "monsters.json")
    code = (
        "import json\n"
        "data = json.load(open('monsters.json'))\n"
        "records = data['monsters'] if isinstance(data, dict) else data\n"
        "selected = [m for m in records if m.get('hp', 0) >= 100]\n"
        "avg = round(sum(m['hp'] for m in selected) / len(selected), 2)\n"
        "print(json.dumps({'names': [m['name'] for m in selected], 'avg': avg}, ensure_ascii=False))\n"
    )
    runner = _runner(
        tmp_path,
        ws,
        [
            _call("runPython", {"code": code}),
            _finish("Orc, Dragon, Wolf 평균 hp 186.67"),
        ],
        ask="n",
    )

    result = runner.run_turn(
        "workspace의 monsters.json에서 hp가 100 이상인 몬스터 이름과 평균 hp를 알려줘."
    )

    assert "Orc" in result.summary
    assert "Dragon" in result.summary
    assert "Wolf" in result.summary
    assert "186.67" in result.summary
    assert runner.deps.llm.calls == 2


# ---------- D2 + D5: dedup/sort, persist, reuse ----------
NORMALIZE_CODE = (
    "import csv\n"
    "def run(input):\n"
    "    with open(input['src'], newline='') as f:\n"
    "        rows = list(csv.reader(f))\n"
    "    header, body = rows[0], rows[1:]\n"
    "    seen = set(); uniq = []\n"
    "    for r in body:\n"
    "        k = tuple(r)\n"
    "        if k not in seen:\n"
    "            seen.add(k); uniq.append(r)\n"
    "    di = header.index('date')\n"
    "    uniq.sort(key=lambda r: r[di])\n"
    "    with open(input['dst'], 'w', newline='') as f:\n"
    "        w = csv.writer(f); w.writerow(header); w.writerows(uniq)\n"
    "    return {'rows': len(uniq)}\n"
)
PANDAS_BAD_NORMALIZE_CODE = (
    "import pandas as pd\n"
    "def run(input):\n"
    "    df = pd.read_csv(input['src']).drop_duplicates().sort_values('date')\n"
    "    df.to_csv(input['dst'], index=False)\n"
    "    return {'rows': len(df)}\n"
)


def _data_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]


def test_d2_persist_then_d5_reuse(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    # --- D2 session: create + run + persist ---
    ws_a = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws_a / "events.csv")
    out1 = ws_a / "out1.csv"
    runner_a = _runner(
        tmp_path / "a",
        ws_a,
        [
            _create("normalize-csv", NORMALIZE_CODE),
            _call("normalize-csv", {"src": "events.csv", "dst": "out1.csv"}),
            _finish(),
        ],
        ask="y",
        skills_dir=skills_dir,
    )
    runner_a.run_turn("dedup and sort events.csv")
    rows1 = _data_rows(out1)
    assert len(rows1) == 5
    dates1 = [r[1] for r in rows1]
    assert dates1 == sorted(dates1)
    assert (skills_dir / "normalize-csv" / "manifest.json").exists()

    # --- D5 session: new runner reloads the skill, reuse WITHOUT create ---
    ws_b = _make_ws(tmp_path / "bdir")
    shutil.copy(DEMORSC / "data" / "events2.csv", ws_b / "events2.csv")
    out2 = ws_b / "out2.csv"
    runner_b = _runner(
        tmp_path / "b",
        ws_b,
        [
            _call("normalize-csv", {"src": str(ws_b / "events2.csv"), "dst": str(out2)}),
            _finish(),
        ],
        ask="y",
        skills_dir=skills_dir,
    )
    # reused tool is registered from the persisted skill at init
    assert any(d.name == "normalize-csv" for d in runner_b.deps.registry.digests())
    runner_b.run_turn("dedup and sort events2.csv")
    rows2 = _data_rows(out2)
    assert len(rows2) == 3
    assert [r[0] for r in rows2] == ["a", "b", "c"]


def test_d2_live_prompt_uses_general_python_then_write_file(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws / "events.csv")
    content = (
        "id,date,type,amount\n"
        "1,2026-01-15,signup,0\n"
        "4,2026-01-15,refund,-200\n"
        "2,2026-02-20,purchase,800\n"
        "3,2026-03-02,purchase,1200\n"
        "5,2026-04-10,purchase,500\n"
    )
    code = (
        "import csv, io, json\n"
        "rows = list(csv.reader(open('events.csv', newline='')))\n"
        "header, body = rows[0], rows[1:]\n"
        "seen = set(); unique = []\n"
        "for row in body:\n"
        "    key = tuple(row)\n"
        "    if key not in seen:\n"
        "        seen.add(key); unique.append(row)\n"
        "unique.sort(key=lambda row: row[header.index('date')])\n"
        "buf = io.StringIO(); writer = csv.writer(buf); writer.writerow(header); writer.writerows(unique)\n"
        "print(json.dumps({'content': buf.getvalue(), 'rows': len(unique)}, ensure_ascii=False))\n"
    )
    runner = _runner(
        tmp_path,
        ws,
        [
            _call("runPython", {"code": code}),
            _call("writeFile", {"path": "events-clean.csv", "content": content}),
            _finish("events-clean.csv 저장 완료"),
        ],
        ask="y",
    )

    result = runner.run_turn(
        "events.csv에서 완전히 중복된 행을 제거하고 date 기준 오름차순으로 정렬해서 "
        "events-clean.csv로 저장해줘."
    )

    assert "events-clean.csv" in result.summary
    assert runner.deps.llm.calls == 3
    rows = _data_rows(ws / "events-clean.csv")
    assert len(rows) == 5
    assert [row[1] for row in rows] == sorted(row[1] for row in rows)
    assert [row[0] for row in rows] == ["1", "4", "2", "3", "5"]


# ---------- D6: object tree grounding + state manip + verify ----------
def test_d6_object_tree(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "world" / "world.json", ws / "world.json")
    code = (
        "import json\n"
        f"def run(input):\n"
        f"    tree = json.load(open(r'{ws}/world.json'))\n"
        "    def prune(node):\n"
        "        kept = []\n"
        "        for c in node.get('children', []):\n"
        "            if c['type'] == 'Entity' and c['props'].get('health', 0) < 100:\n"
        "                continue\n"
        "            prune(c); kept.append(c)\n"
        "        node['children'] = kept\n"
        "    prune(tree['root'])\n"
        "    ents = []\n"
        "    def collect(node):\n"
        "        if node['type'] == 'Entity': ents.append(node['props']['health'])\n"
        "        for c in node.get('children', []): collect(c)\n"
        "    collect(tree['root'])\n"
        f"    json.dump(tree, open(r'{ws}/world.json', 'w'))\n"
        "    return {'count': len(ents), 'avg': round(sum(ents) / len(ents), 2)}\n"
    )
    runner = _runner(
        tmp_path,
        ws,
        [
            _call("searchDocs", {"query": "health"}),
            _create("prune", code),
            _call("prune", {}),
            _finish(),
        ],
        ask="y",
        docs_dir=DEMORSC / "docs",
    )
    result = runner.run_turn("remove entities with health<100 and report average")
    blob = " ".join(result.observations)
    assert "190" in blob and "'count': 3" in blob.replace('"', "'")
    # verify resulting state by re-reading the tree
    tree = json.loads((ws / "world.json").read_text())
    healths: list[int] = []
    ids: list[str] = []

    def walk(n: dict) -> None:  # type: ignore[type-arg]
        ids.append(n["id"])
        if n["type"] == "Entity":
            healths.append(n["props"]["health"])
        for c in n.get("children", []):
            walk(c)

    walk(tree["root"])
    assert len(healths) == 3
    assert all(h >= 100 for h in healths)
    assert "e1" not in ids and "rock1" not in ids
    assert tree["root"]["type"] == "Scene"
    # searchDocs grounding happened (observation carries a doc result)
    assert "object-tree" in blob or "Entity" in blob


# ---------- D7: out-of-workspace write is DENIED ----------
def test_d7_out_of_workspace_denied(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    runner = _runner(
        tmp_path,
        ws,
        [
            _call("writeFile", {"path": "../events-sorted.csv", "content": "x"}),
            _finish(),
        ],
        ask="y",
    )
    result = runner.run_turn("save outside workspace")
    assert any("거부" in o for o in result.observations)
    assert not (ws.parent / "events-sorted.csv").exists()


# ---------- D8: multi-turn + gated writeFile approved ----------
def test_d8_multiturn_gated_write(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "monsters.json", ws / "monsters.json")
    code = (
        "import json\n"
        f"def run(input):\n"
        f"    data = json.load(open(r'{ws}/monsters.json'))['monsters']\n"
        "    sel = [m for m in data if m['hp'] >= 100]\n"
        "    return {'names': [m['name'] for m in sel]}\n"
    )
    table = "| name | hp |\n| --- | --- |\n| Dragon | 300 |\n| Orc | 150 |\n| Wolf | 110 |\n"
    runner = _runner(
        tmp_path,
        ws,
        [
            _create("hp-filter", code),
            _call("hp-filter", {}),
            _finish("filtered"),
            _call("writeFile", {"path": "table.md", "content": table}),
            _finish("saved"),
        ],
        ask="y",
    )
    runner.run_turn("filter hp>=100")
    runner.run_turn("save the filtered set as a markdown table sorted by hp desc")
    md = (ws / "table.md").read_text()
    assert md.index("Dragon") < md.index("Orc") < md.index("Wolf")


# ---------- D3: failure observed, then self-fix closes the loop ----------
# The first version sums the raw string column, so the sandboxed run raises a
# TypeError. The agent reads the error and rewrites the tool with int() + dedup.
SUM_BAD_CODE = (
    "import csv\n"
    "def run(input):\n"
    "    with open(input['src'], newline='') as f:\n"
    "        rows = list(csv.reader(f))[1:]\n"
    "    sums = {}\n"
    "    for r in rows:\n"
    "        sums[r[2]] = sums.get(r[2], 0) + r[3]\n"
    "    return sums\n"
)
SUM_GOOD_CODE = (
    "import csv\n"
    "def run(input):\n"
    "    with open(input['src'], newline='') as f:\n"
    "        rows = list(csv.reader(f))[1:]\n"
    "    seen = set(); uniq = []\n"
    "    for r in rows:\n"
    "        k = tuple(r)\n"
    "        if k not in seen:\n"
    "            seen.add(k); uniq.append(r)\n"
    "    sums = {}\n"
    "    for r in uniq:\n"
    "        sums[r[2]] = sums.get(r[2], 0) + int(r[3])\n"
    "    return sums\n"
)


def test_d3_failure_then_self_fix(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws / "events.csv")
    src = str(ws / "events.csv")
    runner = _runner(
        tmp_path,
        ws,
        [
            _create("sum-by-type", SUM_BAD_CODE),
            _call("sum-by-type", {"src": src}),  # raises TypeError → failure
            _update("sum-by-type", SUM_GOOD_CODE),
            _call("sum-by-type", {"src": src}),  # now correct
            _finish(),
        ],
        ask="n",
    )
    result = runner.run_turn("sum amount by type, counting full duplicates once")
    blob = " ".join(result.observations).replace('"', "'")
    # the loop saw a failure before it produced a correct result
    assert any("실패" in o for o in result.observations)
    assert "'purchase': 2500" in blob
    assert "'signup': 0" in blob
    assert "'refund': -200" in blob
    # the self-fix went through update_tool, recorded as a tool_update event
    assert "tool_update" in _log_kinds(tmp_path)


def test_d3_live_prompt_uses_general_python_without_write_prompt(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws / "events.csv")
    code = (
        "import csv, json\n"
        "rows = list(csv.reader(open('events.csv', newline='')))[1:]\n"
        "seen = set(); unique = []\n"
        "for row in rows:\n"
        "    key = tuple(row)\n"
        "    if key not in seen:\n"
        "        seen.add(key); unique.append(row)\n"
        "sums = {}\n"
        "for row in unique:\n"
        "    sums[row[2]] = sums.get(row[2], 0) + int(row[3])\n"
        "print(json.dumps(sums, ensure_ascii=False))\n"
    )
    runner = _runner(
        tmp_path,
        ws,
        [
            _call("runPython", {"code": code}),
            _finish("purchase 2500, signup 0, refund -200"),
        ],
        ask="n",
    )

    result = runner.run_turn(
        "events.csv에서 완전히 중복된 행은 한 번만 세고, amount 합계를 type별로 구해줘."
    )

    assert "purchase 2500" in result.summary
    assert "signup 0" in result.summary
    assert "refund -200" in result.summary
    assert runner.deps.llm.calls == 2


# ---------- D4: ambiguous request asks first, acts only after clarification ----------
def test_d4_ambiguous_then_clarified(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws / "events.csv")
    out = ws / "out.csv"
    runner = _runner(
        tmp_path,
        ws,
        [
            # turn 1: too vague to act → ask, then yield the turn
            _ask("어떤 데이터를 어떤 기준으로 정리할까요?"),
            _finish("waiting for clarification"),
            # turn 2: concrete request → dedup + sort like D2
            _create("normalize-csv", NORMALIZE_CODE),
            _call("normalize-csv", {"src": "events.csv", "dst": "out.csv"}),
            _finish(),
        ],
        ask="events.csv를 중복 제거하고 date로 정렬",
    )

    # turn 1 must not build a tool or write any file
    runner.run_turn("데이터 좀 정리해줘")
    kinds_after_turn1 = _log_kinds(tmp_path)
    assert "tool_create" not in kinds_after_turn1
    assert not out.exists()
    events = (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    assert any(json.loads(line).get("actionType") == "ask_user" for line in events)

    # turn 2 produces the same result as D2
    runner.run_turn("events.csv에서 중복 행 제거하고 date로 정렬해줘")
    rows = _data_rows(out)
    assert len(rows) == 5
    dates = [r[1] for r in rows]
    assert dates == sorted(dates)


# ---------- D9: external package detour is blocked, then stdlib fallback works ----------
def test_d9_blocks_package_install_prompt_and_falls_back_to_csv(tmp_path: Path) -> None:
    ws = _make_ws(tmp_path)
    shutil.copy(DEMORSC / "data" / "events.csv", ws / "events.csv")
    out = ws / "events-clean.csv"
    asks: list[tuple] = []
    reg = ToolRegistry()
    for tool in build_file_tools(ws):
        reg.register(tool)
    sandbox = ExecutionSandbox(ws, timeout_sec=10, max_output_bytes=16384)
    deps = RunnerDeps(
        llm=FakeLLMClient(
            replies=[
                _create("normalize-csv", PANDAS_BAD_NORMALIZE_CODE),
                _call("normalize-csv", {"src": "events.csv", "dst": "events-clean.csv"}),
                _ask("Pandas 모듈이 설치되어 있지 않습니다. 설치하시겠습니까?"),
                _ask("Pandas를 사용할 수 없습니다. 대신 표준 라이브러리로 진행해도 될까요?"),
                _update("normalize-csv", NORMALIZE_CODE),
                _call("normalize-csv", {"src": "events.csv", "dst": "events-clean.csv"}),
                _finish("saved"),
            ]
        ),
        registry=reg,
        ask=lambda *a: asks.append(a) or "no",
        log_dir=tmp_path / "logs",
    )
    runner = AgentRunner(
        deps,
        generated=GeneratedToolManager(ws / ".session", sandbox),
        skills=SkillStore(tmp_path / "skills"),
        policy=PolicyManager(ask=lambda q: "y"),
    )

    result = runner.run_turn("events.csv dedup, sort by date, save to events-clean.csv")

    assert result.summary == "saved"
    assert asks == []
    assert out.exists()
    rows = _data_rows(out)
    assert len(rows) == 5
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert "tool_update" in _log_kinds(tmp_path)
    assert any("패키지 설치 질문" in o for o in result.observations)
