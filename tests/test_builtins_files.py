import csv

from adaptive_agent.tools.builtins import build_file_tools, build_normalize_csv


def test_write_then_read(tmp_path):
    tools = {t.name: t for t in build_file_tools(workspace=tmp_path)}
    w = tools["writeFile"].handler({"path": "a.txt", "content": "hello"})
    assert w.ok
    r = tools["readFile"].handler({"path": "a.txt"})
    assert r.output["content"] == "hello"


def test_path_escape_blocked(tmp_path):
    tools = {t.name: t for t in build_file_tools(workspace=tmp_path)}
    r = tools["readFile"].handler({"path": "../secret.txt"})
    assert not r.ok
    assert "workspace" in r.error.lower()


def test_list_files(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    tools = {t.name: t for t in build_file_tools(workspace=tmp_path)}
    res = tools["listFiles"].handler({"path": "."})
    names = [e["path"] for e in res.output["entries"]]
    assert "x.txt" in names


def test_list_files_with_symlinked_workspace(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    tools = {t.name: t for t in build_file_tools(workspace=link)}
    tools["writeFile"].handler({"path": "x.txt", "content": "hi"})
    res = tools["listFiles"].handler({"path": "."})
    assert res.ok
    assert "x.txt" in [e["path"] for e in res.output["entries"]]


def test_normalize_csv_dedupes_and_sorts(tmp_path):
    (tmp_path / "events.csv").write_text(
        "id,date,type,amount\n"
        "3,2026-03-02,purchase,1200\n"
        "1,2026-01-15,signup,0\n"
        "2,2026-02-20,purchase,800\n"
        "2,2026-02-20,purchase,800\n"
        "4,2026-01-15,refund,-200\n",
        encoding="utf-8",
    )
    tool = build_normalize_csv(tmp_path)

    res = tool.handler({"src": "events.csv", "dst": "events-clean.csv", "sortBy": "date"})

    assert res.ok
    with (tmp_path / "events-clean.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    assert len(rows) == 4
    assert [row[1] for row in rows] == sorted(row[1] for row in rows)
