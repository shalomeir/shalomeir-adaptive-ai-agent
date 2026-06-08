from adaptive_agent.tools.builtins import build_file_tools


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
