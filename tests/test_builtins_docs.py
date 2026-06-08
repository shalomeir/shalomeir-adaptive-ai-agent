from adaptive_agent.tools.builtins import build_search_docs, build_ask_user


def test_search_docs_finds_term(tmp_path):
    (tmp_path / "a.md").write_text("object tree schema: Entity has health")
    (tmp_path / "b.md").write_text("unrelated content")
    tool = build_search_docs(docs_dir=tmp_path)
    res = tool.handler({"query": "health", "limit": 5})
    ids = [r["docId"] for r in res.output["results"]]
    assert "a.md" in ids
    assert "b.md" not in ids


def test_ask_user_uses_callback():
    tool = build_ask_user(ask=lambda q, choices: "events.csv")
    res = tool.handler({"question": "which file?"})
    assert res.output["answer"] == "events.csv"
