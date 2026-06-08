from adaptive_agent.policy import PolicyManager


def test_read_is_allowed():
    pm = PolicyManager()
    assert pm.evaluate("read_file").decision == "ALLOW"


def test_write_asks_user():
    pm = PolicyManager()
    assert pm.evaluate("write_file").decision == "ASK_USER"


def test_persist_tool_asks_user():
    pm = PolicyManager()
    assert pm.evaluate("persist_tool").decision == "ASK_USER"


def test_out_of_workspace_is_denied():
    pm = PolicyManager()
    assert pm.evaluate("out_of_workspace").decision == "DENY"


def test_confirm_uses_callback():
    answers = iter(["y"])
    pm = PolicyManager(ask=lambda q: next(answers))
    assert pm.confirm("write_file") is True
