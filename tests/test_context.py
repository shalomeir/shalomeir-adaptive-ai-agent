from adaptive_agent.conversation import ConversationStore
from adaptive_agent.context import ContextManager


def test_carry_over_preserved_after_compaction():
    conv = ConversationStore(system="rules")
    for i in range(10):
        conv.add_user(f"msg {i}")
        conv.add_assistant(f"reply {i}")
    cm = ContextManager(token_threshold=10, summarize=lambda msgs: "SUMMARY")
    cm.carry_over_fact("open task: finish report")
    cm.maybe_compact(conv)
    rendered = "\n".join(m.content for m in conv.messages())
    assert "SUMMARY" in rendered
    assert "open task: finish report" in rendered
    assert conv.messages()[0].content == "rules"


def test_no_compaction_below_threshold():
    conv = ConversationStore(system="rules")
    conv.add_user("hi")
    cm = ContextManager(token_threshold=10_000, summarize=lambda msgs: "SUMMARY")
    cm.maybe_compact(conv)
    assert all("SUMMARY" not in m.content for m in conv.messages())
