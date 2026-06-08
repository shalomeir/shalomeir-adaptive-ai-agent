from adaptive_agent.llm import FakeLLMClient
from adaptive_agent.schemas import Message


def test_fake_returns_scripted_replies():
    client = FakeLLMClient(replies=['{"action":"respond","text":"hi"}',
                                    '{"action":"finish"}'])
    out1 = client.chat([Message(role="user", content="x")], digests=[])
    out2 = client.chat([Message(role="user", content="y")], digests=[])
    assert "respond" in out1
    assert "finish" in out2


def test_fake_records_calls():
    client = FakeLLMClient(replies=['{"action":"finish"}'])
    client.chat([Message(role="user", content="x")], digests=[])
    assert client.calls == 1
