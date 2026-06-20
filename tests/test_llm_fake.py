from adaptive_agent import llm
from adaptive_agent.llm import FakeLLMClient, HttpLLMClient
from adaptive_agent.schemas import Message, ToolDigest


def test_fake_returns_scripted_replies():
    client = FakeLLMClient(replies=['{"action":"respond","text":"hi"}', '{"action":"finish"}'])
    out1 = client.chat([Message(role="user", content="x")], digests=[])
    out2 = client.chat([Message(role="user", content="y")], digests=[])
    assert "respond" in out1
    assert "finish" in out2


def test_fake_records_calls():
    client = FakeLLMClient(replies=['{"action":"finish"}'])
    client.chat([Message(role="user", content="x")], digests=[])
    assert client.calls == 1


class _FakeResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")

    def json(self):
        return self._content


def test_http_client_requests_json_object_response(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(json)
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"action":"finish","summary":"ok"}'}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = HttpLLMClient("http://localhost:11434/v1", "model")

    result = client.chat([Message(role="user", content="x")], digests=[])

    assert result == '{"action":"finish","summary":"ok"}'
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_http_client_merges_protocol_and_tools_into_one_system_message(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(json)
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"action":"finish","summary":"ok"}'}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = HttpLLMClient("http://localhost:11434/v1", "model")

    client.chat(
        [
            Message(role="system", content="STRICT OUTPUT CONTRACT"),
            Message(role="user", content="task"),
            Message(role="tool", content="도구 결과"),
        ],
        digests=[
            ToolDigest(
                name="csv-dedupe-sort",
                origin="generated",
                description="dedupe",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "output": {"type": "string"},
                    },
                    "required": ["source", "output"],
                },
            )
        ],
    )

    messages = payloads[0]["messages"]
    system_messages = [message for message in messages if message["role"] == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"].startswith("STRICT OUTPUT CONTRACT")
    assert "사용 가능한 도구" in system_messages[0]["content"]
    assert "csv-dedupe-sort" in system_messages[0]["content"]
    assert "input fields: source, output" in system_messages[0]["content"]
    assert "required: source, output" in system_messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "task"}
    assert messages[2] == {"role": "user", "content": "도구 결과"}


def test_http_client_falls_back_when_json_response_mode_is_unsupported(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(dict(json))
        if len(payloads) == 1:
            return _FakeResponse(400, {})
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"action":"finish","summary":"ok"}'}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = HttpLLMClient("http://localhost:11434/v1", "model")

    result = client.chat([Message(role="user", content="x")], digests=[])

    assert result == '{"action":"finish","summary":"ok"}'
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
