from adaptive_agent import llm
from adaptive_agent.llm import AnthropicMessagesClient, FakeLLMClient, HttpLLMClient
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


def test_http_client_omits_temperature_for_default_sampling_models(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(dict(json))
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"action":"finish","summary":"ok"}'}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = HttpLLMClient("https://api.openai.com/v1", "gpt-5.5")

    result = client.chat([Message(role="user", content="x")], digests=[])

    assert result == '{"action":"finish","summary":"ok"}'
    assert "response_format" in payloads[0]
    assert "temperature" not in payloads[0]
    assert len(payloads) == 1


def test_http_client_falls_back_when_unknown_model_rejects_temperature(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(dict(json))
        if "temperature" in json:
            return _FakeResponse(
                400,
                {
                    "error": {
                        "message": "Unsupported value: 'temperature' does not support 0.",
                        "param": "temperature",
                        "code": "unsupported_value",
                    }
                },
            )
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"action":"finish","summary":"ok"}'}}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = HttpLLMClient("https://api.openai.com/v1", "new-reasoning-model")

    result = client.chat([Message(role="user", content="x")], digests=[])

    assert result == '{"action":"finish","summary":"ok"}'
    assert "response_format" in payloads[0]
    assert "temperature" in payloads[0]
    assert "response_format" in payloads[1]
    assert "temperature" not in payloads[1]


def test_anthropic_messages_client_uses_native_messages_api(monkeypatch):
    requests = []

    def fake_post(url, json, headers, timeout):
        requests.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse(
            200,
            {
                "content": [
                    {"type": "text", "text": '{"action":"finish",'},
                    {"type": "text", "text": '"summary":"ok"}'},
                ]
            },
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = AnthropicMessagesClient("https://api.anthropic.com/v1", "claude-sonnet", "key")

    result = client.chat(
        [
            Message(role="system", content="STRICT OUTPUT CONTRACT"),
            Message(role="user", content="task"),
            Message(role="tool", content="도구 결과"),
        ],
        digests=[
            ToolDigest(
                name="readFile",
                origin="builtin",
                description="read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
    )

    request = requests[0]
    assert result == '{"action":"finish","summary":"ok"}'
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["headers"]["x-api-key"] == "key"
    assert request["headers"]["anthropic-version"] == llm.ANTHROPIC_VERSION
    assert request["json"]["model"] == "claude-sonnet"
    assert "temperature" not in request["json"]
    assert request["json"]["max_tokens"] == llm.ANTHROPIC_DEFAULT_MAX_TOKENS
    assert "response_format" not in request["json"]
    assert request["json"]["system"].startswith("STRICT OUTPUT CONTRACT")
    assert "사용 가능한 도구" in request["json"]["system"]
    assert request["json"]["messages"] == [
        {"role": "user", "content": "task"},
        {"role": "user", "content": "도구 결과"},
    ]


def test_anthropic_messages_client_omits_sampling_params_without_retry(monkeypatch):
    requests = []

    def fake_post(url, json, headers, timeout):
        requests.append({"url": url, "json": dict(json), "headers": headers, "timeout": timeout})
        return _FakeResponse(
            200,
            {"content": [{"type": "text", "text": '{"action":"finish","summary":"ok"}'}]},
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    client = AnthropicMessagesClient("https://api.anthropic.com/v1", "claude-opus-4-8", "key")

    result = client.chat([Message(role="user", content="안녕.")], digests=[])

    assert result == '{"action":"finish","summary":"ok"}'
    assert "temperature" not in requests[0]["json"]
    assert len(requests) == 1
