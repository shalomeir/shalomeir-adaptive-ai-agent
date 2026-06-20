from __future__ import annotations

import json
from typing import Protocol

import httpx

from .schemas import Message, ToolDigest


class LLMClient(Protocol):
    """Structural interface for any LLM backend."""

    def chat(self, messages: list[Message], digests: list[ToolDigest]) -> str: ...


def _schema_field_hint(schema: dict[str, object] | None) -> str:
    if not schema or schema.get("type") != "object":
        return ""
    properties = schema.get("properties")
    property_names = list(properties.keys()) if isinstance(properties, dict) else []
    required = schema.get("required")
    required_names = [str(item) for item in required] if isinstance(required, list) else []
    if property_names and required_names:
        return f" input fields: {', '.join(property_names)}; required: {', '.join(required_names)}"
    if property_names:
        return f" input fields: {', '.join(property_names)}"
    if required_names:
        return f" required input fields: {', '.join(required_names)}"
    return ""


def _render_tool_inventory(digests: list[ToolDigest]) -> str:
    # Build a compact tool inventory so the model knows what is available.
    lines = [
        f"- {d.name} ({d.origin}): {d.description}{_schema_field_hint(d.input_schema)}"
        for d in digests
    ]
    return "사용 가능한 도구:\n" + "\n".join(lines)


def _to_provider_message(message: Message) -> dict[str, str]:
    # "tool" is not a standard OpenAI role; surface it as "user".
    return {
        "role": message.role if message.role != "tool" else "user",
        "content": message.content,
    }


def _build_payload_messages(
    messages: list[Message], digests: list[ToolDigest]
) -> list[dict[str, str]]:
    """Merge protocol and tool inventory into one system message.

    Local models are more likely to drift when they receive multiple system
    messages. Keep the runner's protocol prompt first and append the tool list
    inside the same system message so the output contract stays dominant.
    """
    inventory = _render_tool_inventory(digests)
    if messages and messages[0].role == "system":
        system = {
            "role": "system",
            "content": f"{messages[0].content}\n\n{inventory}",
        }
        return [system, *[_to_provider_message(message) for message in messages[1:]]]
    return [{"role": "system", "content": inventory}, *[_to_provider_message(m) for m in messages]]


class HttpLLMClient:
    """Calls an OpenAI-compatible chat completions endpoint directly.

    No vendor SDK is required — a plain ``httpx`` POST is sufficient and keeps
    the client provider-agnostic (works with Ollama, vLLM, LiteLLM, etc.).
    """

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, timeout: float = 180
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[Message], digests: list[ToolDigest]) -> str:
        payload_messages = _build_payload_messages(messages, digests)
        headers: dict[str, str] = (
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        payload: dict[str, object] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code in {400, 422}:
            # Some OpenAI-compatible local servers do not implement response_format.
            payload.pop("response_format", None)
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        resp.raise_for_status()
        content: str = resp.json()["choices"][0]["message"]["content"]
        return content


class FakeLLMClient:
    """Deterministic test double that returns scripted replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages: list[Message], digests: list[ToolDigest]) -> str:
        self.calls += 1
        if not self._replies:
            return json.dumps({"action": "finish"})
        return self._replies.pop(0)
