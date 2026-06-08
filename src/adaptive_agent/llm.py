from __future__ import annotations

import json
from typing import Protocol

import httpx

from .schemas import Message, ToolDigest


class LLMClient(Protocol):
    """Structural interface for any LLM backend."""

    def chat(self, messages: list[Message], digests: list[ToolDigest]) -> str: ...


def _render_system(digests: list[ToolDigest]) -> str:
    # Build a compact tool inventory so the model knows what is available.
    lines = [f"- {d.name} ({d.origin}): {d.description}" for d in digests]
    return "사용 가능한 도구:\n" + "\n".join(lines)


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
        payload_messages: list[dict[str, str]] = [
            {"role": "system", "content": _render_system(digests)}
        ]
        payload_messages += [
            # "tool" is not a standard OpenAI role; surface it as "user".
            {"role": m.role if m.role != "tool" else "user", "content": m.content}
            for m in messages
        ]
        headers: dict[str, str] = (
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": payload_messages, "temperature": 0},
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
