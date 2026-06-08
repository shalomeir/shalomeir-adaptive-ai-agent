from __future__ import annotations

from typing import Any

from ..schemas import ToolDigest
from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def digests(self) -> list[ToolDigest]:
        return [t.digest() for t in self._tools.values()]

    def call(self, name: str, payload: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"'{name}' 도구가 없습니다. "
                                              "도구 목록을 다시 보거나 create_tool로 만드세요.")
        try:
            return tool.handler(payload)
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
