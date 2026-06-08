from __future__ import annotations

from typing import Any

from ..schemas import ToolDigest, normalize_tool_name
from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return self._resolve(name) is not None

    def get(self, name: str) -> Tool | None:
        return self._resolve(name)

    def _resolve(self, name: str) -> Tool | None:
        """Find a tool by exact name, falling back to kebab-normalized match.

        The model may call a tool in a different casing than it was registered
        (e.g. ``filterMonsters`` vs ``filter-monsters``). Exact match wins first
        so built-in camelCase names stay fast; the normalized pass recovers the
        rest instead of failing with "tool not found".
        """
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        target = normalize_tool_name(name)
        for registered_name, registered_tool in self._tools.items():
            if normalize_tool_name(registered_name) == target:
                return registered_tool
        return None

    def digests(self) -> list[ToolDigest]:
        return [t.digest() for t in self._tools.values()]

    def call(self, name: str, payload: dict[str, Any]) -> ToolResult:
        tool = self._resolve(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error=f"'{name}' 도구가 없습니다. 도구 목록을 다시 보거나 create_tool로 만드세요.",
            )
        try:
            return tool.handler(payload)
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
