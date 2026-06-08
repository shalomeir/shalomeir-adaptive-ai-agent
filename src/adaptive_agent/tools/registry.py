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

    def prepare_call(self, name: str, payload: dict[str, Any]) -> tuple[Tool | None, str | None]:
        tool = self._resolve(name)
        if tool is None:
            return None, f"'{name}' 도구가 없습니다. 도구 목록을 다시 보거나 create_tool로 만드세요."
        error = self._validate_payload(tool, payload)
        if error is not None:
            return tool, error
        return tool, None

    def call(self, name: str, payload: dict[str, Any]) -> ToolResult:
        tool, error = self.prepare_call(name, payload)
        if tool is None:
            assert error is not None
            return ToolResult(ok=False, error=error)
        if error is not None:
            return ToolResult(ok=False, error=error)
        try:
            return tool.handler(payload)
        except Exception as e:
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")

    def _validate_payload(self, tool: Tool, payload: dict[str, Any]) -> str | None:
        schema = tool.input_schema
        if schema.get("type") == "object" and not isinstance(payload, dict):
            return f"{tool.name} input은 object여야 합니다."
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [field for field in required if field not in payload]
            if missing:
                return f"{tool.name} input에 필수 필드가 없습니다: {', '.join(missing)}"
        return None
