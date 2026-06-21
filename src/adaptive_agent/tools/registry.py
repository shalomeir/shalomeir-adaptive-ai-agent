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
            return (
                None,
                f"'{name}' 도구가 없습니다. 도구 목록을 다시 보거나 create_tool로 만드세요.",
            )
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
        properties = schema.get("properties")
        property_names = list(properties) if isinstance(properties, dict) else []
        required = schema.get("required", [])
        if tool.origin == "generated" and not required and property_names:
            required = property_names
        if isinstance(required, list):
            missing = [field for field in required if field not in payload]
            if missing:
                return (
                    f"{tool.name} input에 필수 필드가 없습니다: {', '.join(missing)}. "
                    f"사용할 입력 필드: {', '.join(property_names or required)}"
                )
        if property_names and not any(field in payload for field in property_names):
            return (
                f"{tool.name} input 필드가 스키마와 맞지 않습니다. "
                f"사용할 입력 필드: {', '.join(property_names)}"
            )
        if isinstance(properties, dict):
            for field, value in payload.items():
                field_schema = properties.get(field)
                if not isinstance(field_schema, dict):
                    continue
                expected = field_schema.get("type")
                if expected is None or self._matches_json_schema_type(value, expected):
                    continue
                expected_text = (
                    ", ".join(str(item) for item in expected)
                    if isinstance(expected, list)
                    else str(expected)
                )
                return (
                    f"{tool.name} input 필드 '{field}' 타입이 스키마와 맞지 않습니다. "
                    f"기대 타입: {expected_text}."
                )
        return None

    def _matches_json_schema_type(self, value: Any, expected: Any) -> bool:
        if isinstance(expected, list):
            return any(self._matches_json_schema_type(value, item) for item in expected)
        if expected == "string":
            return isinstance(value, str)
        if expected == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "null":
            return value is None
        return True
