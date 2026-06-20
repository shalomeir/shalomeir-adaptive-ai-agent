from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..schemas import ToolDigest


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    origin: Literal["builtin", "generated", "mcp"]
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    output_schema: dict[str, Any] | None = None

    def digest(self) -> ToolDigest:
        return ToolDigest(
            name=self.name,
            description=self.description,
            origin=self.origin,
            inputSchema=self.input_schema,
        )
