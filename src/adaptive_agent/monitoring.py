from __future__ import annotations

from typing import Any, Protocol


class Exporter(Protocol):
    def export(self, event: dict[str, Any]) -> None: ...


class NoopExporter:
    def export(self, event: dict[str, Any]) -> None:
        return None


def get_exporter(mode: str) -> Exporter:
    if mode == "langfuse":
        try:
            from .langfuse_exporter import LangfuseExporter  # type: ignore[import-untyped]  # optional extra
            return LangfuseExporter()
        except Exception:
            return NoopExporter()
    return NoopExporter()
