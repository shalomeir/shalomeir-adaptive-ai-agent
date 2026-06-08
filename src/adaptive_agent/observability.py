from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console

from .monitoring import Exporter, NoopExporter

_ALLOWED = {
    "durationMs", "model", "inputTokens", "outputTokens", "cacheHit", "actionType",
    "parseOk", "retries", "toolName", "exitCode", "timedOut", "outputBytes",
    "truncated", "policy", "policyReason", "verifyPassed", "verifyReason",
    "fixIteration", "errorKind", "message",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tracer:
    def __init__(self, log_dir: Path | str, console: Console | None = None,
                 exporter: Exporter | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "events.jsonl"
        self.console = console or Console()
        # Local JSONL is the always-on path; the exporter is the seam for an
        # external monitor (e.g. Langfuse). Default no-op keeps it inert.
        self.exporter = exporter or NoopExporter()
        self._trace_id: str | None = None
        self._span_id: str | None = None

    @contextmanager
    def trace(self) -> Iterator[str]:
        self._trace_id = uuid.uuid4().hex
        try:
            yield self._trace_id
        finally:
            self._trace_id = None

    @contextmanager
    def span(self) -> Iterator[str]:
        prev = self._span_id
        self._span_id = uuid.uuid4().hex
        try:
            yield self._span_id
        finally:
            self._span_id = prev

    def log(self, kind: str, **fields: Any) -> None:
        evt: dict[str, Any] = {
            "ts": _now(),
            "traceId": self._trace_id or "no-trace",
            "spanId": self._span_id or uuid.uuid4().hex,
            "parentSpanId": None,
            "kind": kind,
        }
        for key, value in fields.items():
            if key in _ALLOWED:
                evt[key] = value
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
        # Forwarding must never break the core path, so failures are swallowed.
        try:
            self.exporter.export(evt)
        except Exception:
            pass

    def info(self, message: str) -> None:
        self.console.print(message)
