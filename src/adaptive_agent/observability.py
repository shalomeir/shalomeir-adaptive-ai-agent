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
    "durationMs",
    "model",
    "inputTokens",
    "outputTokens",
    "cacheHit",
    "actionType",
    "parseOk",
    "retries",
    "toolName",
    "exitCode",
    "timedOut",
    "outputBytes",
    "truncated",
    "policy",
    "policyReason",
    "verifyPassed",
    "verifyReason",
    "fixIteration",
    "errorKind",
    "message",
    "responsePreview",
    "responseChars",
    "responseTruncated",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tracer:
    def __init__(
        self, log_dir: Path | str, console: Console | None = None, exporter: Exporter | None = None
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "events.jsonl"
        self.console = console or Console()
        # Local JSONL is the always-on path; the exporter is the seam for an
        # external monitor (e.g. Langfuse). Default no-op keeps it inert.
        self.exporter = exporter or NoopExporter()
        self._trace_id: str | None = None
        # Span ids are kept as a stack so each event can record its parent.
        self._span_stack: list[str] = []

    @contextmanager
    def trace(self) -> Iterator[str]:
        self._trace_id = uuid.uuid4().hex
        try:
            yield self._trace_id
        finally:
            self._trace_id = None

    @contextmanager
    def span(self) -> Iterator[str]:
        span_id = uuid.uuid4().hex
        self._span_stack.append(span_id)
        try:
            yield span_id
        finally:
            self._span_stack.pop()

    def log(self, kind: str, **fields: Any) -> None:
        # The current span is the stack top; its parent is the one below it.
        # Events emitted outside any span sit directly under the trace (no parent).
        span_id = self._span_stack[-1] if self._span_stack else uuid.uuid4().hex
        parent_span_id = self._span_stack[-2] if len(self._span_stack) >= 2 else None
        evt: dict[str, Any] = {
            "ts": _now(),
            "traceId": self._trace_id or "no-trace",
            "spanId": span_id,
            "parentSpanId": parent_span_id,
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
