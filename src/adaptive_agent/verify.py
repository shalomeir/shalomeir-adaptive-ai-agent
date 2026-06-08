from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Verify:
    """Result of a single verification check."""

    passed: bool
    reason: str = ""


def verify_file_exists(path: Path | str, must_contain: str | None = None) -> Verify:
    """Check that *path* exists and optionally contains *must_contain*."""
    p = Path(path)
    if not p.exists():
        return Verify(False, f"파일이 존재하지 않습니다: {p}")
    if must_contain is not None and must_contain not in p.read_text("utf-8", "ignore"):
        return Verify(False, f"기대한 내용이 없습니다: {must_contain!r}")
    return Verify(True)


def verify_row_count(rows: list[object], expected: int) -> Verify:
    """Check that *rows* has exactly *expected* elements."""
    if len(rows) != expected:
        return Verify(False, f"행 수가 다릅니다. 기대 {expected}, 실제 {len(rows)}")
    return Verify(True)
