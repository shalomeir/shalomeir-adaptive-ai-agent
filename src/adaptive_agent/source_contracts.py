from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

TOOL_PARAMETER_KEYS = {
    "code",
    "column",
    "content",
    "dst",
    "dst_path",
    "field",
    "filter",
    "group",
    "input",
    "limit",
    "measure",
    "name",
    "output",
    "output_path",
    "path",
    "query",
    "source",
    "source_path",
    "src",
    "src_path",
    "target",
    "target_path",
    "text",
    "threshold",
    "type",
    "value",
}


def mentioned_workspace_paths(text: str) -> list[str]:
    """Extract likely workspace-relative data paths from free-form text."""
    candidates = re.findall(r"(?:[\w.-]+/)*[\w.-]+\.(?:json|csv|md|txt)", text, flags=re.I)
    paths: list[str] = []
    for candidate in candidates:
        path = candidate.strip("`'\".,:;()[]{}")
        if path.startswith("workspace/"):
            path = path.removeprefix("workspace/")
        if path and ".." not in Path(path).parts and path not in paths:
            paths.append(path)
    return paths


def payload_mentions_any_path(payload: Any, paths: list[str]) -> bool:
    """Return whether a structured tool payload contains any requested path."""
    if isinstance(payload, str):
        return any(path == payload or path in payload for path in paths)
    if isinstance(payload, dict):
        return any(payload_mentions_any_path(value, paths) for value in payload.values())
    if isinstance(payload, list):
        return any(payload_mentions_any_path(value, paths) for value in payload)
    return False


def paths_from_payload(payload: Any) -> list[str]:
    """Collect workspace-like paths mentioned anywhere in a tool payload."""
    paths: list[str] = []
    if isinstance(payload, str):
        for path in mentioned_workspace_paths(payload):
            if path not in paths:
                paths.append(path)
        return paths
    if isinstance(payload, dict):
        for value in payload.values():
            for path in paths_from_payload(value):
                if path not in paths:
                    paths.append(path)
        return paths
    if isinstance(payload, list):
        for value in payload:
            for path in paths_from_payload(value):
                if path not in paths:
                    paths.append(path)
    return paths


def payload_contains_inline_dataset(payload: Any) -> bool:
    """Detect structured records passed inline instead of by workspace path."""
    if isinstance(payload, list):
        return True
    if isinstance(payload, dict):
        if any(payload_contains_inline_dataset(value) for value in payload.values()):
            return True
        return _looks_like_inline_record(payload)
    return False


def _looks_like_inline_record(payload: dict[Any, Any]) -> bool:
    lowered_keys = {_normalize_key(key) for key in payload}
    if lowered_keys and lowered_keys <= TOOL_PARAMETER_KEYS:
        return False
    scalar_values = [value for value in payload.values() if _is_scalar(value)]
    return len(lowered_keys) >= 2 and len(scalar_values) >= 2


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def request_contains_inline_structured_data(request: str) -> bool:
    """Detect JSON/CSV-like data embedded directly in the user request."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(request):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(request[index:])
        except json.JSONDecodeError:
            continue
        if payload_contains_inline_dataset(value):
            return True

    lines = [line.strip() for line in request.splitlines() if line.strip()]
    comma_lines = [line for line in lines if "," in line]
    return len(comma_lines) >= 2
