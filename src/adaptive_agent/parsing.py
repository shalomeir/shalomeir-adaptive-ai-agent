from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .schemas import AgentAction, parse_agent_action

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_CANONICAL_ACTIONS = {"respond", "ask_user", "call_tool", "create_tool", "update_tool", "finish"}
_DIRECT_TOOL_ACTIONS = {"readFile", "writeFile", "listFiles", "runPython", "searchDocs", "askUser"}
_ACTION_CONTRACT = (
    '반드시 action 필드를 포함하세요. 최종 답변은 '
    '{"action":"respond","text":"...","final":true} 또는 '
    '{"action":"finish","summary":"..."} 형식이어야 합니다. '
    '도구 결과 JSON을 {"path":...}처럼 그대로 최상위 응답으로 반환하지 마세요.'
)


def _strip_fence(text: str) -> str:
    """Remove markdown code-fence wrapper if present, returning the inner text."""
    m = _FENCE.search(text)
    return m.group(1) if m else text


def _extract_object(text: str) -> str:
    """Slice out the first balanced JSON object from arbitrary surrounding text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSON object was not closed")


def _remove_trailing_commas(text: str) -> str:
    """Strip trailing commas before } or ] to tolerate common LLM formatting errors."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _quote_triple_quoted_fields(text: str) -> str:
    """Convert Python-style triple-quoted code/content fields into JSON strings."""

    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        body = match.group(3)
        return f"{prefix}{json.dumps(body, ensure_ascii=False)}"

    pattern = r'("(?:code|content)"\s*:\s*)("""|\'\'\')(.*?)(\2)'
    return re.sub(pattern, replace, text, flags=re.DOTALL)


def json_repair(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction: strip fences, extract object, fix trailing commas."""
    candidate = _remove_trailing_commas(_quote_triple_quoted_fields(_extract_object(_strip_fence(raw))))
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return _normalize_action_payload(value)


def _normalize_action_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize common model action aliases before strict schema validation."""
    action = value.get("action")
    if not isinstance(action, str):
        return value

    if action == "respond" and "text" not in value:
        text = value.get("output", value.get("message", value.get("summary")))
        if text is not None:
            return {
                **value,
                "text": text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
            }

    if action == "ask_user" and "question" not in value:
        input_data = value.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}
        question = value.get("message", input_data.get("message", input_data.get("question")))
        if question is not None:
            return {
                **value,
                "question": str(question),
                "choices": value.get("choices", input_data.get("choices")),
            }

    if action in _CANONICAL_ACTIONS:
        return value

    if action in _DIRECT_TOOL_ACTIONS:
        return {
            "action": "call_tool",
            "name": action,
            "input": value.get("input", {}),
        }

    if action == "callTool":
        return {
            "action": "call_tool",
            "name": value.get("name"),
            "input": value.get("input", {}),
        }

    if action == "askUser":
        input_data = value.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}
        return {
            "action": "ask_user",
            "question": value.get("question", input_data.get("question")),
            "choices": value.get("choices", input_data.get("choices")),
            "reason": value.get("reason", input_data.get("reason")),
        }

    if action == "createTool":
        return {"action": "create_tool", "spec": value.get("spec", value.get("input"))}

    if action == "updateTool":
        input_data = value.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}
        return {
            "action": "update_tool",
            "name": value.get("name", input_data.get("name")),
            "code": value.get("code", input_data.get("code")),
            "reason": value.get("reason", input_data.get("reason")),
        }

    return value


@dataclass
class ParseResult:
    ok: bool
    action: AgentAction | None = None
    error: str | None = None


def parse_action_text(raw: str) -> ParseResult:
    """Parse raw LLM output into a typed AgentAction, returning human-readable errors on failure."""
    try:
        data = json_repair(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return ParseResult(
            ok=False, error=f"출력이 유효한 JSON이 아닙니다: {e}. 하나의 JSON 객체만 반환하세요."
        )
    if not isinstance(data.get("action"), str):
        preview = json.dumps(data, ensure_ascii=False)
        return ParseResult(
            ok=False,
            error=f"이전 응답은 JSON이지만 action 필드가 없습니다: {preview}. {_ACTION_CONTRACT}",
        )
    try:
        return ParseResult(ok=True, action=parse_agent_action(data))
    except Exception as e:
        return ParseResult(
            ok=False,
            error=f"action 형식을 어겼습니다: {e}. {_ACTION_CONTRACT}",
        )
