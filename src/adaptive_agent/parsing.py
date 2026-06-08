from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .schemas import AgentAction, parse_agent_action

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Remove markdown code-fence wrapper if present, returning the inner text."""
    m = _FENCE.search(text)
    return m.group(1) if m else text


# 한계: rfind로 마지막 '}'를 쓰므로 trailing 텍스트에 '}'가 있으면 outermost 보장이 깨질 수
# 있고, 그 경우 json.loads가 실패해 상위에서 ok=False로 처리된다.
def _extract_object(text: str) -> str:
    """Slice out the outermost JSON object from arbitrary surrounding text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    """Strip trailing commas before } or ] to tolerate common LLM formatting errors."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def json_repair(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction: strip fences, extract object, fix trailing commas."""
    candidate = _remove_trailing_commas(_extract_object(_strip_fence(raw)))
    return json.loads(candidate)


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
        return ParseResult(ok=False, error=f"출력이 유효한 JSON이 아닙니다: {e}. "
                                            "하나의 JSON 객체만 반환하세요.")
    try:
        return ParseResult(ok=True, action=parse_agent_action(data))
    except Exception as e:
        return ParseResult(ok=False, error=f"action 형식을 어겼습니다: {e}. "
                                            "schemas의 action 형식으로 다시 답하세요.")
