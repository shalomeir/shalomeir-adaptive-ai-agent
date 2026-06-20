from __future__ import annotations

from typing import Callable

from .schemas import PolicyDecision

# Actions that require user confirmation before proceeding.
_ASK: set[str] = {
    "write_file",
    "persist_tool",
    "update_persisted_tool",
    "network_access",
    "long_run",
    "destructive",
}

# Actions that are unconditionally blocked and cannot be bypassed by user approval.
_DENY: set[str] = {
    "out_of_workspace",
}


class PolicyManager:
    """Evaluates whether an agent action is allowed, denied, or needs user confirmation."""

    def __init__(
        self,
        ask: Callable[[str], str] | None = None,
        deny: set[str] | None = None,
        ask_set: set[str] | None = None,
    ) -> None:
        # Default ask callback returns "n" (safe default — deny when unattended).
        self._ask = ask or (lambda q: "n")
        self._deny = deny if deny is not None else set(_DENY)
        self._ask_set = ask_set if ask_set is not None else set(_ASK)

    def evaluate(self, action_id: str) -> PolicyDecision:
        """Return a PolicyDecision for the given action identifier."""
        if action_id in self._deny:
            return PolicyDecision(
                decision="DENY",
                action=action_id,
                reason="정책상 금지된 행동",
            )
        if action_id in self._ask_set:
            return PolicyDecision(
                decision="ASK_USER",
                action=action_id,
                reason="부수효과가 있어 사용자 확인이 필요",
            )
        return PolicyDecision(
            decision="ALLOW",
            action=action_id,
            reason="안전한 행동",
        )

    def confirm(self, action_id: str) -> bool:
        """Prompt the user and return True only when they explicitly agree."""
        answer = self._ask(_confirmation_text(action_id))
        return answer.strip().lower() in {"y", "yes"}


def _confirmation_text(action_id: str) -> str:
    """Return a user-facing confirmation prompt for a policy action."""
    if action_id == "write_file":
        return "파일 쓰기가 필요합니다. 진행할까요? (y/n)"
    if action_id == "persist_tool":
        return (
            "방금 만든 도구는 현재 세션에서만 쓸 수 있습니다. 다음 세션에서도 재사용하도록 "
            "영구 저장할까요? (y/n)"
        )
    if action_id.startswith("persist:"):
        name = action_id.split(":", 1)[1]
        return (
            f"생성한 도구 '{name}'은(는) 현재 세션에서만 쓸 수 있습니다. 다음 세션에서도 "
            "재사용하도록 영구 저장할까요? (y/n)"
        )
    if action_id == "update_persisted_tool":
        return "저장된 도구를 수정해야 합니다. 진행할까요? (y/n)"
    if action_id == "network_access":
        return "네트워크 접근이 필요합니다. 진행할까요? (y/n)"
    if action_id == "long_run":
        return "오래 걸릴 수 있는 작업입니다. 진행할까요? (y/n)"
    if action_id == "destructive":
        return "되돌리기 어려운 작업입니다. 진행할까요? (y/n)"
    return f"{action_id} 작업을 진행할까요? (y/n)"
