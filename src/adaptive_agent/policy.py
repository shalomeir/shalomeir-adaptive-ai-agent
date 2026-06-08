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
        answer = self._ask(f"'{action_id}' 작업을 진행할까요? (y/n)")
        return answer.strip().lower() in {"y", "yes"}
