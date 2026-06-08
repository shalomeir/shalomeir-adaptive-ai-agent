from __future__ import annotations

from typing import Callable

from .conversation import ConversationStore
from .schemas import Message


def estimate_tokens(messages: list[Message]) -> int:
    # Provider-agnostic rough estimate: ~4 chars per token.
    return sum(len(m.content) for m in messages) // 4


class ContextManager:
    """Compacts a ConversationStore when estimated token usage exceeds the threshold.

    After compaction the body becomes:
        [carry_message, *recent_messages]
    where ``carry_message`` is a synthetic user message containing the LLM-produced
    summary and any facts registered via ``carry_over_fact``.
    """

    def __init__(
        self,
        token_threshold: int,
        summarize: Callable[[list[Message]], str],
        keep_recent: int = 4,
    ) -> None:
        self.token_threshold = token_threshold
        self.summarize = summarize
        self.keep_recent = keep_recent
        self._facts: list[str] = []

    def carry_over_fact(self, fact: str) -> None:
        """Register a fact that must survive every compaction."""
        if fact not in self._facts:
            self._facts.append(fact)

    def estimated_tokens(self, conv: ConversationStore) -> int:
        return estimate_tokens(conv.messages())

    def maybe_compact(self, conv: ConversationStore) -> bool:
        """Compact *conv* if token estimate exceeds the threshold.

        Returns ``True`` when compaction was performed, ``False`` otherwise.
        """
        if self.estimated_tokens(conv) <= self.token_threshold:
            return False
        body = conv.body()
        if len(body) <= self.keep_recent:
            return False
        old, recent = body[: -self.keep_recent], body[-self.keep_recent :]
        summary = self.summarize(old)
        facts = "\n".join(f"- {f}" for f in self._facts)
        carry = Message(
            role="user",
            content=f"[요약] {summary}\n[보존된 핵심 사실]\n{facts}",
        )
        conv.replace_body([carry, *recent])
        return True
