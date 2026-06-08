from __future__ import annotations

from .schemas import Message


class ConversationStore:
    """Holds the full message history for one agent session.

    The system prompt is kept separately so it always appears first in
    ``messages()`` and is never removed during compaction.
    """

    def __init__(self, system: str) -> None:
        self._system = Message(role="system", content=system)
        self._messages: list[Message] = []

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))

    def add_observation(self, content: str) -> None:
        # Tool observations are surfaced as "tool" role messages.
        self._messages.append(Message(role="tool", content=content))

    def messages(self) -> list[Message]:
        """Return system + all body messages in order."""
        return [self._system, *self._messages]

    def body(self) -> list[Message]:
        """Return only the non-system messages (mutable copy)."""
        return list(self._messages)

    def replace_body(self, messages: list[Message]) -> None:
        """Replace the entire body; used by ContextManager after compaction."""
        self._messages = messages
