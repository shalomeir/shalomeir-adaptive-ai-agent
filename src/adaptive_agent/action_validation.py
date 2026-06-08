from __future__ import annotations

from dataclasses import dataclass


READ_ONLY_CONTEXT_TOOLS = {"readFile", "listFiles", "searchDocs"}


@dataclass
class TurnExecutionState:
    """Track whether a turn has executed work after changing tool capability."""

    requires_execution: bool
    action_index: int = 0
    last_generated_tool_change: int | None = None
    last_successful_work_call: int | None = None
    changed_tool_names: list[str] | None = None

    def advance(self) -> None:
        """Move the state clock forward for one parsed action."""
        self.action_index += 1

    def record_tool_change(self, name: str) -> None:
        """Remember that a generated tool was created or updated in this turn."""
        self.last_generated_tool_change = self.action_index
        if self.changed_tool_names is None:
            self.changed_tool_names = []
        if name not in self.changed_tool_names:
            self.changed_tool_names.append(name)

    def record_tool_call(self, name: str, ok: bool) -> None:
        """Record successful work-producing calls after planning actions."""
        if ok and name not in READ_ONLY_CONTEXT_TOOLS:
            self.last_successful_work_call = self.action_index

    def terminal_block_observation(self) -> str | None:
        """Return an observation if terminal output would skip required execution."""
        if not self.requires_execution or self.last_generated_tool_change is None:
            return None
        if (
            self.last_successful_work_call is not None
            and self.last_successful_work_call > self.last_generated_tool_change
        ):
            return None

        tool_hint = ""
        if self.changed_tool_names:
            tool_hint = f" 생성/수정한 도구: {', '.join(self.changed_tool_names)}."
        return (
            "현재 요청은 파일/작업 결과가 필요한데, 생성 또는 수정한 도구 이후 실제 실행 결과가 "
            "아직 없습니다. 최종 응답으로 종료하지 말고 필요한 도구를 call_tool로 실행한 뒤 "
            "실제 결과를 확인하세요." + tool_hint
        )
