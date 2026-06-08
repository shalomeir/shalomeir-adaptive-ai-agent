from adaptive_agent.action_validation import TurnExecutionState


def test_terminal_is_blocked_after_tool_change_without_work_call():
    state = TurnExecutionState(requires_execution=True)

    state.advance()
    state.record_tool_change("csv-dedupe-sort")

    observation = state.terminal_block_observation()

    assert observation is not None
    assert "실제 실행 결과가 아직 없습니다" in observation
    assert "csv-dedupe-sort" in observation


def test_terminal_is_allowed_after_work_call_following_tool_change():
    state = TurnExecutionState(requires_execution=True)

    state.advance()
    state.record_tool_change("csv-dedupe-sort")
    state.advance()
    state.record_tool_call("csv-dedupe-sort", ok=True)

    assert state.terminal_block_observation() is None


def test_read_only_context_call_does_not_satisfy_tool_change_execution():
    state = TurnExecutionState(requires_execution=True)

    state.advance()
    state.record_tool_change("csv-dedupe-sort")
    state.advance()
    state.record_tool_call("readFile", ok=True)

    assert state.terminal_block_observation() is not None


def test_tool_creation_only_task_can_finish_without_execution():
    state = TurnExecutionState(requires_execution=False)

    state.advance()
    state.record_tool_change("adder")

    assert state.terminal_block_observation() is None
