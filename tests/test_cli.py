from io import StringIO

from rich.console import Console
from typer.testing import CliRunner

from adaptive_agent import cli
from adaptive_agent.cli import app
from adaptive_agent.config import AgentConfig
from adaptive_agent.llm import AnthropicMessagesClient, FakeLLMClient, HttpLLMClient


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.1" in result.stdout


def _patch_llm(monkeypatch, replies):
    fake = FakeLLMClient(replies=replies)
    monkeypatch.setattr(cli, "HttpLLMClient", lambda *a, **k: fake)
    monkeypatch.setattr(cli, "AnthropicMessagesClient", lambda *a, **k: fake)
    return fake


def test_run_command_reports_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch, ['{"action":"finish","summary":"completed-xyz"}'])
    result = CliRunner().invoke(app, ["run", "do something"])
    assert result.exit_code == 0
    assert "completed-xyz" in result.stdout


def test_chat_exits_on_empty_piped_stdin(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_MODEL", "qwen2.5-coder:7b")
    monkeypatch.delenv("AGENT_WORKSPACE_DIR", raising=False)
    fake = _patch_llm(monkeypatch, ['{"action":"finish","summary":"should-not-run"}'])

    result = CliRunner().invoke(app, ["chat"], input="")

    assert result.exit_code == 0
    assert "Adaptive AI Agent CLI" in result.stdout
    assert "0.1.1" in result.stdout
    assert "model:" in result.stdout
    assert "qwen2.5-coder:7b" in result.stdout
    assert "directory:" in result.stdout
    assert "workspace:" in result.stdout
    assert "세션을 시작합니다. 'exit' 또는 '/exit'로 종료." in result.stdout
    assert fake.calls == 0


def test_startup_banner_shows_runtime_context(monkeypatch, tmp_path):
    stream = StringIO()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "console", Console(file=stream, width=200))

    cli._render_startup_banner(AgentConfig(model="custom-model", workspace_dir="custom-workspace"))

    output = stream.getvalue()
    assert "Adaptive AI Agent CLI" in output
    assert "v0.1.1" in output
    assert "model:" in output
    assert "custom-model" in output
    assert "directory:" in output
    assert str(tmp_path) in output
    assert "workspace:" in output
    assert str(tmp_path / "custom-workspace") in output


def test_build_llm_client_uses_anthropic_only_for_anthropic_provider():
    anthropic = cli._build_llm_client(
        AgentConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet",
            api_key="key",
        )
    )
    openai_compatible = cli._build_llm_client(
        AgentConfig(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    )

    assert isinstance(anthropic, AnthropicMessagesClient)
    assert isinstance(openai_compatible, HttpLLMClient)


def test_chat_renders_clarification_as_dialogue(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            '{"action":"ask_user","question":"어떤 데이터를 어떻게 정리할까요?"}',
            (
                '{"action":"create_tool","spec":{"name":"clarified-cleanup","description":"cleanup",'
                '"code":"def run(input):\\n    if False:\\n        open(input.get(\\"path\\", \\"events.csv\\")).read()\\n    return {\\"ok\\": True}",'
                '"inputSchema":{"type":"object"}}}'
            ),
            '{"action":"call_tool","name":"clarified-cleanup","input":{"path":"events.csv"}}',
            '{"action":"finish","summary":"알겠습니다."}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="데이터 정리해줘\nevents.csv\nexit\n")

    assert result.exit_code == 0
    assert "agent: 어떤 데이터를 어떻게 정리할까요?" in result.stdout
    assert "agent: 알겠습니다." in result.stdout


def test_chat_skips_loading_in_non_terminal_capture(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch, ['{"action":"finish","summary":"completed"}'])

    result = CliRunner().invoke(app, ["chat"], input="do something\nexit\n")

    assert result.exit_code == 0
    assert "agent: completed" in result.stdout
    assert "you: agent:" not in result.stdout
    assert "agent: loading" not in result.stdout


def test_chat_slash_exit_ends_session_before_llm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = _patch_llm(monkeypatch, ['{"action":"finish","summary":"should-not-run"}'])

    result = CliRunner().invoke(app, ["chat"], input="/exit\n")

    assert result.exit_code == 0
    assert fake.calls == 0
    assert "should-not-run" not in result.stdout


def test_chat_ignores_empty_interactive_input(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_is_interactive_stdin", lambda: True)
    fake = _patch_llm(monkeypatch, ['{"action":"finish","summary":"completed"}'])

    result = CliRunner().invoke(app, ["chat"], input="\ndo something\nexit\n")

    assert result.exit_code == 0
    assert fake.calls == 1
    assert "agent: completed" in result.stdout


def test_loading_indicator_clears_before_result_for_terminal(monkeypatch):
    class FakeRunner:
        def run_turn(self, request):
            return "done"

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=True))
    monkeypatch.setattr(cli, "LOADING_INTERVAL_SEC", 10)

    result = cli._run_turn_with_loading(FakeRunner(), "go")

    assert result == "done"
    output = stream.getvalue()
    assert f"agent: {cli.LOADING_STYLE_START}loading.{cli.LOADING_STYLE_END}" in output
    assert output.endswith("\r\x1b[K")


def test_loading_indicator_stops_before_interactive_prompt(monkeypatch):
    class PromptingRunner:
        def run_turn(self, request):
            cli._stop_active_loading()
            return "done"

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=True))
    monkeypatch.setattr(cli, "LOADING_INTERVAL_SEC", 10)

    result = cli._run_turn_with_loading(PromptingRunner(), "go")

    assert result == "done"
    output = stream.getvalue()
    assert output.count(f"agent: {cli.LOADING_STYLE_START}loading.{cli.LOADING_STYLE_END}") == 1
    assert output.endswith("\r\x1b[K")


def test_loading_indicator_resumes_after_interactive_prompt(monkeypatch):
    class PromptingRunner:
        def run_turn(self, request):
            answer = cli._ask("어떤 데이터를 정리할까요?")
            assert answer == "events.csv"
            return "done"

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=True))
    monkeypatch.setattr(cli, "LOADING_INTERVAL_SEC", 10)
    monkeypatch.setattr(cli, "_prompt_user", lambda **_kwargs: "events.csv")

    result = cli._run_turn_with_loading(PromptingRunner(), "go")

    assert result == "done"
    output = stream.getvalue()
    loading_frame = f"agent: {cli.LOADING_STYLE_START}loading.{cli.LOADING_STYLE_END}"
    assert output.count(loading_frame) == 2
    assert "agent: 어떤 데이터를 정리할까요?" in output
    assert output.endswith("\r\x1b[K")


def test_loading_can_be_disabled_for_terminal(monkeypatch):
    class FakeRunner:
        def run_turn(self, request):
            return "done"

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, force_terminal=True))

    result = cli._run_turn_with_loading(FakeRunner(), "go", enabled=False)

    assert result == "done"
    assert stream.getvalue() == ""


def test_run_command_does_not_print_loading(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch, ['{"action":"finish","summary":"completed"}'])

    result = CliRunner().invoke(app, ["run", "do something"])

    assert result.exit_code == 0
    assert "completed" in result.stdout
    assert "loading..." not in result.stdout


def test_chat_exit_during_clarification_ends_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = _patch_llm(
        monkeypatch,
        [
            '{"action":"ask_user","question":"어떤 데이터를 어떻게 정리할까요?"}',
            '{"action":"finish","summary":"should-not-run"}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="데이터 정리해줘\nexit\n")

    assert result.exit_code == 0
    assert fake.calls == 1
    assert "should-not-run" not in result.stdout


def test_chat_slash_exit_during_clarification_ends_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = _patch_llm(
        monkeypatch,
        [
            '{"action":"ask_user","question":"어떤 데이터를 어떻게 정리할까요?"}',
            '{"action":"finish","summary":"should-not-run"}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="데이터 정리해줘\n/exit\n")

    assert result.exit_code == 0
    assert fake.calls == 1
    assert "should-not-run" not in result.stdout


def test_chat_renders_policy_confirmation_as_dialogue(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            '{"action":"call_tool","name":"writeFile","input":{"path":"out.txt","content":"hi"}}',
            '{"action":"finish","summary":"저장했습니다."}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="out.txt 저장해줘\ny\nexit\n")

    assert result.exit_code == 0
    assert "agent: 파일 쓰기가 필요합니다. 진행할까요? (y/n)" in result.stdout
    assert "you (n): agent:" not in result.stdout
    assert "agent: 저장했습니다." in result.stdout


def test_chat_exit_at_policy_confirmation_does_not_hide_completed_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            (
                '{"action":"create_tool","spec":{"name":"noop","description":"noop",'
                '"code":"def run(input):\\n    return {\\"ok\\": True}",'
                '"inputSchema":{"type":"object"}}}'
            ),
            '{"action":"call_tool","name":"noop","input":{}}',
            '{"action":"finish","summary":"완료했습니다."}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="도구 만들어줘\nexit\n")

    assert result.exit_code == 0
    assert "agent: 생성한 도구 'noop'은(는) 현재 세션에서만 쓸 수 있습니다." in result.stdout
    assert "재사용하도록 영구 저장할까요? (y/n)" in result.stdout
    assert "agent: 완료했습니다." in result.stdout


def test_run_yes_auto_approves_write(monkeypatch, tmp_path):
    # --yes는 파일 쓰기 y/n 게이트를 자동 승인해, 비대화형에서도 부수효과가 진행된다.
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            '{"action":"call_tool","name":"writeFile","input":{"path":"out.txt","content":"hi"}}',
            '{"action":"finish","summary":"saved"}',
        ],
    )
    result = CliRunner().invoke(app, ["run", "write a file", "--yes"])
    assert result.exit_code == 0
    assert (tmp_path / "workspace" / "out.txt").read_text() == "hi"


def test_run_without_yes_declines_write(monkeypatch, tmp_path):
    # --yes가 없으면 비대화형 기본은 안전하게 거절이라 파일이 만들어지지 않는다.
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            '{"action":"call_tool","name":"writeFile","input":{"path":"out.txt","content":"hi"}}',
            '{"action":"finish","summary":"skipped"}',
        ],
    )
    result = CliRunner().invoke(app, ["run", "write a file"])
    assert result.exit_code == 0
    assert not (tmp_path / "workspace" / "out.txt").exists()


def test_assemble_runner_uses_configured_compaction_threshold(monkeypatch, tmp_path):
    fake = _patch_llm(monkeypatch, ['{"action":"finish","summary":"ok"}'])
    cfg = AgentConfig(
        workspace_dir=str(tmp_path / "workspace"),
        skills_dir=str(tmp_path / "skills"),
        log_dir=str(tmp_path / "logs"),
        compaction_token_threshold=5,
    )

    runner = cli._assemble_runner(
        cfg,
        docs_dir=str(tmp_path / "docs"),
        free_ask=lambda *_a: "n",
        confirm_ask=lambda *_a: "n",
    )

    assert runner.ctx.token_threshold == 5
    assert runner.deps.llm is fake
