from typer.testing import CliRunner

from adaptive_agent import cli
from adaptive_agent.cli import app
from adaptive_agent.llm import FakeLLMClient


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def _patch_llm(monkeypatch, replies):
    fake = FakeLLMClient(replies=replies)
    monkeypatch.setattr(cli, "HttpLLMClient", lambda *a, **k: fake)
    return fake


def test_run_command_reports_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch, ['{"action":"finish","summary":"completed-xyz"}'])
    result = CliRunner().invoke(app, ["run", "do something"])
    assert result.exit_code == 0
    assert "completed-xyz" in result.stdout


def test_chat_exits_on_empty_piped_stdin(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fake = _patch_llm(monkeypatch, ['{"action":"finish","summary":"should-not-run"}'])

    result = CliRunner().invoke(app, ["chat"], input="")

    assert result.exit_code == 0
    assert "세션을 시작합니다" in result.stdout
    assert fake.calls == 0


def test_chat_renders_clarification_as_dialogue(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch_llm(
        monkeypatch,
        [
            '{"action":"ask_user","question":"어떤 데이터를 어떻게 정리할까요?"}',
            '{"action":"finish","summary":"알겠습니다."}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="데이터 정리해줘\nevents.csv\nexit\n")

    assert result.exit_code == 0
    assert "agent: 어떤 데이터를 어떻게 정리할까요?" in result.stdout
    assert "agent: 알겠습니다." in result.stdout


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
            '{"action":"finish","summary":"완료했습니다."}',
        ],
    )

    result = CliRunner().invoke(app, ["chat"], input="도구 만들어줘\nexit\n")

    assert result.exit_code == 0
    assert "agent: 생성한 도구 'noop'을(를) 다음 세션에서도 재사용하도록 저장할까요? (y/n)" in result.stdout
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
