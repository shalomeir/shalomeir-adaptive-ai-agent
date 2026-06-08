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
