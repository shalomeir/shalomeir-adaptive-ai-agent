import pytest
from pydantic import ValidationError

from adaptive_agent.config import AgentConfig


def test_defaults_and_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("AGENT_BASE_URL", "http://custom-host:9999/v1")
    cfg = AgentConfig.load()
    assert cfg.model == "test-model"
    assert cfg.base_url == "http://custom-host:9999/v1"
    assert cfg.max_iterations == 20
    assert cfg.tool_timeout_sec == 20
    assert cfg.network_default == "deny"
    assert cfg.monitoring == "off"


def test_runtime_limits_and_paths_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "7")
    monkeypatch.setenv("AGENT_MAX_FIX_RETRIES", "2")
    monkeypatch.setenv("AGENT_TOOL_TIMEOUT_SEC", "3.5")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SEC", "12")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_BYTES", "1024")
    monkeypatch.setenv("AGENT_COMPACTION_TOKEN_THRESHOLD", "3000")
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", "/tmp/agent-ws")
    monkeypatch.setenv("AGENT_SKILLS_DIR", "/tmp/agent-skills")
    monkeypatch.setenv("AGENT_LOG_DIR", "/tmp/agent-logs")
    monkeypatch.setenv("AGENT_NETWORK_DEFAULT", "allow")

    cfg = AgentConfig.load()

    assert cfg.max_iterations == 7
    assert cfg.max_fix_retries == 2
    assert cfg.tool_timeout_sec == 3.5
    assert cfg.llm_timeout_sec == 12
    assert cfg.max_output_bytes == 1024
    assert cfg.compaction_token_threshold == 3000
    assert cfg.workspace_dir == "/tmp/agent-ws"
    assert cfg.skills_dir == "/tmp/agent-skills"
    assert cfg.log_dir == "/tmp/agent-logs"
    assert cfg.network_default == "allow"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret-key")
    cfg = AgentConfig.load()
    assert cfg.api_key == "secret-key"


def test_invalid_monitoring_raises(monkeypatch):
    monkeypatch.setenv("AGENT_MONITORING", "bad")
    with pytest.raises(ValidationError):
        AgentConfig.load()
