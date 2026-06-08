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


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret-key")
    cfg = AgentConfig.load()
    assert cfg.api_key == "secret-key"


def test_invalid_monitoring_raises(monkeypatch):
    monkeypatch.setenv("AGENT_MONITORING", "bad")
    with pytest.raises(ValidationError):
        AgentConfig.load()
