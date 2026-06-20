from __future__ import annotations

import os
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    provider: str = "openai-compatible"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5-coder:7b"
    api_key: str | None = None
    max_iterations: int = Field(default=20, ge=1)
    max_fix_retries: int = Field(default=3, ge=0)
    tool_timeout_sec: float = Field(default=20, gt=0)
    llm_timeout_sec: float = Field(default=60, gt=0)
    max_output_bytes: int = Field(default=65536, ge=1)
    compaction_token_threshold: int = Field(default=12000, ge=1)
    workspace_dir: str = "./workspace"
    skills_dir: str = "./skills"
    log_dir: str = "./logs"
    monitoring: Literal["off", "langfuse"] = "off"
    network_default: Literal["deny", "allow"] = "deny"

    @classmethod
    def load(cls) -> AgentConfig:
        env = os.environ
        data: dict[str, Any] = {
            "provider": env.get("AGENT_PROVIDER", cls.model_fields["provider"].default),
            "base_url": env.get("AGENT_BASE_URL", cls.model_fields["base_url"].default),
            "model": env.get("AGENT_MODEL", cls.model_fields["model"].default),
            "api_key": env.get("AGENT_API_KEY"),
            "monitoring": env.get("AGENT_MONITORING", "off"),
            "max_iterations": env.get("AGENT_MAX_ITERATIONS"),
            "max_fix_retries": env.get("AGENT_MAX_FIX_RETRIES"),
            "tool_timeout_sec": env.get("AGENT_TOOL_TIMEOUT_SEC"),
            "llm_timeout_sec": env.get("AGENT_LLM_TIMEOUT_SEC"),
            "max_output_bytes": env.get("AGENT_MAX_OUTPUT_BYTES"),
            "compaction_token_threshold": env.get("AGENT_COMPACTION_TOKEN_THRESHOLD"),
            "workspace_dir": env.get("AGENT_WORKSPACE_DIR"),
            "skills_dir": env.get("AGENT_SKILLS_DIR"),
            "log_dir": env.get("AGENT_LOG_DIR"),
            "network_default": env.get("AGENT_NETWORK_DEFAULT"),
        }
        return cls(**{k: v for k, v in data.items() if v not in (None, "")})
