from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Wire(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ToolSpec(_Wire):
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str = Field(min_length=1)
    code: str
    entrypoint: str = "run"
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    output_schema: dict[str, Any] | None = Field(default=None, alias="outputSchema")


class Respond(BaseModel):
    action: Literal["respond"]
    text: str
    final: bool = False


class AskUser(BaseModel):
    action: Literal["ask_user"]
    question: str
    choices: list[str] | None = None
    reason: str | None = None


class CallTool(BaseModel):
    action: Literal["call_tool"]
    name: str
    input: dict[str, Any]


class CreateTool(BaseModel):
    action: Literal["create_tool"]
    spec: ToolSpec


class UpdateTool(BaseModel):
    action: Literal["update_tool"]
    name: str
    code: str
    reason: str | None = None


class Finish(BaseModel):
    action: Literal["finish"]
    summary: str | None = None


AgentAction = Annotated[
    Respond | AskUser | CallTool | CreateTool | UpdateTool | Finish,
    Field(discriminator="action"),
]
_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def parse_agent_action(data: dict[str, Any]) -> AgentAction:
    """Parse a raw dict into a typed AgentAction using discriminated union on 'action'."""
    return _ACTION_ADAPTER.validate_python(data)


class ToolManifest(_Wire):
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    output_schema: dict[str, Any] | None = Field(default=None, alias="outputSchema")
    entrypoint: str = "run"
    runtime: Literal["python"] = "python"
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
    usage_count: int = Field(default=0, alias="usageCount", ge=0)
    trusted_status: Literal["untrusted", "session", "persisted"] = Field(alias="trustedStatus")
    version: int = Field(default=1, ge=1)
    source: Literal["generated", "mcp"] = "generated"


class ToolDigest(BaseModel):
    name: str
    description: str
    origin: Literal["builtin", "generated", "mcp"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class PolicyDecision(BaseModel):
    decision: Literal["ALLOW", "DENY", "ASK_USER"]
    action: str
    reason: str
