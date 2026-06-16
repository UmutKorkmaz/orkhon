"""OpenAI-compatible request/response schemas for the chat completions API.

A minimal but faithful subset of the OpenAI Chat Completions shapes so existing
OpenAI clients can talk to Orkhon unchanged. Only the fields Orkhon actually honors
are modeled; unknown extra fields are ignored (``extra="ignore"``) rather than
rejected, matching the tolerant behavior real clients expect.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]
InboundRole = Literal["system", "user", "assistant"]  # clients may NOT send "tool"


class ChatMessage(BaseModel):
    """A message. Inbound requests use InboundRole; "tool" is server-generated only."""

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str


def _reject_inbound_tool(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Clients cannot supply role='tool' (only the server generates tool turns)."""
    for m in messages:
        if m.role == "tool":
            from pydantic import ValidationError
            raise ValueError("client may not send role='tool' messages (server-only)")
    return messages


class AgentRunStep(BaseModel):
    index: int
    plan: str = ""
    tool_call: dict | None = None
    observation: str = ""
    error: bool = False


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "orkhon"
    messages: list[ChatMessage]
    max_steps: int = Field(default=5, ge=1)
    max_tokens: int = Field(default=160, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_k: int | None = 40
    include_messages: bool = False

    @field_validator("messages")
    @classmethod
    def _no_inbound_tool(cls, v):
        return _reject_inbound_tool(v)


AgentStatus = Literal["completed", "max_steps", "tool_errors", "no_answer"]


class AgentRunResponse(BaseModel):
    object: str = "agent.run"
    model: str = "orkhon"
    status: AgentStatus
    final_answer: str
    steps: list[AgentRunStep] = []
    messages: list[ChatMessage] = []


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "orkhon"
    messages: list[ChatMessage]
    # Generation controls (OpenAI names; mapped to orkhon.model.generate).
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    stream: bool = False


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orkhon"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# --- Streaming (SSE) chunk shapes ---------------------------------------- #


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orkhon"
    choices: list[StreamChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "orkhon"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
