"""Provider-neutral conversation types.

Providers translate these to and from their own wire formats. Anything a provider must
replay verbatim (for example Anthropic thinking blocks) travels in ``provider_state`` and
is never edited by the harness: the conversation is append-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass(frozen=True)
class ImagePart:
    """Binary image attached to a user message."""

    media_type: str
    data: bytes


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model produced arguments that are not valid JSON; the toolbox turns
    # this into an error result the model can correct, instead of crashing the loop.
    parse_error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass
class Message:
    role: Role
    content: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    provider_state: Any = None


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: StopReason
    model: str = ""
    provider_state: Any = None
