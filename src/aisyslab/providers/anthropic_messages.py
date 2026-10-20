"""Anthropic Messages API adapter (official ``anthropic`` SDK).

Assistant turns are replayed exactly as the API returned them (``provider_state``), so
thinking blocks survive the tool loop unchanged. All tool results of one turn go back in a
single user message, failed ones flagged with ``is_error``.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from ..core.messages import Message, ModelTurn, StopReason, ToolCall, ToolSpec, Usage
from ..core.model import ModelError

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Thinking is always on for current models and counts toward max_tokens.
MIN_MAX_TOKENS = 16000


class AnthropicModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str | None = "medium",
        fallbacks: bool = True,
        max_retries: int = 3,
        timeout: float = 300.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the Anthropic SDK: pip install 'applied-ai-systems-lab[anthropic]'") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.model = model
        self.effort = effort
        # Server-side fallback re-runs a request declined by a safety classifier on another
        # model inside the same call, instead of ending the agent run with a refusal.
        self.fallbacks = fallbacks

    @property
    def name(self) -> str:
        return self.model

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_output_tokens, MIN_MAX_TOKENS),
            "system": system,
            "messages": to_anthropic_messages(messages),
            # Automatic prompt caching: the stable prefix (tools, system, earlier turns)
            # is re-read from cache on every step of the loop.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                response = self._client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                response = self._client.messages.create(**kwargs)
        except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return parse_anthropic_response(response)


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        if m.role == "tool":
            pending_results.append(
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content, "is_error": m.is_error}
            )
            continue
        flush()
        if m.role == "user":
            if m.images:
                blocks: list[dict[str, Any]] = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": base64.standard_b64encode(img.data).decode("ascii"),
                        },
                    }
                    for img in m.images
                ]
                blocks.append({"type": "text", "text": m.content})
                out.append({"role": "user", "content": blocks})
            else:
                out.append({"role": "user", "content": m.content})
        elif m.provider_state is not None:
            out.append({"role": "assistant", "content": m.provider_state})
        else:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            content.extend({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls)
            out.append({"role": "assistant", "content": content})
    flush()
    return out


_STOP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def parse_anthropic_response(response: Any) -> ModelTurn:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(arguments)))
    u = response.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    usage = Usage(
        input_tokens=(u.input_tokens or 0) + cache_read + cache_write,
        output_tokens=u.output_tokens or 0,
        cached_input_tokens=cache_read,
    )
    stop = _STOP.get(response.stop_reason or "", "other")
    if calls and stop != "refusal":
        stop = "tool_use"
    return ModelTurn(
        text="".join(texts),
        tool_calls=calls,
        usage=usage,
        stop_reason=stop,
        model=getattr(response, "model", "") or "",
        provider_state=list(response.content),
    )
