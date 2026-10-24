"""OpenAI Chat Completions adapter.

Also serves any OpenAI-compatible server (LM Studio, Ollama, vLLM) through ``base_url``,
which is how the labs run against a local model without an API key.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from typing import Any

from ..core.messages import Message, ModelTurn, StopReason, ToolCall, ToolSpec, Usage
from ..core.model import ModelError, TextSink
from ..core.tokens import estimate_tokens

# Reasoning models spend completion tokens on reasoning before the answer; a small limit can
# leave nothing for the answer itself. For them a caller's smaller limit is raised to this floor
# (only generated tokens are billed); other models keep the caller's limit.
MIN_COMPLETION_TOKENS = 16000
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        token_param: str = "max_completion_tokens",
        max_retries: int = 3,
        timeout: float = 180.0,
        label: str | None = None,
        stream_usage: bool = True,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        self._openai = openai
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key is None and base_url is None:
            raise ModelError("OPENAI_API_KEY is not set (or pass --base-url for a local server).")
        self._client = openai.OpenAI(
            api_key=key or "local", base_url=base_url, max_retries=max_retries, timeout=timeout
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature  # left unset by default: reasoning models reject it
        self.token_param = token_param
        self._label = label or model
        # Token usage of a streamed response is only sent when asked for; some local servers
        # do not support the option, and then usage is estimated from the text instead.
        self.stream_usage = stream_usage

    @property
    def name(self) -> str:
        return self._label

    @property
    def provider(self) -> str:
        # A local server speaks the same API, but reasoning state from one never goes to the other.
        return "local" if self._label.startswith("local:") else "openai"

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        payload = self._payload(system, messages, tools, max_output_tokens)
        try:
            response = self._client.chat.completions.create(**payload)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return parse_openai_response(response)

    def complete_stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        on_text: TextSink,
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        """The same request, streamed: content deltas go to ``on_text``; the final completion is
        parsed exactly like a non-streamed one."""
        payload = self._payload(system, messages, tools, max_output_tokens)
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}
        # The stream helper raises when the length limit or the content filter stops the answer.
        # The completion it carries is parsed like any other: a truncated turn, or a refusal.
        stopped = tuple(
            getattr(self._openai, name)
            for name in ("LengthFinishReasonError", "ContentFilterFinishReasonError")
            if hasattr(self._openai, name)
        )
        try:
            with self._client.chat.completions.stream(**payload) as stream:
                for event in stream:
                    if event.type == "content.delta":
                        on_text(event.delta)
                final = stream.get_final_completion()
        except stopped as exc:
            final = getattr(exc, "completion", None)  # type: ignore[assignment]
            if final is None:  # filtered before anything came back
                usage = Usage(input_tokens=estimate_tokens(system + "".join(m.content for m in messages)))
                return ModelTurn(text="", tool_calls=[], usage=usage, stop_reason="refusal", model=self.model)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        turn = parse_openai_response(final)
        if final.usage is None:
            prompt = system + "".join(m.content for m in messages)
            turn.usage = Usage(input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(turn.text))
        return turn

    def _payload(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int
    ) -> dict[str, Any]:
        reasons = self.reasoning_effort is not None or self.model.startswith(REASONING_PREFIXES)
        floor = reasons and self.token_param != "max_tokens"
        limit = max(max_output_tokens, MIN_COMPLETION_TOKENS) if floor else max_output_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *to_openai_messages(messages)],
            self.token_param: limit,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            if m.images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img in m.images:
                    data = base64.b64encode(img.data).decode("ascii")
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{data}"}})
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if not m.content and not m.tool_calls:
                continue  # an empty turn (output used up by reasoning) is not valid to send back
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
                    }
                    for c in m.tool_calls
                ]
            out.append(entry)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def parse_openai_response(response: Any) -> ModelTurn:
    choice = response.choices[0]
    message = choice.message
    calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        function = getattr(tc, "function", None)
        if function is None:  # non-function tool types are not used here
            continue
        raw = function.arguments or "{}"
        try:
            parsed = json.loads(raw)
            error = None if isinstance(parsed, dict) else "arguments must be a JSON object"
        except json.JSONDecodeError as exc:
            parsed, error = {}, str(exc)
        calls.append(
            ToolCall(
                id=tc.id, name=function.name, arguments=parsed if isinstance(parsed, dict) else {}, parse_error=error
            )
        )

    usage = Usage()
    if response.usage is not None:
        details = getattr(response.usage, "prompt_tokens_details", None)
        usage = Usage(
            input_tokens=response.usage.prompt_tokens or 0,
            output_tokens=response.usage.completion_tokens or 0,
            cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        )

    finish = choice.finish_reason
    stop: StopReason
    if getattr(message, "refusal", None) or finish == "content_filter":
        stop = "refusal"
    elif calls:
        stop = "tool_use"
    elif finish == "length":
        stop = "max_tokens"
    elif finish == "stop":
        stop = "end_turn"
    else:
        stop = "other"
    text = message.content or getattr(message, "refusal", None) or ""
    return ModelTurn(
        text=text, tool_calls=calls, usage=usage, stop_reason=stop, model=getattr(response, "model", "") or ""
    )
