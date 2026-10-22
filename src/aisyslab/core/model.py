"""The contract every model provider implements, plus a scripted stand-in for tests."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .messages import Message, ModelTurn, ToolCall, ToolSpec, Usage
from .tokens import estimate_tokens


class ModelError(RuntimeError):
    """A provider call failed after the SDK's own retries (auth, bad request, outage)."""


class ChatModel(Protocol):
    @property
    def name(self) -> str: ...

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int = 4096,
    ) -> ModelTurn: ...


TextSink = Callable[[str], None]


@runtime_checkable
class StreamingChatModel(ChatModel, Protocol):
    """A model that also hands over its text while it is being generated.

    ``on_text`` receives the text in pieces as they arrive; the returned turn is the same as
    ``complete`` would return. Callers that do not need streaming keep using ``complete``.
    """

    def complete_stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        on_text: TextSink,
        max_output_tokens: int = 4096,
    ) -> ModelTurn: ...


Policy = Callable[[str, Sequence[Message], Sequence[ToolSpec]], ModelTurn]


class ScriptedModel:
    """Deterministic stand-in for an LLM, used to test agent wiring offline.

    It is not an AI: it replays a fixed list of turns or calls a policy function that
    inspects the conversation. Usage is estimated from the prompt size so token budgets
    can be exercised in tests.
    """

    def __init__(self, script: Policy | Sequence[ModelTurn], name: str = "scripted") -> None:
        self._name = name
        if callable(script):
            self._policy: Policy = script
        else:
            turns = iter(script)

            def replay(_s: str, _m: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
                try:
                    return next(turns)
                except StopIteration:
                    return text_turn("Script exhausted.")

            self._policy = replay
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        self.calls += 1
        turn = self._policy(system, messages, tools)
        prompt = system + "".join(m.content for m in messages)
        turn.usage = Usage(input_tokens=estimate_tokens(prompt), output_tokens=turn.usage.output_tokens)
        turn.model = self._name
        return turn

    def complete_stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        on_text: TextSink,
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        turn = self.complete(system=system, messages=messages, tools=tools, max_output_tokens=max_output_tokens)
        for start in range(0, len(turn.text), 16):  # pieces, the way a provider streams them
            on_text(turn.text[start : start + 16])
        return turn


_ids = itertools.count(1)


def tool_turn(name: str, **arguments: Any) -> ModelTurn:
    """A model turn that requests a single tool call."""
    call = ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)
    return ModelTurn(text="", tool_calls=[call], usage=Usage(output_tokens=20), stop_reason="tool_use")


def text_turn(text: str) -> ModelTurn:
    """A model turn that ends with plain text and no tool calls."""
    return ModelTurn(text=text, tool_calls=[], usage=Usage(output_tokens=estimate_tokens(text)), stop_reason="end_turn")
