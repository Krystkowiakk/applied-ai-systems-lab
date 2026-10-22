"""An offline stand-in model, so the service and its UI can be tried without a model provider.

It is not a model and solves nothing: it asks one question when it may, then says that no model
is connected. It streams its text slowly, so streaming, questions, the queue and branches can be
seen working.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..core.messages import Message, ModelTurn, ToolCall, ToolSpec, Usage
from ..core.model import TextSink
from ..core.tokens import estimate_tokens

NOTICE = (
    "This is the offline demo model, so no agent is working on this request. Start the service with "
    "--provider to connect a model. Questions, the queue, branches and the event stream work as usual."
)


class DemoModel:
    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay

    @property
    def name(self) -> str:
        return "offline-demo"

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        usage = Usage(input_tokens=estimate_tokens(system + "".join(m.content for m in messages)), output_tokens=40)
        asked = any(m.role == "tool" and m.tool_name == "ask_human" for m in messages)
        if any(t.name == "ask_human" for t in tools) and not asked and messages and messages[-1].role == "user":
            call = ToolCall(
                id=f"demo_{len(messages)}", name="ask_human", arguments={"question": "Which line do you mean?"}
            )
            return ModelTurn(text="", tool_calls=[call], usage=usage, stop_reason="tool_use", model=self.name)
        return ModelTurn(text=NOTICE, tool_calls=[], usage=usage, stop_reason="end_turn", model=self.name)

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
        words = turn.text.split(" ")
        for index, word in enumerate(words):
            on_text(word if index == len(words) - 1 else word + " ")
            time.sleep(self.delay)
        return turn
