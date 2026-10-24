"""Failure modes found in review: a tool that returns odd values, a turn over budget, a calculator
asked for a huge number, a stream stopped by a filter, an empty turn, a server-side fallback."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import date
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel, Field

from aisyslab.core.agent import AgentConfig, run_agent
from aisyslab.core.calculator import evaluate
from aisyslab.core.messages import Message, ModelTurn, ToolCall, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn
from aisyslab.core.tools import NoArgs, Tool, Toolbox, ToolError
from aisyslab.core.trace import Tracer
from aisyslab.providers.anthropic_messages import parse_anthropic_response, to_anthropic_messages
from aisyslab.providers.openai_chat import to_openai_messages
from aisyslab.runtime.engine import jsonable


def test_the_calculator_caps_the_result_not_just_the_exponent() -> None:
    started = time.perf_counter()
    for expression in ("(((((((9)**9)**9)**9)**9)**9)**9)**9", "1e300**9", "1e308*10"):
        with pytest.raises(ToolError, match="too large"):
            evaluate(expression)
    assert time.perf_counter() - started < 0.5
    assert evaluate("(9**9)**2") == 387420489**2 and evaluate("2**-3") == 0.125


def test_a_tool_value_json_cannot_encode_is_still_an_answer() -> None:
    box = Toolbox([Tool("when", "A date and a set.", NoArgs, lambda _: {"day": date(2026, 10, 24), "ids": {1}})])
    result = box.execute(ToolCall("c1", "when", {}))
    assert not result.is_error and json.loads(result.content)["day"] == "2026-10-24"


def test_an_mcp_error_with_a_message_field_keeps_its_text() -> None:
    error = ToolError.from_payload('{"error": "denied", "message": "server says no", "hint": "ask"}')
    assert str(error) == "denied" and error.hint == "ask" and error.data == {"message": "server says no"}


def test_a_dict_field_keeps_its_value_schema() -> None:
    class Args(BaseModel):
        scores: dict[str, int] = Field(default_factory=dict)

    schema = Tool("t", "d", Args, lambda _: "").spec().parameters
    assert schema["additionalProperties"] is False
    assert schema["properties"]["scores"]["additionalProperties"] == {"type": "integer"}


def test_the_tool_budget_holds_inside_a_single_turn() -> None:
    ran: list[int] = []
    box = Toolbox([Tool("ping", "Ping.", NoArgs, lambda _: ran.append(1) or "pong")])
    calls = [ToolCall(f"c{i}", "ping", {}) for i in range(10)]

    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        return ModelTurn(text="", tool_calls=calls, usage=text_turn("").usage, stop_reason="tool_use")

    run = run_agent(
        model=ScriptedModel(policy),
        toolbox=box,
        system="s",
        task="t",
        tracer=Tracer(),
        config=AgentConfig(max_tool_calls=3),
    )
    answered = [m for m in run.messages if m.role == "tool"]
    assert len(ran) == 3 and run.status == "tool_budget" and len(answered) == 10  # every call gets a result
    assert sum(m.is_error for m in answered) == 7


def test_a_stream_stopped_by_the_content_filter_is_a_refusal() -> None:
    from aisyslab.providers.openai_chat import OpenAIChatModel

    class Filtered(Exception):
        completion = None

    class Manager:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __iter__(self) -> object:
            raise Filtered()

    model = OpenAIChatModel.__new__(OpenAIChatModel)
    model.model, model.reasoning_effort, model.temperature = "m", None, None
    model.token_param, model.stream_usage = "max_completion_tokens", True
    model._openai = NS(APIError=type("APIError", (Exception,), {}), ContentFilterFinishReasonError=Filtered)
    model._client = NS(chat=NS(completions=NS(stream=lambda **_kw: Manager())))
    turn = model.complete_stream(system="s", messages=[Message(role="user", content="x")], tools=[], on_text=print)
    assert turn.stop_reason == "refusal" and turn.text == ""


def test_an_empty_assistant_turn_is_not_sent_back() -> None:
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content=""),
        Message(role="user", content="?"),
    ]
    assert [m["role"] for m in to_openai_messages(history)] == ["user", "user"]
    assert [m["role"] for m in to_anthropic_messages(history)] == ["user", "user"]
    stated = [Message(role="user", content="hi"), Message(role="assistant", content="", provider_state=[])]
    assert [m["role"] for m in to_anthropic_messages(stated)] == ["user"]


def test_a_server_side_fallback_keeps_only_what_the_serving_model_wrote() -> None:
    declined = NS(type="tool_use", id="old", name="delete_everything", input={})
    boundary = NS(type="fallback", from_=NS(model="a"), to=NS(model="b"), trigger=NS(type="refusal"))
    served = [NS(type="text", text="Here it is."), NS(type="tool_use", id="new", name="search", input={"q": "x"})]
    hops = [
        NS(
            type="message", input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
        NS(
            type="fallback_message",
            input_tokens=110,
            output_tokens=30,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    ]
    usage = NS(
        input_tokens=110, output_tokens=30, cache_read_input_tokens=0, cache_creation_input_tokens=0, iterations=hops
    )
    turn = parse_anthropic_response(
        NS(content=[declined, boundary, *served], usage=usage, stop_reason="tool_use", model="b")
    )
    assert [c.id for c in turn.tool_calls] == ["new"] and turn.provider_state == served
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (210, 50)  # both attempts are paid for


def test_stored_provider_blocks_keep_their_wire_names() -> None:
    class Block(BaseModel):
        from_: str = Field(alias="from")
        type: str = "fallback"

    assert jsonable([Block(**{"from": "claude-a"})]) == [{"from": "claude-a", "type": "fallback"}]
