from __future__ import annotations

import json

from pydantic import BaseModel, Field

from aisyslab.core.messages import ToolCall
from aisyslab.core.tools import NoArgs, Tool, Toolbox, ToolError, bounded


class EchoArgs(BaseModel):
    text: str = Field(description="Anything.")
    times: int = Field(default=1, ge=1, le=3)


class Inner(BaseModel):
    title: str


class Outer(BaseModel):
    items: list[Inner]


def make_box(**kwargs: object) -> Toolbox:
    def boom(_: NoArgs) -> None:
        raise KeyError("missing")

    def refuse(_: NoArgs) -> None:
        raise ToolError("Not allowed here.", hint="Ask first.", code=403)

    return Toolbox(
        [
            Tool("echo", "Echo text.", EchoArgs, lambda a: {"echo": a.text * a.times}),
            Tool("boom", "Crashes.", NoArgs, boom),
            Tool("refuse", "Refuses.", NoArgs, refuse),
            Tool("big", "Returns a lot.", NoArgs, lambda _: "x" * 500),
        ],
        **kwargs,  # type: ignore[arg-type]
    )


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=dict(arguments))


def test_valid_call_returns_json() -> None:
    result = make_box().execute(call("echo", text="ab", times=2))
    assert not result.is_error
    assert json.loads(result.content) == {"echo": "abab"}


def test_invalid_arguments_come_back_as_data() -> None:
    result = make_box().execute(call("echo", text="a", times=9))
    assert result.is_error
    assert "times" in json.loads(result.content)["error"]


def test_unknown_tool_lists_alternatives() -> None:
    payload = json.loads(make_box().execute(call("nope")).content)
    assert "Unknown tool" in payload["error"] and "echo" in payload["hint"]


def test_unparseable_arguments_are_reported() -> None:
    result = make_box().execute(ToolCall(id="c", name="echo", arguments={}, parse_error="Expecting value"))
    assert result.is_error and "not valid JSON" in result.content


def test_expected_failure_keeps_hint_and_data() -> None:
    payload = json.loads(make_box().execute(call("refuse")).content)
    assert payload == {"error": "Not allowed here.", "hint": "Ask first.", "code": 403}


def test_bug_in_a_tool_does_not_crash_the_loop() -> None:
    result = make_box().execute(call("boom"))
    assert result.is_error and result.internal_error and "KeyError" in result.content


def test_output_is_bounded_with_a_hint() -> None:
    result = make_box(max_output_chars=100).execute(call("big"))
    assert result.truncated and "Narrow the query" in result.content
    assert bounded("short", 10) == ("short", False)


def test_schema_is_clean_and_closed() -> None:
    spec = Tool("t", "d", Outer, lambda a: a).spec()
    text = json.dumps(spec.parameters)
    assert "$ref" not in text and "$defs" not in text
    item = spec.parameters["properties"]["items"]["items"]
    assert item["additionalProperties"] is False
    assert "title" in item["properties"]  # a field called "title" survives the title stripping
    assert "title" not in spec.parameters
