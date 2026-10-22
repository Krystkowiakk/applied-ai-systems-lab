"""Wire-format conversions, checked offline with stand-in response objects."""

from __future__ import annotations

from types import SimpleNamespace as NS

from aisyslab.core.messages import ImagePart, Message, ToolCall
from aisyslab.providers.anthropic_messages import parse_anthropic_response, to_anthropic_messages
from aisyslab.providers.openai_chat import parse_openai_response, to_openai_messages

CALLS = [ToolCall("a1", "search", {"q": "x"}), ToolCall("a2", "read", {"id": 7})]
HISTORY = [
    Message(role="user", content="find it", images=[ImagePart("image/png", b"\x89PNG")]),
    Message(role="assistant", content="Looking.", tool_calls=CALLS),
    Message(role="tool", content='{"hits": 1}', tool_call_id="a1", tool_name="search"),
    Message(role="tool", content='{"error": "no"}', tool_call_id="a2", tool_name="read", is_error=True),
]


def test_openai_messages() -> None:
    out = to_openai_messages(HISTORY)
    assert out[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert [c["function"]["name"] for c in out[1]["tool_calls"]] == ["search", "read"]
    assert out[2] == {"role": "tool", "tool_call_id": "a1", "content": '{"hits": 1}'}
    assert len(out) == 4  # one tool message per call


def test_openai_response_with_bad_arguments() -> None:
    message = NS(
        content=None,
        refusal=None,
        tool_calls=[
            NS(id="t1", function=NS(name="search", arguments='{"q": "x"}')),
            NS(id="t2", function=NS(name="read", arguments="{oops")),
        ],
    )
    usage = NS(prompt_tokens=120, completion_tokens=30, prompt_tokens_details=NS(cached_tokens=100))
    turn = parse_openai_response(NS(choices=[NS(message=message, finish_reason="tool_calls")], usage=usage, model="m"))
    assert turn.stop_reason == "tool_use" and turn.tool_calls[0].arguments == {"q": "x"}
    assert turn.tool_calls[1].parse_error and turn.usage.cached_input_tokens == 100


def test_openai_refusal_and_length() -> None:
    refused = NS(content=None, refusal="I can't.", tool_calls=None)
    assert (
        parse_openai_response(NS(choices=[NS(message=refused, finish_reason="stop")], usage=None)).stop_reason
        == "refusal"
    )
    cut = NS(content="partial", refusal=None, tool_calls=None)
    assert (
        parse_openai_response(NS(choices=[NS(message=cut, finish_reason="length")], usage=None)).stop_reason
        == "max_tokens"
    )


def test_anthropic_groups_tool_results_in_one_user_message() -> None:
    out = to_anthropic_messages(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"][0]["type"] == "image"
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a1", "a2"]
    assert results[1]["is_error"] is True


def test_anthropic_replays_assistant_turns_verbatim() -> None:
    raw_blocks = [
        NS(type="thinking", thinking="", signature="sig"),
        NS(type="tool_use", id="a1", name="search", input={"q": "x"}),
    ]
    history = [
        Message(role="user", content="go"),
        Message(role="assistant", content="", tool_calls=CALLS[:1], provider_state=raw_blocks),
    ]
    assert to_anthropic_messages(history)[1]["content"] is raw_blocks


def test_anthropic_response_usage_and_stop() -> None:
    blocks = [
        NS(type="thinking", thinking=""),
        NS(type="text", text="Let me check."),
        NS(type="tool_use", id="t1", name="search", input={"q": "x"}),
    ]
    usage = NS(input_tokens=20, output_tokens=40, cache_read_input_tokens=900, cache_creation_input_tokens=80)
    turn = parse_anthropic_response(NS(content=blocks, usage=usage, stop_reason="tool_use", model="claude-opus-5-5"))
    assert turn.text == "Let me check." and turn.tool_calls[0].name == "search"
    assert turn.usage.input_tokens == 1000 and turn.usage.cached_input_tokens == 900
    assert turn.provider_state == blocks and turn.stop_reason == "tool_use"
    refusal = parse_anthropic_response(NS(content=[], usage=usage, stop_reason="refusal", model="m"))
    assert refusal.stop_reason == "refusal"


class _Manager:
    """Stands in for an SDK stream manager: a context manager around a stream."""

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __enter__(self) -> object:
        return self.stream

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_anthropic_stream_passes_text_on_and_parses_the_final_message() -> None:
    from aisyslab.providers.anthropic_messages import AnthropicModel

    final = NS(
        content=[NS(type="text", text="Hello world")],
        usage=NS(input_tokens=10, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        stop_reason="end_turn",
        model="claude-x",
    )
    stream = NS(text_stream=iter(["Hello ", "world"]), get_final_message=lambda: final)
    sent: dict = {}
    model = AnthropicModel.__new__(AnthropicModel)  # no SDK client: the transport is faked below
    model.model, model.effort, model.fallbacks = "claude-x", None, False
    model._anthropic = NS(APIStatusError=type("E1", (Exception,), {}), APIConnectionError=type("E2", (Exception,), {}))
    model._client = NS(messages=NS(stream=lambda **kw: sent.update(kw) or _Manager(stream)))
    pieces: list[str] = []
    turn = model.complete_stream(
        system="s", messages=[Message(role="user", content="hi")], tools=[], on_text=pieces.append
    )
    assert pieces == ["Hello ", "world"] and turn.text == "Hello world" and turn.usage.input_tokens == 10
    assert sent["system"] == "s" and sent["cache_control"] == {"type": "ephemeral"}


def test_openai_stream_passes_deltas_on_and_estimates_missing_usage() -> None:
    from aisyslab.providers.openai_chat import OpenAIChatModel

    final = NS(
        choices=[NS(message=NS(content="Hi there", refusal=None, tool_calls=None), finish_reason="stop")],
        usage=None,  # local servers may not report usage for a stream
        model="local-x",
    )
    events = [NS(type="chunk"), NS(type="content.delta", delta="Hi "), NS(type="content.delta", delta="there")]

    class Stream:
        def __iter__(self) -> object:
            return iter(events)

        def get_final_completion(self) -> object:
            return final

    sent: dict = {}
    model = OpenAIChatModel.__new__(OpenAIChatModel)
    model.model, model.reasoning_effort, model.temperature = "local-x", None, None
    model.token_param, model.stream_usage = "max_tokens", False
    model._openai = NS(APIError=type("APIError", (Exception,), {}))
    model._client = NS(chat=NS(completions=NS(stream=lambda **kw: sent.update(kw) or _Manager(Stream()))))
    pieces: list[str] = []
    turn = model.complete_stream(
        system="s", messages=[Message(role="user", content="hi")], tools=[], on_text=pieces.append
    )
    assert pieces == ["Hi ", "there"] and turn.text == "Hi there" and turn.usage.output_tokens > 0
    assert "stream_options" not in sent and sent["max_tokens"] == 4096
