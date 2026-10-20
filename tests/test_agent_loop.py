from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from aisyslab.core.agent import AgentConfig, run_agent
from aisyslab.core.messages import Message, ModelTurn, ToolSpec, Usage
from aisyslab.core.model import ModelError, ScriptedModel, text_turn, tool_turn
from aisyslab.core.tools import Tool, Toolbox
from aisyslab.core.trace import Tracer


class AddArgs(BaseModel):
    a: int
    b: int


def toolbox(log: list[int] | None = None) -> Toolbox:
    def add(args: AddArgs) -> dict[str, int]:
        if log is not None:
            log.append(args.a + args.b)
        return {"sum": args.a + args.b}

    return Toolbox([Tool("add", "Add two integers.", AddArgs, add)])


def run(model: ScriptedModel, **config: int) -> tuple:
    tracer = Tracer()
    result = run_agent(
        model=model, system="s", task="t", toolbox=toolbox(), tracer=tracer, config=AgentConfig(**config)
    )
    return result, tracer


def test_tool_then_answer() -> None:
    model = ScriptedModel([tool_turn("add", a=2, b=3), text_turn("The sum is 5.")])
    result, tracer = run(model)
    assert result.status == "finished" and result.final_text == "The sum is 5."
    assert result.model_calls == 2 and result.tool_calls == 1
    tool_message = result.messages[2]
    assert tool_message.role == "tool" and '"sum": 5' in tool_message.content
    assert tracer.count("generation") == 2 and tracer.count("tool") == 1


def test_step_budget_is_enforced_in_code() -> None:
    model = ScriptedModel(lambda *_: tool_turn("add", a=1, b=1))
    result, _ = run(model, max_steps=4)
    assert result.status == "step_budget" and result.model_calls == 4


def test_tool_budget() -> None:
    model = ScriptedModel(lambda *_: tool_turn("add", a=1, b=1))
    result, _ = run(model, max_steps=50, max_tool_calls=3)
    assert result.status == "tool_budget" and result.tool_calls == 3


def test_token_budget() -> None:
    model = ScriptedModel(lambda *_: tool_turn("add", a=1, b=1))
    result, _ = run(model, max_steps=50, max_input_tokens=1)
    assert result.status == "token_budget" and result.model_calls == 1


def test_goal_reached_stops_the_loop() -> None:
    seen: list[int] = []
    model = ScriptedModel(lambda *_: tool_turn("add", a=2, b=2))
    result = run_agent(
        model=model, system="s", task="t", toolbox=toolbox(seen), tracer=Tracer(), goal_reached=lambda: len(seen) >= 2
    )
    assert result.status == "goal_reached" and len(seen) == 2


def test_refusal_and_model_errors_end_the_run() -> None:
    refusal = ModelTurn(text="I can't help with that.", tool_calls=[], usage=Usage(), stop_reason="refusal")
    assert run(ScriptedModel([refusal]))[0].status == "refused"

    class Broken:
        name = "broken"

        def complete(self, **_: object) -> ModelTurn:
            raise ModelError("401 Unauthorized")

    result = run_agent(model=Broken(), system="s", task="t", toolbox=toolbox(), tracer=Tracer())
    assert result.status == "model_error" and "401" in (result.error or "")


def test_bad_arguments_let_the_model_recover() -> None:
    model = ScriptedModel([tool_turn("add", a="two", b=3), tool_turn("add", a=2, b=3), text_turn("5")])
    result, _ = run(model)
    assert result.messages[2].is_error and not result.messages[4].is_error
    assert result.status == "finished"


def test_compaction_replaces_history_with_observations() -> None:
    def policy(system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        if not tools:  # the observer call
            return text_turn("Observed: 1+1=2 already computed.")
        if any("<observations>" in m.content for m in messages):
            return text_turn("done")
        return tool_turn("add", a=1, b=1)

    tracer = Tracer()
    result = run_agent(
        model=ScriptedModel(policy),
        system="s",
        task="the task",
        toolbox=toolbox(),
        tracer=tracer,
        config=AgentConfig(compact_above_tokens=1),
    )
    assert result.compactions == 1 and result.status == "finished"
    assert result.messages[0].content.startswith("the task") and "Observed" in result.messages[0].content
    assert tracer.count("compaction") == 1
