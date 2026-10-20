"""A bounded tool-use loop.

Every limit lives in code, not in the prompt: model calls, tool calls and cumulative input
tokens. The loop never edits earlier turns (providers that sign reasoning blocks reject
edited history), so context control is done by compaction that replaces the whole history
with an observation log, the way observational memory does it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .messages import Message, Usage
from .model import ChatModel, ModelError
from .tokens import estimate_tokens
from .tools import Toolbox
from .trace import Tracer, preview

AgentStatus = Literal[
    "finished",  # the model stopped calling tools
    "goal_reached",  # the environment signalled success (e.g. the verifier accepted)
    "step_budget",
    "tool_budget",
    "token_budget",
    "refused",
    "truncated",  # the final answer hit max_output_tokens
    "model_error",
]

OBSERVER_SYSTEM = """You maintain the working memory of an agent whose context is about to be reset.
From the transcript, write compact observations: facts established (exact values, ids, dates),
actions taken and their results, verifier feedback received, and what is still missing.
Do not speculate and do not solve the task. At most 250 words."""


@dataclass
class AgentConfig:
    max_steps: int = 25
    max_tool_calls: int = 60
    max_input_tokens: int = 300_000
    max_output_tokens: int = 4096
    compact_above_tokens: int | None = None


@dataclass
class AgentRun:
    status: AgentStatus
    final_text: str
    model_calls: int
    tool_calls: int
    usage: Usage
    compactions: int
    error: str | None
    messages: list[Message]


def run_agent(
    *,
    model: ChatModel,
    system: str,
    task: str,
    toolbox: Toolbox,
    tracer: Tracer,
    config: AgentConfig | None = None,
    goal_reached: Callable[[], bool] = lambda: False,
) -> AgentRun:
    config = config or AgentConfig()
    specs = toolbox.specs()
    messages: list[Message] = [Message(role="user", content=task)]
    usage = Usage()
    model_calls = tool_calls = compactions = 0
    status: AgentStatus = "step_budget"
    final_text = ""
    error: str | None = None
    tracer.event("agent_start", model=model.name, tools=[s.name for s in specs], config=asdict(config))

    for step in range(1, config.max_steps + 1):
        if usage.input_tokens >= config.max_input_tokens:
            status = "token_budget"
            break
        try:
            turn = model.complete(
                system=system, messages=messages, tools=specs, max_output_tokens=config.max_output_tokens
            )
        except ModelError as exc:
            status, error = "model_error", str(exc)
            tracer.event("model_error", step=step, error=error)
            break
        model_calls += 1
        usage = usage + turn.usage
        tracer.event(
            "generation",
            step=step,
            model=turn.model,
            input_tokens=turn.usage.input_tokens,
            output_tokens=turn.usage.output_tokens,
            cached_input_tokens=turn.usage.cached_input_tokens,
            stop_reason=turn.stop_reason,
            text=preview(turn.text),
            tool_calls=[c.name for c in turn.tool_calls],
        )
        messages.append(
            Message(
                role="assistant",
                content=turn.text,
                tool_calls=list(turn.tool_calls),
                provider_state=turn.provider_state,
            )
        )
        if turn.stop_reason == "refusal":
            status, final_text = "refused", turn.text
            break
        if not turn.tool_calls:
            final_text = turn.text
            status = "truncated" if turn.stop_reason == "max_tokens" else "finished"
            break

        for call in turn.tool_calls:
            result = toolbox.execute(call)
            tool_calls += 1
            tracer.event(
                "tool",
                step=step,
                name=call.name,
                arguments=call.arguments,
                is_error=result.is_error,
                internal_error=result.internal_error,
                truncated=result.truncated,
                duration_ms=round(result.duration_ms, 1),
                result=preview(result.content, 400),
            )
            messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    is_error=result.is_error,
                )
            )

        if goal_reached():
            status = "goal_reached"
            break
        if tool_calls >= config.max_tool_calls:
            status = "tool_budget"
            break
        if config.compact_above_tokens and turn.usage.input_tokens > config.compact_above_tokens:
            try:
                messages, observer_usage = _compact(model, task, messages, tracer)
            except ModelError as exc:
                status, error = "model_error", str(exc)
                tracer.event("model_error", step=step, error=error)
                break
            model_calls += 1
            compactions += 1
            usage = usage + observer_usage

    tracer.event("agent_end", status=status, model_calls=model_calls, tool_calls=tool_calls, **asdict(usage))
    return AgentRun(status, final_text, model_calls, tool_calls, usage, compactions, error, messages)


def _compact(model: ChatModel, task: str, messages: Sequence[Message], tracer: Tracer) -> tuple[list[Message], Usage]:
    transcript = render_transcript(messages)
    turn = model.complete(
        system=OBSERVER_SYSTEM,
        messages=[Message(role="user", content=transcript)],
        tools=[],
        max_output_tokens=1024,
    )
    observations = turn.text.strip()
    fresh = [
        Message(
            role="user",
            content=(
                f"{task}\n\n<observations>\n{observations}\n</observations>\n"
                "Earlier turns were compacted into the observations above. Continue from there."
            ),
        )
    ]
    tracer.event(
        "compaction",
        before_tokens=estimate_tokens(transcript),
        after_tokens=estimate_tokens(fresh[0].content),
        observations=preview(observations, 400),
    )
    return fresh, turn.usage


def render_transcript(messages: Sequence[Message], per_message_chars: int = 1500) -> str:
    lines: list[str] = []
    for m in messages:
        if m.role == "assistant":
            if m.content:
                lines.append(f"ASSISTANT: {m.content[:per_message_chars]}")
            for call in m.tool_calls:
                lines.append(f"CALL {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")
        elif m.role == "tool":
            label = "TOOL ERROR" if m.is_error else "TOOL RESULT"
            lines.append(f"{label} ({m.tool_name}): {m.content[:per_message_chars]}")
        else:
            lines.append(f"USER: {m.content[:per_message_chars]}")
    return "\n".join(lines)
