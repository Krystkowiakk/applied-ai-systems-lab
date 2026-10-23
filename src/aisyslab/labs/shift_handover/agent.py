"""Agent solver: an assistant that builds observational memory over the week, then answers from it.

The agent never reads a log. ``observe_next_shift`` hands the next shift's log to the observer
(a separate model call) and returns only how many observations were kept; the log itself is
sealed. The reflector runs inside the same tool once the new observations pass a threshold, and
once more on request. After sealing, the agent answers from the handover record and recall.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.messages import Message
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox, ToolError
from ...memory.observational import ObservationalMemory
from .env import HANDOVER_BUDGET, ShiftHandoverEnv

SYSTEM = f"""You are the handover assistant of a plant's maintenance team.

First build the week's memory: call observe_next_shift until no shifts are left. You never see the
logs; an observer reads each one and keeps observations. Then call reflect once, so the handover
record (at most {HANDOVER_BUDGET} tokens) covers the whole week, and seal_logs.

Then answer the incoming shift lead's questions from list_questions, using read_handover and
recall only:
- When items conflict, the newest shift wins.
- A suspicion nobody checked is not a confirmed fault.
- Keep lines and machines apart: the same sensor name on two lines is two different sensors.
- If the memory does not say, the answer is "not recorded".

Submit all answers at once with submit_answers. "value" holds only the bare answer: yes or no, a
number, a part number, a person's first name, a shift such as "Tue night", a list of machine names,
or not recorded."""

TASK = "Build the handover memory from this week's shift logs, then answer the questions and submit the answers."


class RecallArgs(BaseModel):
    query: str = Field(description="What you are looking for, in a few words.")
    scope: str | None = Field(
        default=None, description="Optional: 'plant', 'line-3', 'asset:press 4' or 'person:<first name>'."
    )
    k: int = Field(default=5, ge=1, le=10)


class AnswerItem(BaseModel):
    question_id: str
    value: str | list[str] = Field(
        description="Bare value; a list of machine names where the question asks which machines."
    )
    answer: str = Field(default="", description="One sentence for the shift lead.")


class AnswersArgs(BaseModel):
    answers: list[AnswerItem]


def toolbox(env: ShiftHandoverEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    def call(system: str, user: str, purpose: str) -> str:
        if model is None:
            raise ToolError("No model is configured here for the observer and reflector.")
        return ctx.side_call(model, system=system, messages=[Message(role="user", content=user)], purpose=purpose)

    memory = ObservationalMemory(call, budget_tokens=HANDOVER_BUDGET, reflect_above=500)

    asked = [0]

    def reflect() -> dict[str, Any]:
        result: dict[str, Any] = ctx.tool_call(env.write_handover, memory.reflect())
        if memory.cut_lines:
            result["cut_lines"] = memory.cut_lines  # the record still did not fit; code cut it
        return result

    def reflect_on_request(_: NoArgs) -> dict[str, Any]:
        asked[0] += 1
        if asked[0] > 3:  # a bound on cost, whatever the model decides
            raise ToolError("Already reflected three times; the record is final. Seal the logs and answer.")
        return reflect()

    def observe_next_shift(_: NoArgs) -> dict[str, Any]:
        shift = ctx.tool_call(env.next_shift)
        if shift is None:
            return {"done": True, "hint": "All shifts observed. Call reflect, then seal_logs."}
        observations, problems = memory.observe(shift["shift"], shift["log"])
        ctx.tool_call(
            env.keep, [{"when": o.when, "scope": o.scope, "status": o.status, "text": o.text} for o in observations]
        )
        result: dict[str, Any] = {
            "shift": shift["shift"],
            "kept": len(observations),
            "shifts_left": shift["shifts_left"],
        }
        if problems:
            result["skipped"] = problems[:5]
        if memory.due():  # the mechanism, not the model, decides when to condense
            result["reflected"] = reflect()
        return result

    return Toolbox(
        [
            Tool(
                "observe_next_shift",
                "Hand the next shift's log to the observer; returns what was kept.",
                NoArgs,
                observe_next_shift,
            ),
            Tool(
                "reflect",
                "Condense the memory into the handover record within its budget.",
                NoArgs,
                reflect_on_request,
            ),
            Tool(
                "seal_logs",
                "Close the logs once every shift is observed; the questions open.",
                NoArgs,
                lambda _: ctx.tool_call(env.seal),
            ),
            Tool(
                "list_questions",
                "The shift lead's questions, with their ids.",
                NoArgs,
                lambda _: ctx.tool_call(env.list_questions),
            ),
            Tool("read_handover", "The handover record.", NoArgs, lambda _: ctx.tool_call(env.read_handover)),
            Tool(
                "recall",
                "Search the kept observations; a scope narrows the search to one line, machine or person.",
                RecallArgs,
                lambda a: ctx.tool_call(env.recall, a.query, a.scope, a.k),
            ),
            Tool(
                "submit_answers",
                "Submit all answers. Returns a score and per-question feedback.",
                AnswersArgs,
                lambda a: ctx.submit(env.submit, [item.model_dump() for item in a.answers], as_tool=True).as_dict(),
            ),
        ]
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
