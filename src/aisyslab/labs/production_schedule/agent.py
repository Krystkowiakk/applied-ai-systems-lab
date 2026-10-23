"""Agent solver: code mode. The agent writes a program; the program plans; code judges the plan.

Planning a day on three presses is search, and a model is not a search engine: it reads the
planner's notes, turns them into constraints and writes a program that calls the ``plant``
library (``schedule``, ``check``, ``propose``) to search. The program runs in the sandbox.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox
from .env import ProductionScheduleEnv

SYSTEM = """You plan tomorrow's production on three presses. get_plan gives the orders, the presses and
the families each can run, the changeover times, the maintenance window, the horizon and the
planner's notes. Times are minutes after 06:00.

Do not plan in your head: write Python and run it with run_python. Your code can `import plant`:
- plant.data() returns the same data;
- plant.schedule(sequence, machine_for=None, after=None, cutoff=None) places the orders in the
  given sequence, each on the press where it ends earliest; machine_for pins orders to presses,
  after={later: first} makes an order wait for another, cutoff={press: minute} ends a press early;
- plant.check(plan) returns feasibility and total lateness on the data's rules (not the notes);
- plant.propose(plan) prints a plan as your proposal.
Only the standard library is available; there is no network.

The notes are rules too: turn each into a constraint in your code, check it yourself, then
search for low total lateness (for example by trying orders of the sequence). Submit the best
plan with submit_schedule; fix what the feedback names and resubmit."""

TASK = "Plan tomorrow's orders on the three presses and submit the plan."


class CodeArgs(BaseModel):
    code: str = Field(description="A complete Python program; standard library and `import plant` only.")


class Placement(BaseModel):
    order: str
    machine: str
    start: int = Field(description="Minutes after 06:00.")


class PlanArgs(BaseModel):
    plan: list[Placement]


def toolbox(env: ProductionScheduleEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    def submit(args: PlanArgs) -> dict[str, Any]:
        return ctx.submit(env.submit, [p.model_dump() for p in args.plan], as_tool=True).as_dict()

    return Toolbox(
        [
            Tool(
                "get_plan",
                "Tomorrow's orders, presses, changeovers, maintenance and the planner's notes.",
                NoArgs,
                lambda _: ctx.tool_call(env.get_plan),
            ),
            Tool(
                "run_python",
                "Run a Python program in the sandbox (15 s, standard library, `import plant`). Returns its output "
                "and, if it printed a proposal, the check of that plan on the data's rules.",
                CodeArgs,
                lambda a: ctx.tool_call(env.run_code, a.code),
            ),
            Tool(
                "submit_schedule", "Submit the plan. Returns feasibility, lateness and what to fix.", PlanArgs, submit
            ),
        ],
        max_output_chars=8000,
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
