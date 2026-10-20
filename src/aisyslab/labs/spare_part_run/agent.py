"""Agent solver: the model proposes plans, a local simulator judges every one of them.

Sampling several different candidates is where a model's non-determinism helps; the
simulator makes sure none of them reaches the verifier unchecked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import Tool, Toolbox
from .env import MAX_COMMANDS, SparePartRunEnv

SYSTEM = f"""You plan an urgent internal delivery in a manufacturing plant: a spare pump must get
from the stores to the line 3 maintenance door, or the line stays down.

First find the documents you need with search_docs and read them: the site map, the vehicle
rules and the budget. A plan is a list of commands (up, down, left, right, dismount), at most
{MAX_COMMANDS}. Counting grid steps in your head is error-prone: draft several genuinely
different candidate routes, check each one with simulate_plan, read where it fails, and fix
it. Only submit a plan that the simulator says arrives within budget."""

TASK = "Plan the spare-pump run and submit it with submit_plan."


class QueryArgs(BaseModel):
    query: str = Field(description="Words describing the document you need.")


class DocArgs(BaseModel):
    name: str


class PlanArgs(BaseModel):
    commands: list[str] = Field(description="Commands in order, e.g. ['up', 'up', 'right', 'dismount', 'right'].")


def solve(env: SparePartRunEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    tools = Toolbox(
        [
            Tool(
                "search_docs",
                "Find site documents by keywords.",
                QueryArgs,
                lambda a: ctx.tool_call(env.search_docs, a.query),
            ),
            Tool(
                "read_doc", "Read a document by its exact name.", DocArgs, lambda a: ctx.tool_call(env.read_doc, a.name)
            ),
            Tool(
                "simulate_plan",
                "Run a plan through the site simulator. Returns arrival, minutes, battery, or the first failing step.",
                PlanArgs,
                lambda a: ctx.tool_call(env.simulate, a.commands),
            ),
            Tool(
                "submit_plan",
                "Submit the final plan to the dispatcher. It is checked by the same rules.",
                PlanArgs,
                lambda a: ctx.submit(env.submit, a.commands, as_tool=True).as_dict(),
            ),
        ]
    )
    return run_agent(
        model=model,
        system=SYSTEM,
        task=TASK,
        toolbox=tools,
        tracer=ctx.tracer,
        config=ctx.agent,
        goal_reached=lambda: env.accepted,
    )
