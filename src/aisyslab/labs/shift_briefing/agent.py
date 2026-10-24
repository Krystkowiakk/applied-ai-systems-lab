"""Agent solver: an orchestrator with three specialist sub-agents and a shared workspace.

The orchestrator has no access to the data. It delegates each question to a specialist that
runs as its own agent (its own context, tools, budgets and verifier), reads what the
specialists write to the shared workspace, and submits the briefing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import Tool, Toolbox
from ...orchestration.delegate import Specialist, delegate_tool
from ...orchestration.workspace import Workspace
from ..line_incident import LAB as INCIDENT
from ..plant_database import LAB as PARTS
from ..quality_gate import LAB as QUALITY
from .env import ShiftBriefingEnv

SYSTEM = """You coordinate the 06:00 operations briefing. Three specialists can investigate for
you. Each works in its own context with its own data and tools; none can see the others or
this conversation.

Delegate each question to the right specialist with a precise task: what to find and what to
write in the workspace. Read the documents they write (workspace_read), check that each answer
is complete, then submit the briefing with submit_briefing. You have no access to the data
yourself: only to the specialists and the workspace. If the reviewer rejects a section, send
that specialist back with the feedback."""


class BriefingArgs(BaseModel):
    root_cause: str = Field(description="The system where the stoppage started, named as the log names it.")
    blocked_work_orders: list[str] = Field(description="Work order IDs blocked by a missing part.")
    held_batches: list[str] = Field(description="IDs of tonight's batches held at the quality gate.")


def specialists(env: ShiftBriefingEnv) -> list[Specialist]:
    assert INCIDENT.spec and PARTS.spec and QUALITY.spec
    return [
        Specialist(
            "incident_analyst",
            "reads the line-3 log; finds where the stoppage started",
            INCIDENT.spec,
            env.incident,
            INCIDENT.agent_config,
        ),
        Specialist(
            "parts_planner",
            "queries the plant database; finds work orders blocked by a missing part",
            PARTS.spec,
            env.parts,
            PARTS.agent_config,
        ),
        Specialist(
            "quality_engineer",
            "learns the quality lab's rule; decides which of tonight's batches are held",
            QUALITY.spec,
            env.quality,
            QUALITY.agent_config,
        ),
    ]


def toolbox(env: ShiftBriefingEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    workspace = Workspace()
    return Toolbox(
        [
            delegate_tool(specialists(env), model, ctx, workspace),
            *workspace.tools("orchestrator"),
            Tool(
                "submit_briefing",
                "Submit the three answers; the reviewer says which sections are wrong.",
                BriefingArgs,
                lambda a: ctx.submit(
                    env.submit_briefing, a.root_cause, a.blocked_work_orders, a.held_batches, as_tool=True
                ).as_dict(),
            ),
        ]
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda env: env.instruction, toolbox=toolbox)
solve = SPEC.solve
