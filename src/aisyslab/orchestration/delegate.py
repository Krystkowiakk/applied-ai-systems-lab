"""Delegation: an orchestrator agent hands a task to a specialist that runs as its own agent.

The specialist starts with an empty context: only the task the orchestrator wrote and its own
lab's instructions. It has its own tools, its own budgets and its own verifier. It reports
through the shared workspace and a short final message, so the orchestrator's context grows
by a summary, not by the specialist's transcript. The specialist's model calls count toward
the run's cost, and its events land in the same trace, tagged with its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..core.agent import AgentConfig, AgentSpec, run_agent
from ..core.lab import RunContext
from ..core.model import ChatModel
from ..core.tools import Tool, ToolError
from .workspace import Workspace, ref

REPORTING = """

You work for an orchestrator that cannot see your conversation. When your own checks pass,
write your findings to the shared workspace as {doc} (workspace_write, expected_version 0)
and end with a one-line summary. Write facts the orchestrator can use without your context:
exact values and ids, no narrative."""


@dataclass(frozen=True)
class Specialist:
    name: str
    description: str
    spec: AgentSpec
    env: Any
    config: AgentConfig

    @property
    def document(self) -> str:
        return f"{self.name}.md"


class DelegateArgs(BaseModel):
    agent: str = Field(description="Which specialist to use.")
    task: str = Field(description="What to find out. The specialist sees nothing but this and its own tools.")


def delegate_tool(
    specialists: list[Specialist], model: ChatModel | None, ctx: RunContext, workspace: Workspace
) -> Tool:
    by_name = {s.name: s for s in specialists}
    roster = "; ".join(f"{s.name}: {s.description}" for s in specialists)

    def delegate(args: DelegateArgs) -> dict[str, Any]:
        specialist = by_name.get(args.agent)
        if specialist is None:
            raise ToolError(f"No specialist '{args.agent}'.", hint=f"Specialists: {roster}")
        if model is None:
            raise ToolError("Delegation needs a model.")
        child = RunContext(tracer=ctx.tracer.child(specialist.name), retry=ctx.retry, agent=specialist.config)
        toolbox = specialist.spec.toolbox(specialist.env, model, child).extended(workspace.tools(specialist.name))
        ctx.tracer.event("delegate", agent=specialist.name, task=args.task[:400])
        run = run_agent(
            model=model,
            system=specialist.spec.system + REPORTING.format(doc=ref(specialist.document)),
            task=f"{args.task}\n\nYour own assignment:\n{specialist.spec.task(specialist.env)}",
            toolbox=toolbox,
            tracer=child.tracer,
            config=specialist.config,
        )
        ctx.side_usage = ctx.side_usage + run.usage + child.side_usage
        ctx.side_calls += run.model_calls + child.side_calls
        return {
            "agent": specialist.name,
            "status": run.status,
            "verified_by_its_lab": specialist.spec.done(specialist.env),
            "summary": run.final_text.strip()[:600],
            "documents": [d for d in workspace.entries() if d["author"] == specialist.name],
        }

    return Tool("delegate", f"Hand a task to a specialist agent. {roster}", DelegateArgs, delegate)
