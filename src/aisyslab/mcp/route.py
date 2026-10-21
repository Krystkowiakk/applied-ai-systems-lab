"""Route a lab's tools to its agent through MCP, through tool discovery, or both.

``via_mcp`` serves the lab's toolbox from an in-process MCP server and hands the agent the
host's merged, prefixed tools, so every call crosses the protocol exactly as it would for a
remote server. ``discover`` replaces the list of tools with ``search_tools`` and
``call_tool``. The environment, verifier and budgets stay the same, so results stay comparable.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..core.agent import AgentRun, run_agent
from ..core.lab import Lab
from ..core.model import ChatModel
from .discovery import discovery_toolbox


def server_prefix(lab: Lab) -> str:
    return lab.id.replace("-", "_")


def routed(lab: Lab, *, via_mcp: bool = False, discover: bool = False) -> Lab:
    """A copy of the lab whose agent gets its tools through MCP and/or discovery."""
    if not (via_mcp or discover):
        return lab
    spec = lab.spec
    if spec is None:
        raise ValueError(f"Lab '{lab.id}' has no agent spec.")

    def solve(env: Any, model: ChatModel, ctx: Any) -> AgentRun:
        host = None
        try:
            if via_mcp:
                from .host import McpHost
                from .server import lab_server

                served = lab_server(lab, env=env, ctx=ctx, model=model)
                host = McpHost({server_prefix(lab): served.server})
                ctx.tracer.event("mcp_audit", servers=[dataclasses.asdict(a) for a in host.audit()])
                box = host.toolbox()
            else:
                box = spec.toolbox(env, model, ctx)
            if discover:
                box = discovery_toolbox(box)
            return run_agent(
                model=model,
                system=spec.system,
                task=spec.task(env),
                toolbox=box,
                tracer=ctx.tracer,
                config=ctx.agent,
                goal_reached=lambda: spec.done(env),
            )
        finally:
            if host is not None:
                host.close()

    return dataclasses.replace(lab, agent=solve)
