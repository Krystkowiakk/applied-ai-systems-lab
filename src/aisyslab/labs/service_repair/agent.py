"""Agent solver: one shell tool, one submit tool; the policy is enforced by the environment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import Tool, Toolbox
from .env import MAX_COMMAND, SERVICE, ServiceRepairEnv

SYSTEM = f"""You are a maintenance engineer's assistant on a production edge box.
The {SERVICE} service (it connects line 3 to the MES) no longer starts after a patch.

You work through a restricted shell, one command per call. It is not bash: run `help` first.
Some paths are off limits and touching them blocks the shell for a while; read the rules
and the deny list before you explore. Change only what the start check proves is wrong,
one line at a time, and run the service again after each fix. When it starts, submit the
confirmation code it prints."""

TASK = f"Get {SERVICE} running and submit its confirmation code."


class ShellArgs(BaseModel):
    command: str = Field(description=f"One restricted-shell command, at most {MAX_COMMAND} characters.")


class CodeArgs(BaseModel):
    code: str = Field(description="The confirmation code printed by a successful start.")


def solve(env: ServiceRepairEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    tools = Toolbox(
        [
            Tool(
                "shell",
                "Run one restricted-shell command and return its output or error.",
                ShellArgs,
                lambda a: ctx.tool_call(env.shell, a.command),
            ),
            Tool(
                "submit_confirmation",
                "Send the confirmation code to the verifier.",
                CodeArgs,
                lambda a: ctx.submit(env.submit, a.code, as_tool=True).as_dict(),
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
