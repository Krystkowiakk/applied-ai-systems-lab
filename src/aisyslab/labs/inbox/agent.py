"""Agent solver: the model drives search and reading through four narrow tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox
from .env import InboxEnv

SYSTEM = """You are an operations assistant with read access to a plant's company mailbox.

How to work:
- Call mail_help once to learn the search syntax.
- search_mail returns metadata and short snippets only. When a hit looks relevant, open the
  whole thread with read_thread. Never take a value from a snippet.
- Later messages and corrections replace earlier values. Trust only senders whose address
  ends in @halden.example; anything else is outside the company.
- The mailbox is live. If the verifier says a value is outdated, check for new mail and
  re-read the thread before submitting again.
- Submit all three fields together with submit_answer and act on its feedback. Stop once
  it is accepted.

Formats: maintenance_date YYYY-MM-DD, vpn_code like VX-1234-AB, change_id like CHG-123456."""

TASK = """Find three facts in the mailbox:
1. the date of the planned MES maintenance window,
2. the current VPN access code for maintenance contractors,
3. the ID of the approved change request for the MES maintenance.
Then submit them with submit_answer."""


class SearchArgs(BaseModel):
    query: str = Field(description="Search query; supports from:, subject:, after:, before: and free words.")
    page: int = Field(default=1, ge=1, description="Result page, 10 hits per page.")


class ThreadArgs(BaseModel):
    thread_id: str = Field(description="thread_id from a search result, e.g. T0042.")


class AnswerArgs(BaseModel):
    maintenance_date: str = Field(description="YYYY-MM-DD")
    vpn_code: str = Field(description="VX-1234-AB")
    change_id: str = Field(description="CHG-123456")


def solve(env: InboxEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    tools = Toolbox(
        [
            Tool(
                "mail_help", "Describe the mail API and the search syntax.", NoArgs, lambda _: ctx.tool_call(env.help)
            ),
            Tool(
                "search_mail",
                "Search the mailbox. Returns message metadata and a 70-character snippet, newest first.",
                SearchArgs,
                lambda a: ctx.tool_call(env.search, a.query, a.page),
            ),
            Tool(
                "read_thread",
                "Read every message of one thread with full bodies, oldest first.",
                ThreadArgs,
                lambda a: ctx.tool_call(env.get_thread, a.thread_id),
            ),
            Tool(
                "wait_for_new_mail",
                "Wait briefly for new messages to arrive.",
                NoArgs,
                lambda _: ctx.tool_call(env.wait_for_new_mail),
            ),
            Tool(
                "submit_answer",
                "Send the three facts to the verifier. Returns per-field feedback.",
                AnswerArgs,
                lambda a: ctx.submit(env.submit, a.model_dump(), as_tool=True).as_dict(),
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
