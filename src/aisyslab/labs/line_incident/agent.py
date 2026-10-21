"""Agent solver: the model never sees the raw log, only what its queries return."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tokens import estimate_tokens
from ...core.tools import NoArgs, Tool, Toolbox
from .env import LINE_LIMIT, TOKEN_LIMIT, LineIncidentEnv

SYSTEM = f"""You are the on-call assistant for a manufacturing plant. Line 3 stopped overnight.
You cannot read the whole log; query it with the tools.

Deliver a timeline the on-call engineer can diagnose from:
- at most {LINE_LIMIT} lines and {TOKEN_LIMIT} tokens; check with count_tokens before submitting,
- one event per line, formatted exactly: YYYY-MM-DD HH:MM [LEVEL] component: short message,
- every line must be a real event from the log (same component, minute and level); shorten
  the message freely, never invent an event,
- chronological order; cover every system in the chain of cause and effect that stopped the
  line, and leave out chronic noise that also occurs on normal days.

Start with log_overview, find the first critical event, then look around it. The reviewer's
feedback tells you what is still missing; extend the timeline only where it says so."""

TASK = "Investigate the line-3 stoppage and submit the timeline with submit_timeline."


class SearchArgs(BaseModel):
    component: str | None = Field(default=None, description="Exact component name, e.g. furnace-3.")
    min_level: Literal["INFO", "WARN", "ERROR", "CRIT"] = Field(default="WARN")
    contains: str | None = Field(default=None, description="Case-insensitive text the message must contain.")
    start: str | None = Field(default=None, description="HH:MM, inclusive.")
    end: str | None = Field(default=None, description="HH:MM, inclusive.")
    collapse_repeats: bool = Field(default=True, description="Group identical messages with first/last time and count.")
    limit: int = Field(default=30, ge=1, le=60)


class TextArgs(BaseModel):
    text: str


def toolbox(env: LineIncidentEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    return Toolbox(
        [
            Tool(
                "log_overview",
                "Counts per level and component, the list of components, first CRIT time.",
                NoArgs,
                lambda _: ctx.tool_call(env.overview),
            ),
            Tool(
                "search_log",
                "Query log events by component, level, text and time window. Repeated messages are grouped by default.",
                SearchArgs,
                lambda a: ctx.tool_call(
                    env.search, a.component, a.min_level, a.contains, a.start, a.end, a.collapse_repeats, a.limit
                ),
            ),
            Tool(
                "count_tokens",
                f"Count tokens of a draft timeline the same way the reviewer does (limit {TOKEN_LIMIT}).",
                TextArgs,
                lambda a: {
                    "tokens": estimate_tokens(a.text),
                    "limit": TOKEN_LIMIT,
                    "lines": len(a.text.strip().splitlines()),
                },
            ),
            Tool(
                "submit_timeline",
                "Send the timeline to the on-call reviewer. Returns accepted or a list of problems.",
                TextArgs,
                lambda a: ctx.submit(env.submit, a.text, as_tool=True).as_dict(),
            ),
        ]
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
