"""Agent solver: a deep research loop, then a report written section by section.

The loop is the agent's: search, fetch what matters, note each fact with its page, name what is
still missing and search again. The report is not one long generation: each section is written
by a separate call from the numbered notes, every sentence cites its source, and code assembles
the sections, lists the sources and checks that every citation points to a page that was fetched.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.messages import Message
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox, ToolError
from .env import PartResearchEnv

SYSTEM = """You research a parts question for a maintenance planner, in a loop: search, fetch the
pages that matter, note each fact with the page it came from, then name what is still missing (a
current datasheet, a current lead time, whether an alternative fits) and search again.

- Prefer current pages to archived ones, and a datasheet's current revision to a superseded one.
- A forum answer is a claim to check against a datasheet, not a fact.
- The crawler keeps to robots.txt; a page it refuses is not yours to read.
- When the notes answer the question, call write_report with a title and an outline of three to five
  headings, then submit_answer. Fix what the feedback names."""

TASK = "Answer the planner's question from the question tool."

SECTION_SYSTEM = """You write one section of a short research report for a maintenance planner.
Section heading: {heading}
Use only the numbered notes below and cite every sentence with its source number in square brackets,
for example [2]. If the notes do not cover the section, say so in one sentence.
Notes:
{notes}
Return only the section's text, without the heading."""


class SearchArgs(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=8)


class FetchArgs(BaseModel):
    url: str


class NoteArgs(BaseModel):
    url: str = Field(description="The fetched page the fact comes from.")
    fact: str = Field(description="One fact, in your words, with its numbers.")


class ReportArgs(BaseModel):
    title: str
    outline: list[str] = Field(min_length=1, max_length=5)


class Rejected(BaseModel):
    part: str
    reason: str


class AnswerArgs(BaseModel):
    replacement: str = Field(description="The part number only, e.g. VT-999.")
    lead_time_weeks: int
    max_pressure_bar: int
    rejected: list[Rejected] = Field(default_factory=list)


def toolbox(env: PartResearchEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    reports = [0]

    def write_report(args: ReportArgs) -> dict[str, Any]:
        if model is None:
            raise ToolError("No model is configured here to write the report.")
        if not env.notes:
            raise ToolError("Take notes first: the report is written from them.")
        reports[0] += 1
        if reports[0] > 2:
            raise ToolError("The report was written twice already; submit the answer.")
        sources = list(dict.fromkeys(n["url"] for n in env.notes))
        numbered = "\n".join(f"[{sources.index(n['url']) + 1}] {n['fact']}" for n in env.notes)
        sections = []
        for heading in args.outline:
            text = ctx.side_call(
                model,
                system=SECTION_SYSTEM.format(heading=heading, notes=numbered),
                messages=[Message(role="user", content="Write the section.")],
                purpose="report section",
            )
            sections.append(f"## {heading}\n\n{text.strip()}")
        listing = "\n".join(f"{i}. {url}" for i, url in enumerate(sources, 1))
        report = f"# {args.title}\n\n" + "\n\n".join(sections) + f"\n\n## Sources\n\n{listing}\n"
        stored = ctx.tool_call(env.store_report, report, sources)
        return stored | {"report": report}

    def submit(args: AnswerArgs) -> dict[str, Any]:
        rejected = [r.model_dump() for r in args.rejected]
        verdict = ctx.submit(
            env.submit, args.replacement, args.lead_time_weeks, args.max_pressure_bar, rejected, as_tool=True
        )
        return verdict.as_dict()

    return Toolbox(
        [
            Tool("question", "The planner's question.", NoArgs, lambda _: ctx.tool_call(env.question)),
            Tool(
                "web_search",
                "Search the web; returns URLs, titles and snippets.",
                SearchArgs,
                lambda a: ctx.tool_call(env.search, a.query, a.k),
            ),
            Tool(
                "fetch",
                "Fetch a page (robots.txt and a per-host interval are respected).",
                FetchArgs,
                lambda a: ctx.tool_call(env.fetch, a.url),
            ),
            Tool(
                "note",
                "Keep one fact with the fetched page it comes from.",
                NoteArgs,
                lambda a: ctx.tool_call(env.note, a.url, a.fact),
            ),
            Tool(
                "write_report",
                "Write the report from the notes, section by section, with checked citations.",
                ReportArgs,
                write_report,
            ),
            Tool("submit_answer", "Submit the answer. Returns what is right and what to fix.", AnswerArgs, submit),
        ]
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
