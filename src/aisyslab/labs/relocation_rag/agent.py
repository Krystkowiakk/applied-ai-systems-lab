"""Agent solver: agentic search over the handbook, answers with citations, refusal when unsupported."""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.calculator import calculator_tool
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox, ToolError
from .env import RelocationRagEnv

SYSTEM = """You are a relocation assistant for people moving to Arvenland for work. You answer only
from the relocation handbook, which you reach through search_handbook and read_section.

Rules:
- Search with English keywords (the handbook is in English) even when the question is not,
  and pass the full question as well; then read the sections you rely on.
- Prefer sections with status "current". A superseded edition is history, not an answer.
- Check exceptions: a later sentence can override the general rule for the asker's case.
- Do date and number arithmetic with add_days and calculate, never in your head.
- If the handbook does not answer a question, the value is "not covered" and you cite nothing.
  Do not fill gaps from general knowledge.

Submit all answers at once with submit_answers. "value" holds only the bare answer: a number
without units or separators, a date as YYYY-MM-DD, yes or no, or not covered. Cite the
section_id values that support each answer. Fix what the feedback flags and resubmit."""

TASK = "Answer every question from list_questions and submit the answers."


class SearchArgs(BaseModel):
    keywords: str = Field(description="English keywords for exact matching, e.g. 'salary threshold graduate'.")
    question: str | None = Field(default=None, description="The full question, used for similarity ranking.")
    k: int = Field(default=5, ge=1, le=8)


class SectionArgs(BaseModel):
    section_id: str


class DaysArgs(BaseModel):
    start: str = Field(description="YYYY-MM-DD")
    days: int


class AnswerItem(BaseModel):
    question_id: str
    value: str = Field(description="Bare value: digits, YYYY-MM-DD, yes, no, or not covered.")
    answer: str = Field(description="One-sentence answer for the person asking.")
    citations: list[str] = Field(default_factory=list, description="section_id values that support the answer.")


class AnswersArgs(BaseModel):
    answers: list[AnswerItem]


def solve(env: RelocationRagEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    def add_days(args: DaysArgs) -> dict[str, str]:
        try:
            start = date.fromisoformat(args.start)
        except ValueError as exc:
            raise ToolError("start must be YYYY-MM-DD.") from exc
        return {"start": args.start, "days": str(args.days), "result": (start + timedelta(days=args.days)).isoformat()}

    tools = Toolbox(
        [
            Tool(
                "list_questions",
                "The questions to answer, with their ids.",
                NoArgs,
                lambda _: ctx.tool_call(env.list_questions),
            ),
            Tool(
                "search_handbook",
                "Hybrid search (keywords plus similarity, fused by rank). Returns section ids, edition, status and snippets.",
                SearchArgs,
                lambda a: ctx.tool_call(env.search, a.keywords, a.question, a.k),
            ),
            Tool(
                "read_section",
                "Full text and metadata of one handbook section.",
                SectionArgs,
                lambda a: ctx.tool_call(env.read_section, a.section_id),
            ),
            Tool("add_days", "Add a number of days to a date.", DaysArgs, add_days),
            calculator_tool(),
            Tool(
                "submit_answers",
                "Submit all answers. Returns a score and per-question feedback.",
                AnswersArgs,
                lambda a: ctx.submit(env.submit, [item.model_dump() for item in a.answers], as_tool=True).as_dict(),
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
