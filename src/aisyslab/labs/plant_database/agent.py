"""Agent solver: discover the schema, compute in SQL, submit, act on the reviewer's feedback.

The model sees the database only through three read tools. The database itself refuses
writes, so the prompt explains the rule but does not guard it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox
from .env import ROW_LIMIT, PlantDatabaseEnv

SYSTEM = f"""You answer a maintenance planner's questions from the plant database (SQLite).

You can only read. Explore before you answer: list_tables, then describe_table for the tables
that matter (each shows its columns, keys and sample rows). Map every term in the question to
columns, then compute the answer in SQL, in one query where possible; do not do arithmetic in
your head. Prefer what the records say over summary or status fields maintained by hand.

run_query returns at most {ROW_LIMIT} rows; aggregate or filter instead of paging. Submit the
work order IDs with submit_answer. If the reviewer rejects the answer, its feedback names the
kind of mistake: fix the query, not the list by hand."""


class TableArgs(BaseModel):
    table: str = Field(description="A table name from list_tables.")


class QueryArgs(BaseModel):
    sql: str = Field(description="One SELECT statement in the SQLite dialect; WITH clauses are fine.")
    limit: int = Field(default=ROW_LIMIT, ge=1, le=ROW_LIMIT, description="Maximum rows to return.")


class AnswerArgs(BaseModel):
    work_order_ids: list[str] = Field(description="Work order IDs, for example ['WO-2101', 'WO-2140'].")


def solve(env: PlantDatabaseEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    tools = Toolbox(
        [
            Tool(
                "list_tables",
                "Tables in the plant database with their row counts.",
                NoArgs,
                lambda _: ctx.tool_call(env.tables),
            ),
            Tool(
                "describe_table",
                "A table's definition (columns, keys) and three sample rows.",
                TableArgs,
                lambda a: ctx.tool_call(env.describe, a.table),
            ),
            Tool(
                "run_query",
                "Run one read-only SELECT and return the rows.",
                QueryArgs,
                lambda a: ctx.tool_call(env.query, a.sql, a.limit),
            ),
            Tool(
                "submit_answer",
                "Submit the work order IDs; the reviewer checks them.",
                AnswerArgs,
                lambda a: ctx.submit(env.submit, a.work_order_ids, as_tool=True).as_dict(),
            ),
        ]
    )
    return run_agent(
        model=model,
        system=SYSTEM,
        task=env.instruction,
        toolbox=tools,
        tracer=ctx.tracer,
        config=ctx.agent,
        goal_reached=lambda: env.accepted,
    )
