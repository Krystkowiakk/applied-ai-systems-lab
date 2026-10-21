"""Agent solver: the model chooses the method and the features; code fits and predicts.

The model can read rows, but numbers are judged by a decision tree it trains through a tool.
What the model adds is judgement: which columns describe the process, whether the learned
rules make physical sense, and what to do with a batch the tree cannot decide.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...core.agent import AgentRun, run_agent
from ...core.lab import RunContext
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox
from .env import QualityGateEnv

SYSTEM = """You assist the quality engineer at the end of the night shift: decide which of tonight's
batches pass the quality gate.

The quality lab judged earlier batches OK or NOK, but its rule was never written down. Do not
judge batches by reading numbers. Look at the data (describe_data, read_rows), choose the columns
that describe the process, fit a decision tree with train_tree, check that its rules make physical
sense, then decide tonight's batches with predict_tonight. A column that is only filled in after
the inspection cannot be a feature. A batch with a missing measurement is held, and passing a bad
batch is worse than holding a good one. Submit the IDs of the passing batches with submit_answer."""


class RowsArgs(BaseModel):
    dataset: Literal["history", "tonight"]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=25)


class TrainArgs(BaseModel):
    features: list[str] = Field(description="Columns to learn from, for example ['torque_nm'].")
    max_depth: int = Field(default=3, ge=1, le=5)


class PredictArgs(BaseModel):
    model_id: str = Field(description="The model_id returned by train_tree.")


class AnswerArgs(BaseModel):
    passing_batch_ids: list[str] = Field(description="IDs of tonight's batches that pass, e.g. ['B-261234'].")


def solve(env: QualityGateEnv, model: ChatModel, ctx: RunContext) -> AgentRun:
    tools = Toolbox(
        [
            Tool(
                "describe_data",
                "Both datasets: sizes, columns, missing values, and per-verdict summaries of every column.",
                NoArgs,
                lambda _: ctx.tool_call(env.describe),
            ),
            Tool(
                "read_rows",
                "Raw rows from 'history' (with the lab's verdict) or 'tonight', 25 at a time.",
                RowsArgs,
                lambda a: ctx.tool_call(env.rows, a.dataset, a.offset, a.limit),
            ),
            Tool(
                "train_tree",
                "Fit a decision tree on the history with the chosen columns; returns its rules and accuracy.",
                TrainArgs,
                lambda a: ctx.tool_call(env.train, a.features, a.max_depth),
            ),
            Tool(
                "predict_tonight",
                "Apply a trained tree to tonight's batches.",
                PredictArgs,
                lambda a: ctx.tool_call(env.predict, a.model_id),
            ),
            Tool(
                "submit_answer",
                "Submit the IDs of the passing batches; the quality lab checks them.",
                AnswerArgs,
                lambda a: ctx.submit(env.submit, a.passing_batch_ids, as_tool=True).as_dict(),
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
