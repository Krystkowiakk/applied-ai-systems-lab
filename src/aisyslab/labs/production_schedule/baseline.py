"""Deterministic solver: the planning rule from the spreadsheet, without the notes.

Earliest due date first, each order on the press where it ends earliest, using the same library
the agent's code gets. It never reads the planner's notes, so it keeps a job on M3 past 18:00,
starts a job before the one sharing its mould, or plans a rush order as an ordinary one.
"""

from __future__ import annotations

from ...core.lab import RunContext
from .env import ProductionScheduleEnv
from .plantlib import list_schedule


def solve(env: ProductionScheduleEnv, ctx: RunContext) -> None:
    data = ctx.call(env.get_plan)
    sequence = [o["id"] for o in sorted(data["orders"], key=lambda o: (o["due"], o["id"]))]
    ctx.submit(env.submit, list_schedule(data, sequence))
