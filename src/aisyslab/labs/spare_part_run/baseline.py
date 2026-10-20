"""Deterministic solver: discover the documents, parse them, search the state space exactly.

Planning over a fully specified map is a search problem, so code solves it optimally on the
first try. That is the point of this lab's comparison.
"""

from __future__ import annotations

import json

from ...core.lab import RunContext
from .env import SparePartRunEnv, solve_grid


def solve(env: SparePartRunEnv, ctx: RunContext) -> None:
    names = {hit.get("name") for query in ("site map", "vehicle rules") for hit in ctx.call(env.search_docs, query)}
    assert {"site_map", "vehicle_rules"} <= names, names
    grid, start, goal = parse_map(ctx.call(env.read_doc, "site_map"))
    budget = json.loads(ctx.call(env.read_doc, "vehicle_rules"))["budget"]
    plan = solve_grid(grid, start, goal, battery_limit=budget["battery"], minutes_limit=budget["minutes"])
    if plan is None:
        raise RuntimeError("no feasible plan within the budget")
    ctx.submit(env.submit, plan[2])


def parse_map(text: str) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    grid: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("Legend"):
            break
        tokens = line.split()
        if tokens and tokens[0].isdigit() and len(tokens) > 2 and not tokens[1].isdigit():
            grid.append(tokens[1:])
    cells = {(r, c): v for r, row in enumerate(grid) for c, v in enumerate(row)}
    start = next(pos for pos, v in cells.items() if v == "S")
    goal = next(pos for pos, v in cells.items() if v == "G")
    return grid, start, goal
