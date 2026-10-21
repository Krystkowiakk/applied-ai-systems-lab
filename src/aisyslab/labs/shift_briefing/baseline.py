"""Deterministic solver: three investigations as a graph, run in parallel, then one report.

When the plan is known in advance, orchestration is a dependency graph, not an agent: the
three analyses share nothing, so they run at once, and the briefing waits for all three.
"""

from __future__ import annotations

from typing import Any

from ...core.lab import RunContext
from ...orchestration.dag import Step, run_dag
from ..line_incident.baseline import incident_events
from ..plant_database.baseline import BLOCKED_ORDERS
from ..quality_gate.baseline import MEASUREMENTS
from ..quality_gate.tree import fit, predict
from .env import ShiftBriefingEnv


def solve(env: ShiftBriefingEnv, ctx: RunContext) -> None:
    def root_cause(_: dict[str, Any]) -> str:
        return incident_events(ctx.call(env.incident.download_log))[0][2]

    def blocked(_: dict[str, Any]) -> list[str]:
        return [wo for (wo,) in ctx.call(env.parts.query, BLOCKED_ORDERS)["rows"]]

    def held(_: dict[str, Any]) -> list[str]:
        history = ctx.call(env.quality.rows, "history", 0, 500)["rows"]
        tonight = ctx.call(env.quality.rows, "tonight", 0, 500)["rows"]
        tree = fit(history, [b["lab_verdict"] for b in history], MEASUREMENTS, max_depth=3)
        return [b["batch_id"] for b in tonight if any(b[m] is None for m in MEASUREMENTS) or predict(tree, b) != "OK"]

    def briefing(results: dict[str, Any]) -> None:
        ctx.submit(env.submit_briefing, results["root_cause"], results["blocked"], results["held"])

    run_dag(
        [
            Step("root_cause", root_cause),
            Step("blocked", blocked),
            Step("held", held),
            Step("briefing", briefing, needs=("root_cause", "blocked", "held")),
        ],
        tracer=ctx.tracer,
    )
