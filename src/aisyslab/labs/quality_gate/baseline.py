"""Deterministic solver: a decision tree on the process measurements, fitted in code.

Code that knows the plant knows which columns are measurements; batch numbers, lines, shifts
and rework tickets are not. The tree finds the lab's thresholds in the history and decides
tonight's batches. A batch with a missing measurement is held.
"""

from __future__ import annotations

from ...core.lab import RunContext
from .env import QualityGateEnv
from .tree import fit, predict

MEASUREMENTS = ("torque_nm", "cure_temp_c", "cure_time_s", "humidity_pct")


def solve(env: QualityGateEnv, ctx: RunContext) -> None:
    history = ctx.call(env.rows, "history", 0, 500)["rows"]
    tonight = ctx.call(env.rows, "tonight", 0, 500)["rows"]
    tree = fit(history, [b["lab_verdict"] for b in history], MEASUREMENTS, max_depth=3)
    passing = [
        b["batch_id"] for b in tonight if all(b[m] is not None for m in MEASUREMENTS) and predict(tree, b) == "OK"
    ]
    ctx.submit(env.submit, passing)
