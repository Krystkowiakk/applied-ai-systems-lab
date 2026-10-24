"""Scheduling core for tomorrow's press plan: a list scheduler and an evaluator.

Standard library only, on purpose: the same file runs inside the sandbox as the ``plant``
library, so code a model writes calls these functions instead of making one tool call per
decision. Times are minutes after 06:00; the horizon is two shifts.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

SHIFT_START = 6 * 60


def hhmm(minute: int) -> str:
    total = SHIFT_START + int(minute)
    return f"{total // 60 % 24:02d}:{total % 60:02d}"


def changeover(data: dict[str, Any], before: str | None, family: str) -> int:
    if before is None:
        return 0
    return int(data["changeover"]["same" if before == family else "different"])


def _after_maintenance(data: dict[str, Any], machine: str, start: int, minutes: int) -> int:
    for begin, end in data.get("maintenance", {}).get(machine, []):
        if start < end and start + minutes > begin:
            start = end
    return start


def list_schedule(
    data: dict[str, Any],
    sequence: list[str],
    *,
    machine_for: dict[str, str] | None = None,
    after: dict[str, str] | None = None,
    cutoff: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Place the orders in ``sequence`` order, each where it ends earliest.

    ``machine_for`` pins an order to a machine, ``after`` makes an order wait for another one to
    finish, ``cutoff`` gives the minute by which a machine's jobs must end. An order that fits
    nowhere is left out; ``evaluate`` reports it. An order whose predecessor comes later in the
    sequence is an error: it cannot wait for something not yet placed. One whose predecessor
    was left out is left out too.
    """
    orders = {o["id"]: o for o in data["orders"]}
    free = dict.fromkeys(data["machines"], 0)
    last: dict[str, str | None] = dict.fromkeys(data["machines"], None)
    ends: dict[str, int] = {}
    plan = []
    for position, oid in enumerate(sequence):
        o = orders[oid]
        first = (after or {}).get(oid)
        if first is not None and first not in ends:
            if first in sequence[position + 1 :]:
                raise ValueError(f"{oid} has to wait for {first}: put {first} earlier in the sequence.")
            continue  # its predecessor was left out, so it cannot start either
        if machine_for and oid in machine_for:
            candidates = [machine_for[oid]]
        else:
            candidates = [m for m, spec in data["machines"].items() if o["family"] in spec["families"]]
        best = None
        for m in candidates:
            start = free[m] + changeover(data, last[m], o["family"])
            if after and oid in after:
                start = max(start, ends.get(after[oid], start))
            start = _after_maintenance(data, m, start, o["minutes"])
            end = start + o["minutes"]
            if end > data["horizon"] or (cutoff and m in cutoff and end > cutoff[m]):
                continue
            if best is None or end < best[2]:
                best = (m, start, end)
        if best is None:
            continue
        m, start, end = best
        free[m], last[m], ends[oid] = end, o["family"], end
        plan.append({"order": oid, "machine": m, "start": start})
    return plan


def evaluate(
    data: dict[str, Any], plan: list[dict[str, Any]], constraints: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Feasibility and lateness of a plan. ``constraints`` adds rules that are not in the data."""
    rules = constraints or {}
    orders = {o["id"]: o for o in data["orders"]}
    due = {oid: o["due"] for oid, o in orders.items()} | rules.get("due", {})
    problems: list[str] = []
    placed: dict[str, dict[str, Any]] = {}
    for p in plan:
        oid, m = str(p.get("order")), str(p.get("machine"))
        if oid not in orders:
            problems.append(f"unknown order {oid}")
            continue
        if oid in placed:
            problems.append(f"{oid} placed twice")
            continue
        if m not in data["machines"] or orders[oid]["family"] not in data["machines"][m]["families"]:
            problems.append(f"{oid} on a machine that cannot run family {orders[oid]['family']}")
            continue
        placed[oid] = {
            "machine": m,
            "start": int(p.get("start", 0)),
            "end": int(p.get("start", 0)) + orders[oid]["minutes"],
        }
    missing = sorted(set(orders) - set(placed))
    if missing:
        problems.append(f"not scheduled: {', '.join(missing)}")
    for m in data["machines"]:
        jobs = sorted((v["start"], v["end"], oid) for oid, v in placed.items() if v["machine"] == m)
        for (_s1, e1, a), (s2, _e2, b) in pairwise(jobs):
            gap = changeover(data, orders[a]["family"], orders[b]["family"])
            if s2 < e1 + gap:
                problems.append(f"{a} and {b} overlap on {m} (with a {gap} min changeover)")
        for begin, end in data.get("maintenance", {}).get(m, []):
            problems += [f"{oid} runs during maintenance on {m}" for s, e, oid in jobs if s < end and e > begin]
        problems += [f"{oid} runs past the end of the day" for s, e, oid in jobs if s < 0 or e > data["horizon"]]
        cut = rules.get("cutoff", {}).get(m)
        if cut is not None:
            problems += [f"{oid} ends after {hhmm(cut)} on {m}" for _s, e, oid in jobs if e > cut]
    for later, first in rules.get("after", {}).items():
        if later in placed and first in placed and placed[later]["start"] < placed[first]["end"]:
            problems.append(f"{later} starts before {first} has finished")
    for a, b in rules.get("same_machine", []):
        if a in placed and b in placed and placed[a]["machine"] != placed[b]["machine"]:
            problems.append(f"{a} and {b} are on different machines")
    lateness = sum(max(0, v["end"] - due[oid]) for oid, v in placed.items())
    late = sorted(oid for oid, v in placed.items() if v["end"] > due[oid])
    return {"feasible": not problems, "problems": problems[:12], "lateness_min": lateness, "late_orders": late}


def sandbox_library(data: dict[str, Any]) -> str:
    """The source of the ``plant`` module the sandbox gets: this file plus the data, bound."""
    import inspect
    import sys

    source = inspect.getsource(sys.modules[__name__])
    return (
        source
        + "\n\nDATA = "
        + json.dumps(data)
        + '''


def data():
    """Tomorrow's orders, machines, changeovers, maintenance windows and horizon."""
    return DATA


def schedule(sequence, **kw):
    """list_schedule on tomorrow's data."""
    return list_schedule(DATA, sequence, **kw)


def check(plan):
    """evaluate on tomorrow's data: the data's rules only; the planner's notes are yours to keep."""
    return evaluate(DATA, plan)


def propose(plan):
    """Print the plan as the run's proposal (the last proposal printed counts)."""
    print("PROPOSAL " + json.dumps(plan))
'''
    )
