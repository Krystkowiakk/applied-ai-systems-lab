"""Tomorrow's press plan, generated per seed: orders, machines, changeovers, and the planner's notes.

The data is structured; the planner's notes are not. Each seed adds two notes in plain words (a
shared mould that fixes the order of two jobs, a machine without an operator after 18:00, a rush
order, two jobs that need the same fixture). The verifier holds them as rules; whoever plans has
to read them.
"""

from __future__ import annotations

import random
from typing import Any

from .plantlib import evaluate, hhmm, list_schedule

MACHINES = {"M1": {"families": ["A", "B"]}, "M2": {"families": ["A", "B", "C"]}, "M3": {"families": ["B", "C"]}}
HORIZON = 16 * 60  # 06:00 to 22:00


def build(seed: int) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    r = random.Random(seed * 71 + 5)
    orders: list[dict[str, Any]] = []
    for n in range(1, 17):
        family = r.choice("ABC")
        minutes = r.randrange(50, 160, 5)
        orders.append({"id": f"O-{n:02d}", "family": family, "minutes": minutes, "due": r.randrange(150, 780, 15)})
    data = {
        "orders": orders,
        "machines": MACHINES,
        "changeover": {"same": 10, "different": 40},
        "maintenance": {"M2": [[240, 360]]},
        "horizon": HORIZON,
    }
    ids = [o["id"] for o in orders]
    families: dict[str, str] = {o["id"]: o["family"] for o in orders}
    notes: list[str] = []
    rules: dict[str, Any] = {"due": {}, "after": {}, "cutoff": {}, "same_machine": []}
    for kind in r.sample(["mould", "operator", "rush", "fixture"], 2):
        if kind == "mould":
            a, b = r.sample(ids, 2)
            notes.append(f"{a} and {b} share a mould: {a} has to be finished before {b} starts.")
            rules["after"][b] = a
        elif kind == "operator":
            notes.append("M3 has no operator after 18:00 tomorrow: nothing may run on M3 past 18:00.")
            rules["cutoff"]["M3"] = 12 * 60
        elif kind == "rush":
            rush = r.choice(ids)
            due = r.randrange(150, 270, 15)
            notes.append(f"{rush} is now a rush order: the customer collects it at {hhmm(due)}.")
            rules["due"][rush] = due
        else:
            pairs = [(a, b) for a in ids for b in ids if a < b and families[a] == families[b] == "B"]
            a, b = r.choice(pairs) if pairs else r.sample(ids, 2)
            notes.append(f"{a} and {b} need the same fixture, so they have to run on the same press.")
            rules["same_machine"].append([a, b])
    return data, notes, rules


def reference(data: dict[str, Any], rules: dict[str, Any], passes: int = 30) -> tuple[list[dict[str, Any]], int]:
    """A planner who reads the notes: due-date order with the notes as constraints, then swaps
    neighbours while the lateness drops. Its lateness sets the bar."""
    due = {o["id"]: o["due"] for o in data["orders"]} | rules["due"]
    after = rules["after"]

    def ordered(sequence: list[str]) -> list[str]:
        for later, first in after.items():  # a job waits for its predecessor: keep the predecessor earlier
            if sequence.index(first) > sequence.index(later):
                sequence.remove(first)
                sequence.insert(sequence.index(later), first)
        return sequence

    pins: list[dict[str, str]] = [{}]
    for a, b in rules["same_machine"]:
        common = [
            m
            for m, spec in data["machines"].items()
            if all(f in spec["families"] for f in (_family(data, a), _family(data, b)))
        ]
        pins = [p | {a: m, b: m} for p in pins for m in common]

    best: tuple[list[dict[str, Any]], int] | None = None
    for pin in pins:
        sequence = ordered(sorted(due, key=lambda oid: (due[oid], oid)))

        def score(seq: list[str], pin: dict[str, str] = pin) -> tuple[list[dict[str, Any]], int]:
            plan = list_schedule(data, seq, machine_for=pin, after=after, cutoff=rules["cutoff"])
            result = evaluate(data, plan, rules)
            return plan, result["lateness_min"] if result["feasible"] else 10**9

        plan, cost = score(sequence)
        for _ in range(passes):
            improved = False
            for i in range(len(sequence) - 1):
                trial = sequence[:]
                trial[i], trial[i + 1] = trial[i + 1], trial[i]
                trial = ordered(trial)
                trial_plan, trial_cost = score(trial)
                if trial_cost < cost:
                    sequence, plan, cost, improved = trial, trial_plan, trial_cost, True
            if not improved:
                break
        if best is None or cost < best[1]:
            best = (plan, cost)
    assert best is not None
    return best


def _family(data: dict[str, Any], oid: str) -> str:
    return next(o["family"] for o in data["orders"] if o["id"] == oid)
