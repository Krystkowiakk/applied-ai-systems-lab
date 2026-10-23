"""Deterministic solver: keyword rules over the logs, no model.

It reads every shift once and keeps what its rules recognise: the latest word about the line 3
bypass, numbers next to press 2, part numbers next to pump 2, machines with "temporary" or "for
now" in the entry, the first leak on press 4, and who asked for batch numbers. Rules work on
wording they were written for. They cannot tell a suspicion from a confirmed fault, and a
"no leaks seen" on a routine round looks like a leak to them.
"""

from __future__ import annotations

import re

from ...core.lab import RunContext
from .env import ShiftHandoverEnv
from .logs import TEMPORARY

TEMPORARY_WORDS = re.compile(r"temporary|for now|provisional|tie-wrapped|taped")
UNDONE_WORDS = re.compile(r"removed|\boff\b|\bout\b|fitted|repaired")


def solve(env: ShiftHandoverEnv, ctx: RunContext) -> None:
    facts: dict[str, str] = {}
    temporary: set[str] = set()
    while (shift := ctx.call(env.next_shift)) is not None:
        for line in shift["log"].splitlines()[1:]:
            text = line.split(": ", 1)[1]
            low = text.lower()
            if "s4" in low and "line 3" in low:
                facts["line3-bypass"] = "no" if UNDONE_WORDS.search(low) else "yes"
            if "press 2" in low and (m := re.search(r"(\d{3,4})\s*(cycles|strokes|/h)", low)):
                facts["press2-limit"] = m.group(1)
            if "pump 2" in low and (m := re.search(r"P-\d{4}", text)):
                facts["pump2-part"] = m.group(0)
            if "press 4" in low and re.search(r"leak|weep", low):
                facts.setdefault("press4-leak", shift["shift"])
            if "gearbox" in low:
                facts["c3-gearbox"] = "no" if "fine" in low else "yes"
            if m := re.search(r"(\w+) \(planner\) asks", text):
                facts["batch-request"] = m.group(1)
            for machine in TEMPORARY:
                if machine.split()[-1].lower() in low and machine.split()[0] in low:
                    if TEMPORARY_WORDS.search(low):
                        temporary.add(machine)
                    elif UNDONE_WORDS.search(low):
                        temporary.discard(machine)
    ctx.call(
        env.keep, [{"when": "week", "scope": "plant", "status": "info", "text": f"{k}: {v}"} for k, v in facts.items()]
    )
    ctx.call(env.write_handover, "\n".join(f"{k}: {v}" for k, v in facts.items()))
    ctx.call(env.seal)
    answers = []
    for q in ctx.call(env.list_questions):
        qid = q["question_id"]
        value: str | list[str] = sorted(temporary) if qid == "temporary-fixes" else facts.get(qid, "not recorded")
        answers.append({"question_id": qid, "value": value})
    ctx.submit(env.submit, answers)
