"""Maintenance procedures of a fictional plant, generated per seed.

Fictional on purpose: answers can be verified exactly, and nothing here can be mistaken for a
real plant's safety rules. The traps mirror real document-control questions: a superseded
revision with older numbers, an exception that overrides the main rule, a date that needs
computing, a question asked in another language than the documents, and a question the
procedures do not answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class Section:
    id: str
    doc: str
    title: str
    text: str
    edition: str
    status: str  # "current" or "superseded"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    value: str  # normalised expected value: digits, ISO date, yes/no, or "not covered"
    cites: tuple[str, ...]  # any one of these section ids supports the answer
    stale: str | None = None  # the value a superseded source would give


def _draw(r: random.Random) -> dict[str, Any]:
    t16 = r.randrange(190, 235, 5)
    t12 = r.randrange(80, 115, 5)
    tss = r.randrange(55, 76)
    old = {k: v - r.randrange(10, 30, 5) for k, v in (("t16", t16), ("t12", t12), ("tss", tss))}
    weeks, old_weeks = r.choice([4, 6, 8]), r.choice([10, 12])
    recheck = r.choice([24, 48, 72])
    wait_min = r.choice([5, 10, 15])
    permit_hours, fire_watch = r.choice([4, 8, 12]), r.choice([30, 60])
    cal_days = r.choice([90, 180, 365])
    mentor_days = r.choice([5, 10, 15])
    kit_cost, coupling_kit = r.randrange(140, 260, 5), r.randrange(60, 120, 5)
    calibrated = date(2026, r.randint(1, 9), r.randint(1, 28))
    return {
        "t16": t16,
        "t12": t12,
        "tss": tss,
        "old": old,
        "weeks": weeks,
        "old_weeks": old_weeks,
        "recheck": recheck,
        "wait_min": wait_min,
        "permit_hours": permit_hours,
        "fire_watch": fire_watch,
        "cal_days": cal_days,
        "mentor_days": mentor_days,
        "kit_cost": kit_cost,
        "coupling_kit": coupling_kit,
        "calibrated": calibrated,
        "last_day": calibrated + timedelta(days=cal_days),
    }


def facts(seed: int) -> dict[str, Any]:
    """The values the procedures state for a seed, for checks outside the lab (synthetic users)."""
    return _draw(random.Random(seed * 31 + 7))


def build(seed: int) -> tuple[list[Section], list[Question]]:
    r = random.Random(seed * 31 + 7)
    f = _draw(r)
    t16, t12, tss, old = f["t16"], f["t12"], f["tss"], f["old"]
    weeks, old_weeks, recheck, wait_min = f["weeks"], f["old_weeks"], f["recheck"], f["wait_min"]
    permit_hours, fire_watch, cal_days = f["permit_hours"], f["fire_watch"], f["cal_days"]
    mentor_days, kit_cost, coupling_kit, calibrated = (
        f["mentor_days"],
        f["kit_cost"],
        f["coupling_kit"],
        f["calibrated"],
    )

    def manual(edition: str, status: str, v: dict[str, int], w: int) -> list[Section]:
        doc = f"maintenance-{edition}"
        intro = (
            "This revision applies to work orders opened from 1 January 2026."
            if status == "current"
            else "This revision applies to work orders opened before 1 January 2026. It is superseded by the 2026 revision."
        )
        return [
            Section(
                f"{doc}#1",
                doc,
                "Scope",
                f"{intro} Coupling work on pumps is done by a qualified mechanical technician; operators do not open couplings.",
                edition,
                status,
            ),
            Section(
                f"{doc}#2",
                doc,
                "Coupling bolt torque",
                f"Tighten pump coupling bolts to {v['t16']} Nm for M16 bolts and to {v['t12']} Nm for M12 bolts. Stainless bolts in the coolant circuit are tightened to {v['tss']} Nm whatever their size, because stainless threads gall at the standard torque. Re-check the torque after the first {recheck} hours of running.",
                edition,
                status,
            ),
            Section(
                f"{doc}#3",
                doc,
                "Hydraulic hoses",
                f"Hydraulic hoses are inspected every {w} weeks and replaced at the first sign of cracking, bulging or weeping. Log the batch number of every replacement hose on the work order.",
                edition,
                status,
            ),
        ]

    sections = [
        *manual("2026", "current", {"t16": t16, "t12": t12, "tss": tss}, weeks),
        *manual("2025", "superseded", old, old_weeks),
        Section(
            "lockout#1",
            "lockout",
            "Isolation",
            "Before working on a machine, switch off and lock every energy source listed on its isolation card. Each person working on the machine applies their own lock and tag, apprentices included; a supervisor's lock does not replace them.",
            "2026",
            "current",
        ),
        Section(
            "lockout#2",
            "lockout",
            "Stored energy",
            f"Hydraulic presses keep pressure in their accumulators after isolation. Wait {wait_min} minutes after isolating a press and check that the pressure gauge reads zero before opening any line.",
            "2026",
            "current",
        ),
        Section(
            "permits#1",
            "permits",
            "Hot work",
            f"Welding, grinding and cutting outside the workshop need a hot-work permit issued by the area owner. A permit is valid for {permit_hours} hours, and a fire watch stays for {fire_watch} minutes after the work ends.",
            "2026",
            "current",
        ),
        Section(
            "permits#2",
            "permits",
            "Work at height",
            "Work more than 2 m above the floor needs a harness anchored to a certified point. Ladders are for access, not for work that takes longer than 30 minutes.",
            "2026",
            "current",
        ),
        Section(
            "calibration#1",
            "calibration",
            "Torque tools",
            f"A torque wrench's calibration is valid for {cal_days} days after the calibration date shown on the tool's label. From the day after that the wrench may not be used; it goes to the tool crib for recalibration.",
            "2026",
            "current",
        ),
        Section(
            "training#1",
            "training",
            "New technicians",
            f"A new technician works alongside a named mentor for their first {mentor_days} working days and signs off maintenance tasks alone only after that.",
            "2026",
            "current",
        ),
        Section(
            "stores#1",
            "stores",
            "Repair kits",
            f"The central store issues a repair kit for a hydraulic hose (two fittings, ferrules and two metres of hose) for {kit_cost} EUR, charged to the cost centre on the work order. A coupling kit costs {coupling_kit} EUR.",
            "2026",
            "current",
        ),
        Section(
            "ppe#1",
            "ppe",
            "Hearing and eye protection",
            "Hearing protection is mandatory in areas above 85 dB(A), marked with blue signs. Safety glasses are worn everywhere on the shop floor.",
            "2026",
            "current",
        ),
    ]

    pool = [
        Question(
            "torque-m16",
            f"What torque do I use for the M16 coupling bolts on the {r.choice(['press 4', 'press 2', 'extruder 1'])} hydraulic pump?",
            str(t16),
            ("maintenance-2026#2",),
            str(old["t16"]),
        ),
        Question(
            "torque-stainless",
            "I am fitting new stainless M16 coupling bolts on coolant pump 2. What torque applies?",
            str(tss),
            ("maintenance-2026#2",),
            str(old["tss"]),
        ),
        Question(
            "hose-weeks",
            "How many weeks apart are hydraulic hoses inspected?",
            str(weeks),
            ("maintenance-2026#3",),
            str(old_weeks),
        ),
        Question(
            "wrench-last-day",
            f"The torque wrench on line 3 was calibrated on {calibrated.isoformat()}. What is the last day it may be used?",
            (calibrated + timedelta(days=cal_days)).isoformat(),
            ("calibration#1",),
        ),
        Question(
            "apprentice-lock",
            "Does an apprentice put their own lock on a machine during lockout?",
            "yes",
            ("lockout#1",),
        ),
        Question(
            "kit-cost-pl", "Ile kosztuje zestaw naprawczy do przewodu hydraulicznego?", str(kit_cost), ("stores#1",)
        ),
        Question(
            "mentor-days",
            "For how many working days does a new technician work with a mentor?",
            str(mentor_days),
            ("training#1",),
        ),
        Question("permit-hours", "For how many hours is a hot-work permit valid?", str(permit_hours), ("permits#1",)),
        Question("sunday-overtime", "What is the overtime rate for Sunday shifts?", "not covered", ()),
    ]
    fixed = ("torque-m16", "torque-stainless", "wrench-last-day", "kit-cost-pl", "sunday-overtime")
    questions = [q for q in pool if q.id in fixed]
    questions += r.sample([q for q in pool if q not in questions], 2)
    return sections, questions


def normalise(value: str) -> str:
    """Digits for amounts and counts, ISO for dates, yes/no, or 'not covered'."""
    v = value.strip().lower()
    if v in ("yes", "no", "not covered"):
        return v
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return v
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or v
