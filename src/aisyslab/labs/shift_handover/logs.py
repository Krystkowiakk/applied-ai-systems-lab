"""A week of shift logs for a maintenance team, generated per seed.

Each log is what technicians wrote during one shift: mostly routine, with a few entries that
matter later. The week hides the situations a handover memory has to get right: a bypass that
a later shift removed, a limit that changed, temporary fixes of which one was made permanent, a
suspicion nobody confirmed, the same sensor name on two lines, a part on order next to another
part number, and a question the logs do not answer at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
SHIFTS = ("morning", "afternoon", "night")
LABELS = tuple(f"{day} {shift}" for day in DAYS for shift in SHIFTS)
START_HOUR = {"morning": 6, "afternoon": 14, "night": 22}
PEOPLE = ("Marek", "Jonas", "Elif", "Tomasz", "Piotr", "Ines", "Karol", "Lena")
PLANNERS = ("Anna", "Ruth", "Sanne")
ASSETS = ("press 2", "press 4", "extruder 1", "coolant pump 2", "conveyor C3", "air compressor C2", "the robot cell")

ROUTINE = (
    "Checked the oil level on {asset}, ok.",
    "Cleaned the suction filter on {asset}.",
    "Changeover on line {line} done in {n} min.",
    "Nothing to report on line {line}.",
    "Replaced a blown fuse in the {asset} control panel.",
    "Topped up coolant on line {line}.",
    "Swept around {asset}, no leaks seen.",
    "Safety walk done, all guards in place.",
    "Calibrated the checkweigher at the end of line {line}.",
    "Helped production clear a jam on line {line}, {n} min.",
    "Main air header steady at {bar} bar.",
    "Tightened a loose cable gland on {asset}.",
    "Filled in the weekly lubrication sheet for line {line}.",
    "Greased the chains on conveyor C{c}.",
    "Walked the line {line} PM round, all ticked.",
)

TEMPORARY = {
    "press 4": (
        "Press 4 hydraulic hose weeping at the fitting. Clamp fitted as a temporary fix, new hose ordered.",
        "New hose fitted on press 4, clamp removed.",
    ),
    "conveyor C3": (
        "Conveyor C3 guard bracket cracked, tie-wrapped for now until the welder is free.",
        "Welder repaired the C3 guard bracket, tie-wraps off.",
    ),
    "extruder 1": (
        "Extruder 1 zone 3 thermocouple failed; running on the spare taped to the barrel, provisional.",
        "Proper thermocouple fitted in extruder 1 zone 3, the taped spare is out.",
    ),
}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: str  # yesno, number, tag, tags, shift, name, or none (not recorded)
    value: str | tuple[str, ...]
    stale: str | None = None  # what was true before a later shift changed it


@dataclass
class Week:
    logs: list[str]
    questions: list[Question]
    facts: dict[str, object] = field(default_factory=dict)


def build(seed: int) -> Week:
    r = random.Random(seed * 97 + 13)
    entries: list[list[str]] = [[] for _ in LABELS]

    def at(shift: int, text: str) -> None:
        entries[shift].append(text)

    def later(after: int, gap: int = 2, last: int = len(LABELS) - 2) -> int:
        return r.randint(min(after + gap, last), last)

    # A bypass on line 3 that a later shift removes, and a bypass on line 4 that stays.
    bypass_on = r.randint(0, 6)
    bypass_off = later(bypass_on)
    at(
        bypass_on,
        r.choice(
            (
                "Line 3 S4 sensor flickering, bypassed it with a jumper so the line keeps running. New S4 ordered.",
                "S4 on line 3 keeps dropping out. Jumpered it for now; replacement sensor on order.",
            )
        ),
    )
    at(
        bypass_off,
        r.choice(
            (
                "Fitted the new S4 on line 3 and took the jumper out.",
                "Line 3: S4 replaced, bypass removed, running on the new sensor.",
                "S4 swapped on line 3, jumper out, tested ok.",
            )
        ),
    )
    at(
        later(bypass_off, 1, len(LABELS) - 1),
        "Line 4 S4 alignment drifted, bypassed it for the rest of the shift. Check it.",
    )

    # A bearing on order for coolant pump 2, next to a part that was fitted elsewhere.
    part = f"P-{r.randint(1000, 9999)}"
    other = f"P-{r.randint(1000, 9999)}"
    bearing = r.randint(0, 7)
    at(
        bearing,
        r.choice(
            (
                f"Coolant pump 2 bearing noisy at the drive end. Greased it; new bearing {part} ordered.",
                f"Pump 2 (coolant) rumbling at the drive end, bearing {part} requested from stores.",
            )
        ),
    )
    at(later(bearing), f"Coolant pump 2 still noisy, bearing {part} not delivered yet.")
    at(r.randint(0, len(LABELS) - 1), f"Seal kit {other} fitted on press 2, no leaks after the test run.")

    # A limit on press 2 that changes once.
    low = r.randrange(700, 900, 50)
    high = low + r.randrange(150, 350, 50)
    limited = r.randint(0, 6)
    raised = later(limited)
    at(
        limited,
        r.choice(
            (
                f"Press 2 limited to {low} cycles/h until the new die is run in.",
                f"New die on press 2: running at {low} strokes per hour max for now.",
            )
        ),
    )
    at(
        raised,
        r.choice(
            (
                f"Die run in on press 2, limit raised to {high} cycles/h.",
                f"Press 2 back up to {high}/h after the die check.",
            )
        ),
    )

    # Three temporary fixes; one is made permanent later in the week.
    fixed_at: dict[str, int] = {}
    for machine, (temporary, _) in TEMPORARY.items():
        fixed_at[machine] = r.randint(0, 8)
        at(fixed_at[machine], temporary)
    made_permanent = r.choice(sorted(TEMPORARY))
    at(later(fixed_at[made_permanent]), TEMPORARY[made_permanent][1])
    still_temporary = tuple(sorted(m for m in TEMPORARY if m != made_permanent))

    # A suspicion nobody confirmed; half the time a later shift rules it out.
    suspected = r.randint(0, 9)
    at(
        suspected,
        r.choice(
            (
                "Operator on line 3 thinks the C3 gearbox is whining. Not checked yet.",
                "Someone mentioned a noise from the conveyor C3 gearbox, didn't have time to look.",
            )
        ),
    )
    if r.random() < 0.5:
        at(later(suspected), "Checked conveyor C3: the gearbox is fine, the noise was a dry return roller, greased it.")

    # A standing request from a planner.
    planner = r.choice(PLANNERS)
    at(r.randint(0, 10), f"{planner} (planner) asks that every hydraulic job is logged with the hose batch number.")

    logs = []
    for index, label in enumerate(LABELS):
        lines = entries[index] + [_routine(r) for _ in range(r.randint(14, 20))]
        r.shuffle(lines)
        hour = START_HOUR[label.split()[1]]
        stamped = []
        minute = r.randint(5, 30)
        for text in lines:
            stamped.append(f"{(hour + minute // 60) % 24:02d}:{minute % 60:02d} {r.choice(PEOPLE)}: {text}")
            minute += r.randint(10, 28)
        logs.append(f"Shift log, {label}\n" + "\n".join(stamped))

    pool = [
        Question("line3-bypass", "Is the S4 sensor on line 3 still bypassed?", "yesno", "no", stale="yes"),
        Question(
            "press2-limit",
            "What is the cycle limit on press 2 now, in cycles per hour?",
            "number",
            str(high),
            stale=str(low),
        ),
        Question("temporary-fixes", "Which machines still have a temporary fix in place?", "tags", still_temporary),
        Question("c3-gearbox", "Is there a confirmed gearbox fault on conveyor C3?", "yesno", "no"),
        Question("scrap-rate", "What was the scrap rate on line 3 on Wednesday?", "none", "not recorded"),
        Question("pump2-part", "Which part number is coolant pump 2 waiting for?", "tag", part),
        Question(
            "press4-leak", "Which shift first reported the hose leak on press 4?", "shift", LABELS[fixed_at["press 4"]]
        ),
        Question("batch-request", "Who asked for hose batch numbers to be logged on hydraulic jobs?", "name", planner),
    ]
    fixed = ("line3-bypass", "press2-limit", "temporary-fixes", "c3-gearbox", "scrap-rate")
    questions = [q for q in pool if q.id in fixed]
    questions += r.sample([q for q in pool if q not in questions], 2)
    facts: dict[str, object] = {"bypass_off": LABELS[bypass_off], "made_permanent": made_permanent, "other_part": other}
    return Week(logs, questions, facts)


def _routine(r: random.Random) -> str:
    asset = r.choice(ASSETS)
    return r.choice(ROUTINE).format(
        asset=asset,
        line=r.choice((1, 2, 3, 4)),
        n=r.randint(8, 45),
        bar=round(r.uniform(6.2, 6.9), 1),
        c=r.choice((1, 2, 3)),
    )
