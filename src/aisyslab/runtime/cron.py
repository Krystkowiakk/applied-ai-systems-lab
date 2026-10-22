"""Time triggers: a small cron parser, and fan-out grouped in time instead of linked by dependencies.

A fan-out starts several agents together and, at a later fixed time, starts an assembler with
whatever they produced by then. There are no edges between the jobs: a slow or failed collector
cannot block the others or the assembler, it is reported as missing. The price is the slack
between the two times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

FIELDS = (("minute", 0, 59), ("hour", 0, 23), ("day", 1, 31), ("month", 1, 12), ("weekday", 0, 7))


def _field(text: str, name: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in text.split(","):
        base, _, step_text = part.partition("/")
        try:
            step = int(step_text) if step_text else 1
            if base == "*":
                start, end = low, high
            elif "-" in base:
                first, last = base.split("-", 1)
                start, end = int(first), int(last)
            else:
                start = int(base)
                end = high if step_text else start
        except ValueError:
            raise ValueError(f"Cron {name} field '{text}' is not a number, range, list or step.") from None
        if step < 1 or not low <= start <= end <= high:
            raise ValueError(f"Cron {name} field '{text}' is outside {low}-{high}.")
        values.update(range(start, end + 1, step))
    return values


class Cron:
    """Five fields: minute hour day month weekday (0 or 7 is Sunday). Supports *, lists, ranges, steps."""

    def __init__(self, expression: str) -> None:
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError("A cron expression has five fields: minute hour day month weekday.")
        self.expression = expression
        minutes, hours, days, months, weekdays = (
            _field(part, name, low, high) for part, (name, low, high) in zip(parts, FIELDS, strict=True)
        )
        self.minutes, self.hours, self.days, self.months = minutes, hours, days, months
        self.weekdays = {0 if d == 7 else d for d in weekdays}
        self._any_day, self._any_weekday = parts[2] == "*", parts[4] == "*"

    def _day_matches(self, t: datetime) -> bool:
        in_month = t.day in self.days
        in_week = t.isoweekday() % 7 in self.weekdays
        if self._any_day and self._any_weekday:
            return True
        if self._any_day:
            return in_week
        if self._any_weekday:
            return in_month
        return in_month or in_week  # both restricted: either one matches, as in classic cron

    def next_after(self, t: datetime) -> datetime:
        t = t.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = t + timedelta(days=366 * 5)
        while t < limit:
            if t.month not in self.months:
                t = (t.replace(day=1, hour=0, minute=0) + timedelta(days=32)).replace(day=1)
            elif not self._day_matches(t):
                t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            elif t.hour not in self.hours:
                t = (t + timedelta(hours=1)).replace(minute=0)
            elif t.minute not in self.minutes:
                t += timedelta(minutes=1)
            else:
                return t
        raise ValueError(f"'{self.expression}' never fires.")

    def next_epoch(self, after: float, tz: str = "UTC") -> float:
        """The next firing time after ``after`` (epoch seconds), on the wall clock of ``tz``."""
        return self.next_after(datetime.fromtimestamp(after, ZoneInfo(tz))).timestamp()


@dataclass(frozen=True)
class FanOut:
    """Collectors start together at ``collect_at``; the assembler starts at ``assemble_at`` with
    the reports that are finished by then. ``process`` names the folder of process files."""

    name: str
    collectors: tuple[str, ...]
    assembler: str
    collect_at: str
    assemble_at: str
    process: str = ""
    seed: int = 0
    tz: str = "UTC"

    def __post_init__(self) -> None:
        Cron(self.collect_at)
        Cron(self.assemble_at)
        ZoneInfo(self.tz)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "collectors": list(self.collectors),
            "assembler": self.assembler,
            "collect_at": self.collect_at,
            "assemble_at": self.assemble_at,
            "process": self.process,
            "seed": self.seed,
            "tz": self.tz,
        }


MORNING_BRIEFING = FanOut(
    name="morning-briefing",
    collectors=("line-incident", "plant-database", "quality-gate"),
    assembler="briefing-writer",
    collect_at="0 4 * * 1-5",  # an hour of slack before the briefing is assembled
    assemble_at="0 5 * * 1-5",
    process="morning-briefing",
)
