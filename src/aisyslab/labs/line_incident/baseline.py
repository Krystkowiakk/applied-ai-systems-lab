"""Deterministic solver: filter, deduplicate, window, compress; expand on reviewer feedback.

Code can read the whole log because it has no context window. It removes chronic noise
(warnings that recur across many hours), keeps the first occurrence of each distinct event
inside the incident window, and adds a system only when the reviewer says it is missing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta

from ...core.lab import RunContext
from .env import RANK, LineIncidentEnv

RAW = re.compile(r"^\[(\S+ \S+)\] \[(\w+)\] \[([a-z0-9-]+)\] (.+)$")
CHRONIC_HOURS = 4
WINDOW_BEFORE = timedelta(minutes=60)
WINDOW_AFTER = timedelta(minutes=45)
MAX_WORDS = 9

Event = tuple[datetime, str, str, str]


def solve(env: LineIncidentEnv, ctx: RunContext, max_attempts: int = 4) -> None:
    events = [_parse(line) for line in ctx.call(env.download_log).splitlines()]
    serious = [e for e in events if RANK[e[1]] >= RANK["WARN"]]
    chronic = _chronic(serious)
    first_crit = min(e[0] for e in serious if e[1] == "CRIT")
    window = [e for e in serious if first_crit - WINDOW_BEFORE <= e[0] <= first_crit + WINDOW_AFTER]
    selected = _first_of_each([e for e in window if (e[2], _template(e[3])) not in chronic])

    for _ in range(max_attempts):
        verdict = ctx.submit(env.submit, _render(selected))
        if verdict.accepted:
            return
        for component in verdict.details.get("missing", []):
            candidates = [e for e in serious if e[2] == component]
            if candidates:
                selected.append(min(candidates))
        selected = sorted(set(selected))


def _parse(line: str) -> Event:
    m = RAW.match(line)
    if m is None:
        raise ValueError(f"unparseable log line: {line!r}")
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2), m.group(3), m.group(4)


def _template(message: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", message)


def _chronic(events: list[Event]) -> set[tuple[str, str]]:
    hours: dict[tuple[str, str], set[int]] = defaultdict(set)
    for at, _level, component, message in events:
        hours[(component, _template(message))].add(at.hour)
    return {key for key, seen in hours.items() if len(seen) >= CHRONIC_HOURS}


def _first_of_each(events: list[Event]) -> list[Event]:
    first: dict[tuple[str, str, str], Event] = {}
    for e in sorted(events):
        first.setdefault((e[2], e[1], _template(e[3])), e)
    return sorted(first.values())


def _render(events: list[Event]) -> str:
    lines = []
    for at, level, component, message in sorted(events):
        short = " ".join(message.split()[:MAX_WORDS])
        lines.append(f"{at:%Y-%m-%d %H:%M} [{level}] {component}: {short}")
    return "\n".join(lines)
