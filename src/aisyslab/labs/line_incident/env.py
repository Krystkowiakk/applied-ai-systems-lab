"""A production line stopped overnight. One day of logs, one causal chain, lots of noise.

The solver must hand the on-call engineer a short timeline that explains the stoppage. The
reviewer checks four things: length (a token budget), format, that every line points to an
event that really is in the log, and that every system in the causal chain is covered.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tokens import estimate_tokens
from ...core.tools import ToolError
from ...core.verdict import Verdict

DAY = date(2026, 3, 14)
TOKEN_LIMIT = 500
LINE_LIMIT = 25
LEVELS = ("INFO", "WARN", "ERROR", "CRIT")
RANK = {level: i for i, level in enumerate(LEVELS)}
LINE_FORMAT = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}) \[(INFO|WARN|ERROR|CRIT)\] ([a-z0-9-]+): (.+)$")

# Causal chains: (component, level, message template, repeats, minutes between repeats).
CHAINS: list[list[tuple[str, str, str, int, int]]] = [
    [
        ("cooling-pump-2", "WARN", "Vibration {a:.1f} mm/s above alarm limit 7.1 mm/s", 8, 2),
        ("cooling-pump-2", "WARN", "Bearing temperature {b} C and rising", 5, 3),
        ("cooling-pump-2", "ERROR", "Outlet flow {c} m3/h below minimum 42 m3/h", 4, 2),
        ("cooling-loop", "ERROR", "Loop pressure {d:.1f} bar below setpoint 3.2 bar", 5, 2),
        ("furnace-3", "CRIT", "Zone 2 temperature {e} C above limit 940 C", 2, 1),
        ("furnace-3", "CRIT", "Overtemperature interlock tripped, burners off", 1, 1),
        ("conveyor-l3", "ERROR", "Line halted by upstream interlock (furnace-3)", 6, 5),
        ("mes-connector", "WARN", "Order booking backlog {f} messages, retrying", 6, 5),
    ],
    [
        ("compressor-a", "WARN", "Discharge pressure {p:.1f} bar decaying, setpoint 7.5 bar", 7, 2),
        ("compressor-a", "ERROR", "Dryer outlet dew point {w} C out of range", 3, 3),
        ("press-1", "ERROR", "Pneumatic clamp pressure low on tool {c}", 4, 2),
        ("press-1", "CRIT", "Safety stop: clamp not confirmed before stroke", 1, 1),
        ("plc-l3", "ERROR", "Fault {g}: upstream station press-1 not ready", 3, 2),
        ("conveyor-l3", "ERROR", "Line halted, no parts from press-1", 6, 5),
        ("mes-connector", "WARN", "Order booking backlog {f} messages, retrying", 6, 5),
    ],
    [
        ("scada-gw", "WARN", "Packet loss {c}% on field network segment B", 6, 2),
        ("scada-gw", "ERROR", "Lost connection to PLC rack 2, reconnecting", 4, 2),
        ("plc-l3", "CRIT", "Heartbeat from supervisory system lost for {h} s", 2, 1),
        ("plc-l3", "CRIT", "Line placed in safe stop by watchdog", 1, 1),
        ("conveyor-l3", "ERROR", "Line halted by PLC safe stop", 6, 5),
        ("historian-db", "WARN", "Tag gaps detected for {f} line-3 tags", 4, 6),
        ("mes-connector", "WARN", "Order booking backlog {f} messages, retrying", 6, 5),
    ],
]

ALL_COMPONENTS = (
    "cooling-pump-2",
    "cooling-loop",
    "furnace-3",
    "conveyor-l3",
    "mes-connector",
    "compressor-a",
    "press-1",
    "plc-l3",
    "scada-gw",
    "historian-db",
    "label-printer-l3",
    "badge-reader-2",
    "hvac-hall-b",
    "camera-qc-l3",
)
HEARTBEATS = {
    "cooling-pump-2": "Running, flow nominal",
    "cooling-loop": "Pressure nominal",
    "furnace-3": "All zones in band",
    "conveyor-l3": "Running at set speed",
    "mes-connector": "Bookings in sync",
    "compressor-a": "Pressure nominal",
    "press-1": "Cycle OK",
    "plc-l3": "Scan cycle OK",
    "scada-gw": "All PLC links up",
    "historian-db": "Write latency nominal",
    "label-printer-l3": "Printed batch labels",
    "badge-reader-2": "Door event logged",
    "hvac-hall-b": "Hall temperature in band",
    "camera-qc-l3": "Inspection images uploaded",
}


@dataclass(frozen=True)
class Event:
    at: datetime
    level: str
    component: str
    message: str

    def line(self) -> str:
        return f"[{self.at:%Y-%m-%d %H:%M:%S}] [{self.level}] [{self.component}] {self.message}"


class LineIncidentEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        r = self.rng
        chain = CHAINS[seed % len(CHAINS)]
        self.required = tuple(dict.fromkeys(c for c, *_ in chain))
        start = datetime(DAY.year, DAY.month, DAY.day, r.randint(1, 20), r.randint(0, 59))
        self.incident_start = start
        events: list[Event] = []

        # The causal chain: escalating repeats, each step a few minutes after the previous one.
        t = start
        for component, level, template, repeats, every in chain:
            for k in range(repeats):
                values = {
                    "a": 7.2 + 0.4 * k + r.random() * 0.2,
                    "b": 71 + 3 * k + r.randint(0, 2),
                    "c": r.randint(18, 39),
                    "d": 3.0 - 0.15 * k - r.random() * 0.05,
                    "e": 945 + 7 * k + r.randint(0, 5),
                    "f": 40 * (k + 1) + r.randint(0, 30),
                    "g": r.choice([1302, 1417, 2210]),
                    "h": 10 + 3 * k,
                    "p": 7.2 - 0.3 * k - r.random() * 0.1,
                    "w": r.randint(8, 14),
                }
                events.append(
                    Event(
                        t + timedelta(minutes=k * every, seconds=r.randint(0, 59)),
                        level,
                        component,
                        template.format(**values),
                    )
                )
            t += timedelta(minutes=r.randint(2, 5))
        self.first_crit = min(e.at for e in events if e.level == "CRIT")
        last_chain = max(e.at for e in events)

        # Recovery after the incident.
        recovery = last_chain + timedelta(minutes=r.randint(10, 25))
        for component in self.required:
            events.append(
                Event(
                    recovery + timedelta(minutes=r.randint(0, 8)), "INFO", component, "Recovered after operator reset"
                )
            )

        # Heartbeats all day long for every system.
        day_start = datetime(DAY.year, DAY.month, DAY.day)
        for component, message in HEARTBEATS.items():
            minute = r.randint(0, 9)
            while minute < 24 * 60:
                events.append(
                    Event(day_start + timedelta(minutes=minute, seconds=r.randint(0, 59)), "INFO", component, message)
                )
                minute += r.randint(6, 14)

        # Chronic noise: warnings that repeat all day and explain nothing.
        for hour in range(24):
            events.append(
                Event(
                    day_start + timedelta(hours=hour, minutes=r.randint(0, 59)),
                    "WARN",
                    "hvac-hall-b",
                    "Filter differential pressure high, change due in 12 days",
                )
            )
            if hour % 2 == 0:
                events.append(
                    Event(
                        day_start + timedelta(hours=hour, minutes=r.randint(0, 59)),
                        "WARN",
                        "label-printer-l3",
                        "Ribbon low (15%), replace at next changeover",
                    )
                )
        for _ in range(30):
            events.append(
                Event(
                    day_start + timedelta(minutes=r.randint(0, 24 * 60 - 1)),
                    "WARN",
                    "badge-reader-2",
                    "Card read retry on door 4",
                )
            )

        # A red herring: a real error, hours earlier, that resolved itself.
        herring = max(day_start, start - timedelta(hours=r.randint(3, 6)))
        for k in range(3):
            events.append(
                Event(
                    herring + timedelta(minutes=2 * k),
                    "ERROR",
                    "camera-qc-l3",
                    "Image upload failed (timeout), retry succeeded",
                )
            )

        self.events = sorted(events, key=lambda e: e.at)
        self._index = {(e.component, e.at.strftime("%H:%M"), e.level) for e in self.events}

    # ----- API ------------------------------------------------------------------------
    def download_log(self) -> str:
        """The whole log as text, for programs. An agent should never receive this."""
        self._api("download_log")
        return "\n".join(e.line() for e in self.events)

    def overview(self) -> dict[str, Any]:
        self._api("overview")
        levels = Counter(e.level for e in self.events)
        serious = Counter(e.component for e in self.events if RANK[e.level] >= RANK["WARN"])
        return {
            "date": DAY.isoformat(),
            "lines": len(self.events),
            "by_level": dict(levels),
            "components": list(ALL_COMPONENTS),
            "warn_or_worse_by_component": dict(serious.most_common()),
            "first_crit_at": self.first_crit.strftime("%H:%M:%S"),
        }

    def search(
        self,
        component: str | None = None,
        min_level: str = "WARN",
        contains: str | None = None,
        start: str | None = None,
        end: str | None = None,
        collapse_repeats: bool = True,
        limit: int = 30,
    ) -> dict[str, Any]:
        self._api("search")
        if min_level not in RANK:
            raise ToolError(f"min_level must be one of {', '.join(LEVELS)}.")
        lo, hi = _clock(start, "00:00"), _clock(end, "23:59")
        hits = [
            e
            for e in self.events
            if RANK[e.level] >= RANK[min_level]
            and (component is None or e.component == component)
            and (contains is None or contains.lower() in e.message.lower())
            and lo <= e.at.strftime("%H:%M") <= hi
        ]
        if collapse_repeats:
            groups: dict[tuple[str, str, str], list[Event]] = {}
            for e in hits:
                groups.setdefault((e.component, e.level, _template(e.message)), []).append(e)
            rows = [
                {
                    "first": g[0].at.strftime("%H:%M:%S"),
                    "last": g[-1].at.strftime("%H:%M:%S"),
                    "count": len(g),
                    "level": g[0].level,
                    "component": g[0].component,
                    "example": g[0].message,
                }
                for g in groups.values()
            ]
        else:
            rows = [
                {"at": e.at.strftime("%H:%M:%S"), "level": e.level, "component": e.component, "message": e.message}
                for e in hits
            ]
        result: dict[str, Any] = {"matches": len(rows), "rows": rows[:limit]}
        if len(rows) > limit:
            result["hint"] = f"{len(rows) - limit} more rows. Narrow by component, time window or level."
        if not rows:
            result["hint"] = "Nothing matched. Lower min_level or widen the time window."
        return result

    def submit(self, timeline: str) -> Verdict:
        self._api("submit")
        lines = [line.strip() for line in timeline.strip().splitlines() if line.strip()]
        problems: list[str] = []
        tokens = estimate_tokens(timeline)
        if not lines:
            return self._record(Verdict(False, "EMPTY", "The timeline is empty.", {"tokens": 0}))
        if tokens > TOKEN_LIMIT:
            problems.append(f"too long: {tokens} tokens (limit {TOKEN_LIMIT})")
        if len(lines) > LINE_LIMIT:
            problems.append(f"too many lines: {len(lines)} (limit {LINE_LIMIT})")
        covered: set[str] = set()
        previous: tuple[str, str] | None = None
        for i, line in enumerate(lines, 1):
            m = LINE_FORMAT.match(line)
            if not m:
                problems.append(f"line {i}: expected 'YYYY-MM-DD HH:MM [LEVEL] component: message'")
                continue
            day, hh, mm, level, component = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            clock = f"{hh}:{mm}"
            if component not in ALL_COMPONENTS:
                problems.append(f"line {i}: unknown component '{component}'")
            elif day != DAY.isoformat() or (component, clock, level) not in self._index:
                problems.append(f"line {i}: the log has no {level} event for {component} at {clock}")
            elif RANK[level] >= RANK["WARN"]:
                covered.add(component)
            if previous is not None and (day, clock) < previous:
                problems.append(f"line {i}: not in chronological order")
            previous = (day, clock)
        missing = [c for c in self.required if c not in covered]
        if missing:
            problems.append("the engineer still cannot explain the stoppage: no evidence for " + ", ".join(missing))
        accepted = not problems
        feedback = "Timeline accepted." if accepted else "; ".join(problems)
        return self._record(
            Verdict(
                accepted,
                "OK" if accepted else "REJECTED",
                feedback,
                {"tokens": tokens, "lines": len(lines), "missing": missing},
            )
        )


def _template(message: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", message)


def _clock(value: str | None, default: str) -> str:
    if value is None:
        return default
    if not re.fullmatch(r"\d{2}:\d{2}", value):
        raise ToolError(f"Time '{value}' must be HH:MM.")
    return value
