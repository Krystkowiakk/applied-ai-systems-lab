"""The 06:00 operations briefing: three investigations, one report.

The plant manager wants three answers before the morning meeting: where tonight's line-3
stoppage started, which corrective work orders cannot start for lack of parts, and which of
tonight's batches are held at the quality gate. Each answer lives in its own system with its
own data and its own checks (the line-incident, plant-database and quality-gate labs, built
from the same seed). This lab is about the whole: splitting the work, keeping contexts apart
and putting the results together.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ...core.lab import EnvOptions, LabEnv
from ...core.verdict import Verdict
from ..line_incident.env import LineIncidentEnv
from ..plant_database.env import PlantDatabaseEnv
from ..quality_gate.env import QualityGateEnv


class ShiftBriefingEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.incident = LineIncidentEnv(seed, options)
        self.parts = PlantDatabaseEnv(seed, options)
        self.quality = QualityGateEnv(seed, options)
        self._count_lock = threading.Lock()
        for sub in (self.incident, self.parts, self.quality):
            sub._api = self._counted(sub._api)  # type: ignore[method-assign,assignment]  # every system's calls count here too
        self.root_cause = self.incident.required[0]
        self.blocked = set(self.parts.expected)
        self.held = {b["batch_id"] for b in self.quality.tonight} - self.quality.expected
        self.instruction = (
            "Prepare the 06:00 operations briefing for the plant manager, with three answers. "
            "root_cause: the system where tonight's line-3 stoppage started (the first system in the chain, "
            "not the first critical alarm). blocked_work_orders: the open corrective work orders on line 3 "
            "that cannot start because a required spare part is short. held_batches: tonight's batches that "
            "fail the quality gate or miss a measurement. Each answer has its own data: the line log, the "
            "plant database and the quality lab's history. Submit all three with submit_briefing."
        )

    def _counted(self, api: Callable[[str], None]) -> Callable[[str], None]:
        def counted(action: str) -> None:
            with self._count_lock:
                self.api_calls += 1
            api(action)

        return counted

    def submit_briefing(self, root_cause: str, blocked_work_orders: list[str], held_batches: list[str]) -> Verdict:
        self._api("submit_briefing")
        problems: list[str] = []
        ok = 0
        if root_cause.strip().lower() == self.root_cause:
            ok += 1
        else:
            problems.append("root_cause is not the system where the chain started")
        for label, given, truth in (
            ("blocked_work_orders", blocked_work_orders, self.blocked),
            ("held_batches", held_batches, self.held),
        ):
            answer = {x.strip().upper() for x in given if x.strip()}
            missing, extra = len(truth - answer), len(answer - truth)
            if missing or extra:
                problems.append(f"{label}: {missing} missing, {extra} that should not be there")
            else:
                ok += 1
        accepted = not problems
        return self._record(
            Verdict(
                accepted,
                "OK" if accepted else "REJECTED",
                "Briefing complete and correct." if accepted else "; ".join(problems) + ".",
                {"score": round(ok / 3, 3), "sections_ok": ok},
            )
        )
