"""Production schedule: tomorrow's orders on three presses, planned by code, judged by code.

The environment gives the data and the planner's notes, runs code in the sandbox with the
``plant`` library and tomorrow's data next to it, and judges a submitted plan: every order once,
on a press that can run it, no overlap (changeovers included), nothing during maintenance or
after the day ends, and the notes kept. A feasible plan is accepted when its total lateness is no
worse than what a planner who reads the notes achieves, with some slack.
"""

from __future__ import annotations

import json
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict
from ...sandbox.runner import run_python
from .plan import build, reference
from .plantlib import evaluate, sandbox_library


class ProductionScheduleEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.data, self.notes, self.rules = build(seed)
        _plan, best = reference(self.data, self.rules)
        self.target = int(best * 1.15) + 30  # what a careful planner reaches, with slack
        self.runs = 0

    def get_plan(self) -> dict[str, Any]:
        self._api("get_plan")
        return {**self.data, "notes": self.notes, "times": "minutes after 06:00"}

    def run_code(self, code: str) -> dict[str, Any]:
        """Run code in the sandbox; a line 'PROPOSAL [...]' it prints is checked on the data's rules."""
        self._api("run_code")
        self.runs += 1
        result = run_python(code, files={"plant.py": sandbox_library(self.data)}, timeout_s=15)
        out: dict[str, Any] = result.as_dict()
        proposals = [line[len("PROPOSAL ") :] for line in result.stdout.splitlines() if line.startswith("PROPOSAL ")]
        if proposals:
            try:
                plan = json.loads(proposals[-1])
            except ValueError:
                out["proposal"] = "the PROPOSAL line is not valid JSON"
            else:
                out["proposal"] = {"plan": plan, "check": evaluate(self.data, plan)}
        return out

    def submit(self, plan: list[dict[str, Any]]) -> Verdict:
        self._api("submit")
        if not isinstance(plan, list):
            raise ToolError("Submit the plan as a list of {order, machine, start}.")
        result = evaluate(self.data, plan, self.rules)
        details = {**result, "target_lateness_min": self.target}
        if not result["feasible"]:
            return self._record(Verdict(False, "INFEASIBLE", "; ".join(result["problems"]), details | {"score": 0.0}))
        lateness = result["lateness_min"]
        if lateness > self.target:
            feedback = f"Feasible, but {lateness} min late in total; a careful plan stays within {self.target} min."
            return self._record(Verdict(False, "LATE", feedback, details | {"score": round(self.target / lateness, 3)}))
        return self._record(Verdict(True, "OK", f"Feasible, {lateness} min late in total.", details | {"score": 1.0}))
