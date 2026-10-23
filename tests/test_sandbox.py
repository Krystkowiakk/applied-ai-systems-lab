"""The sandbox for model-written code, and the production-schedule lab that plans with it."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import pytest

from aisyslab.core.lab import EnvOptions, RunContext
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.runner import run_once
from aisyslab.core.trace import Tracer
from aisyslab.labs import LABS
from aisyslab.labs.production_schedule import baseline
from aisyslab.labs.production_schedule.env import ProductionScheduleEnv
from aisyslab.labs.production_schedule.plan import build, reference
from aisyslab.labs.production_schedule.plantlib import evaluate, list_schedule
from aisyslab.sandbox import run_python


def test_the_sandbox_runs_code_with_its_files_and_bounded_output() -> None:
    result = run_python("import lib\nprint(lib.X * 2)\nprint('x' * 50000)", files={"lib.py": "X = 21"}, max_output=200)
    assert result.ok and result.stdout.startswith("42") and "characters cut" in result.stdout


def test_the_sandbox_has_no_keys_no_network_no_subprocesses_and_a_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-for-the-sandbox")
    keys = run_python("import os\nprint(sorted(os.environ))")
    assert keys.ok and "OPENAI_API_KEY" not in keys.stdout
    for code in (
        "import socket\nsocket.create_connection(('example.com', 80))",
        "import subprocess\nsubprocess.run(['ls'])",
        "import os\nos.system('ls')",
    ):
        blocked = run_python(code)
        assert not blocked.ok and "disabled in this sandbox" in blocked.stderr
    assert "sk-test" in os.environ["OPENAI_API_KEY"]  # the parent keeps its environment
    slow = run_python("while True:\n    pass", timeout_s=1)
    assert slow.timed_out and not slow.ok
    with pytest.raises(ValueError, match="working directory"):
        run_python("pass", files={"../escape.py": ""})


def test_the_plan_is_seeded_and_the_reference_keeps_every_note() -> None:
    assert build(4) == build(4) and build(4)[1] != build(5)[1]
    for seed in range(8):
        data, notes, rules = build(seed)
        plan, lateness = reference(data, rules)
        result = evaluate(data, plan, rules)
        assert result["feasible"] and result["lateness_min"] == lateness and len(notes) == 2


def test_the_evaluator_names_what_is_wrong() -> None:
    data, _notes, _rules = build(0)
    rules = {"after": {"O-02": "O-01"}, "cutoff": {"M3": 60}, "same_machine": [["O-03", "O-04"]], "due": {}}
    plan = [
        {"order": "O-01", "machine": "M2", "start": 250},  # inside the M2 maintenance window
        {"order": "O-02", "machine": "M2", "start": 255},
    ]
    problems = " | ".join(evaluate(data, plan, rules)["problems"])
    for expected in ("maintenance on M2", "overlap on M2", "O-02 starts before O-01 has finished", "not scheduled"):
        assert expected in problems


@pytest.mark.parametrize("seed", range(5))
def test_the_spreadsheet_rule_ignores_the_notes(seed: int) -> None:
    env = ProductionScheduleEnv(seed, EnvOptions())
    baseline.solve(env, RunContext(tracer=Tracer()))
    verdict = env.submissions[-1]
    assert verdict.code in ("OK", "LATE", "INFEASIBLE")
    assert sum(ProductionScheduleEnv(s, EnvOptions()) is not None for s in range(1)) == 1


def test_code_in_the_sandbox_plans_with_the_library_and_the_agent_submits_it() -> None:
    probe = ProductionScheduleEnv(2, EnvOptions())
    plan, _ = reference(probe.data, probe.rules)
    sequence = [p["order"] for p in sorted(plan, key=lambda p: (p["start"], p["machine"]))]
    program = (
        "import plant\n"
        f"plan = plant.schedule({json.dumps(sequence)}, after={json.dumps(probe.rules['after'])}, "
        f"cutoff={json.dumps(probe.rules['cutoff'])})\n"
        "print(plant.check(plan)['feasible'])\n"
        "plant.propose(plan)\n"
    )

    def policy(_system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        last = messages[-1]
        if last.role == "user":
            return tool_turn("get_plan")
        if last.tool_name == "get_plan":
            return tool_turn("run_python", code=program)
        if last.tool_name == "run_python":
            result = json.loads(last.content)
            return tool_turn("submit_schedule", plan=result["proposal"]["plan"])
        return text_turn("Done.")

    direct = list_schedule(probe.data, sequence, after=probe.rules["after"], cutoff=probe.rules["cutoff"])
    report = run_once(LABS["production-schedule"], "agent", 2, model=ScriptedModel(policy))
    expected = evaluate(probe.data, direct, probe.rules)
    assert report.accepted == (expected["feasible"] and expected["lateness_min"] <= probe.target)
    assert report.tool_calls == 3
