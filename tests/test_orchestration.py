"""Several agents on one task: workspace, dependency graphs, delegation and the briefing lab."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence

import pytest

from aisyslab.core.lab import EnvOptions
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.runner import run_once
from aisyslab.core.tools import ToolError
from aisyslab.core.trace import Tracer
from aisyslab.labs import LABS
from aisyslab.labs.line_incident.env import LineIncidentEnv
from aisyslab.labs.plant_database.baseline import BLOCKED_ORDERS
from aisyslab.labs.shift_briefing.env import ShiftBriefingEnv
from aisyslab.orchestration.dag import Step, check, run_dag
from aisyslab.orchestration.workspace import Workspace


def test_workspace_refuses_a_write_based_on_a_stale_version() -> None:
    ws = Workspace()
    assert ws.write("notes/incident.md", "v1", author="a")["version"] == 1
    with pytest.raises(ToolError, match="Conflict"):
        ws.write("workspace://notes/incident.md", "overwrite", author="b", expected_version=0)
    assert ws.write("notes/incident.md", "v2", author="b", expected_version=1)["version"] == 2
    assert ws.read("notes/incident.md")["text"] == "v2" and ws.conflicts == 1
    with pytest.raises(ToolError, match="Invalid"):
        ws.read("../etc/passwd")


def test_dag_runs_independent_steps_in_parallel_and_rejects_cycles() -> None:
    barrier = threading.Barrier(2, timeout=5)  # only passes if both steps run at the same time

    def meet(_: dict) -> str:  # type: ignore[type-arg]
        barrier.wait()
        return threading.current_thread().name

    tracer = Tracer()
    results = run_dag(
        [Step("a", meet), Step("b", meet), Step("join", lambda r: sorted(r), needs=("a", "b"))], tracer=tracer
    )
    assert results["a"] != results["b"] and results["join"] == ["a", "b"]
    assert tracer.count("dag_step") == 6
    with pytest.raises(ValueError, match="Cycle"):
        check([Step("x", lambda r: 1, needs=("y",)), Step("y", lambda r: 1, needs=("x",))])
    with pytest.raises(ValueError, match="unknown"):
        check([Step("x", lambda r: 1, needs=("nope",))])


def test_child_tracer_lands_in_the_parent_trace_with_the_agent_name() -> None:
    parent = Tracer()
    parent.child("parts_planner").event("tool", name="run_query")
    assert parent.events[-1]["agent"] == "parts_planner" and parent.events[-1]["name"] == "run_query"


def test_briefing_feedback_names_the_wrong_sections() -> None:
    env = ShiftBriefingEnv(2, EnvOptions())
    verdict = env.submit_briefing("furnace-3", sorted(env.blocked), [])
    assert not verdict.accepted and verdict.details["sections_ok"] == 1
    assert "root_cause" in verdict.feedback and "held_batches" in verdict.feedback
    assert "blocked_work_orders" not in verdict.feedback


def test_orchestrator_delegates_reads_the_workspace_and_submits() -> None:
    seed = 4
    root = LineIncidentEnv(seed, EnvOptions()).required[0]
    roles = ["incident_analyst", "parts_planner", "quality_engineer"]

    def results(messages: Sequence[Message], tool: str) -> list[dict]:  # type: ignore[type-arg]
        return [json.loads(m.content) for m in messages if m.role == "tool" and m.tool_name == tool]

    def policy(_s: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        names = {t.name for t in tools}
        done = [m.tool_name for m in messages if m.role == "tool"]
        if "delegate" in names:  # the orchestrator
            if done.count("delegate") < 3:
                role = roles[done.count("delegate")]
                return tool_turn("delegate", agent=role, task=f"Write your answer to {role}.md as JSON.")
            if done.count("workspace_read") < 3:
                return tool_turn("workspace_read", path=f"{roles[done.count('workspace_read')]}.md")
            if "submit_briefing" not in done:
                notes = [json.loads(r["text"]) for r in results(messages, "workspace_read")]
                return tool_turn(
                    "submit_briefing", root_cause=notes[0], blocked_work_orders=notes[1], held_batches=notes[2]
                )
            return text_turn("Briefing submitted.")
        if "run_query" in names:  # parts planner, with its real tools
            if not done:
                return tool_turn("run_query", sql=BLOCKED_ORDERS)
            if "workspace_write" not in done:
                rows = results(messages, "run_query")[0]["rows"]
                return tool_turn("workspace_write", path="parts_planner.md", text=json.dumps([r[0] for r in rows]))
            return text_turn("Blocked orders written.")
        if "train_tree" in names:  # quality engineer, with its real tools
            if not done:
                return tool_turn("train_tree", features=["torque_nm", "cure_temp_c", "cure_time_s", "humidity_pct"])
            if done == ["train_tree"]:
                return tool_turn("predict_tonight", model_id=results(messages, "train_tree")[0]["model_id"])
            if "workspace_write" not in done:
                held = [
                    p["batch_id"]
                    for p in results(messages, "predict_tonight")[0]["predictions"]
                    if p["prediction"] != "OK"
                ]
                return tool_turn("workspace_write", path="quality_engineer.md", text=json.dumps(held))
            return text_turn("Held batches written.")
        if "workspace_write" not in done:  # incident analyst
            return tool_turn("workspace_write", path="incident_analyst.md", text=json.dumps(root))
        return text_turn("Root cause written.")

    report = run_once(LABS["shift-briefing"], "agent", seed, model=ScriptedModel(policy))
    assert report.accepted and report.status == "goal_reached"
    assert report.model_calls > 10  # the specialists' calls count toward the run
