"""Observational memory and the shift-handover lab built on it."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from aisyslab.core.lab import EnvOptions, RunContext
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.runner import run_once
from aisyslab.core.tools import ToolError
from aisyslab.core.trace import Tracer
from aisyslab.labs import LABS
from aisyslab.labs.shift_handover import baseline
from aisyslab.labs.shift_handover.env import HANDOVER_BUDGET, ShiftHandoverEnv
from aisyslab.labs.shift_handover.logs import LABELS, build
from aisyslab.memory import ObservationalMemory, fit, normalise_scope, parse_observations
from aisyslab.memory.observational import in_scope


def test_observer_output_is_validated() -> None:
    text = json.dumps(
        {
            "observations": [
                {"scope": "asset:press 4", "status": "open", "text": "Hose weeping, clamp fitted."},
                {"scope": "Asset: Press_2", "status": "open", "text": "Limited to 800 cycles/h."},
                {"scope": "the plant", "status": "open", "text": "bad scope"},
                {"scope": "plant", "status": "maybe", "text": "bad status"},
                {"scope": "line-3", "status": "resolved", "text": "  "},
            ]
        }
    )
    kept, problems = parse_observations(f"```json\n{text}\n```", "Tue night")
    assert [o.scope for o in kept] == ["asset:press 4", "asset:press 2"]
    assert problems == ["unknown scope 'the plant'", "unknown status 'maybe'", "an empty observation"]
    assert normalise_scope("Line 3") == "line-3" and normalise_scope("person: Anna") == "person:anna"
    assert parse_observations("The pump is fine.", "x") == ([], ["the observer did not return JSON"])


def test_memory_observes_reflects_within_budget_and_recalls_by_scope() -> None:
    calls: list[tuple[str, str]] = []

    def model(system: str, user: str, purpose: str) -> str:
        calls.append((purpose, user))
        if purpose == "observer":
            return json.dumps(
                {
                    "observations": [
                        {"scope": "line-3", "status": "open", "text": "S4 bypassed with a jumper."},
                        {"scope": "line-4", "status": "open", "text": "S4 bypassed for the night."},
                    ]
                }
            )
        too_long = "x " * 400
        return too_long if len([c for c in calls if c[0] == "reflector"]) == 1 else "[Mon] (line-3) open: S4 bypassed."

    memory = ObservationalMemory(model, budget_tokens=50, reflect_above=10)
    memory.observe("Mon morning", "log text")
    assert memory.due() and "S4 bypassed with a jumper" in memory.context()
    record = memory.reflect()
    assert record == "[Mon] (line-3) open: S4 bypassed." and not memory.due()
    assert "budget is 50" in calls[-1][1]  # the second attempt is told how far over it was
    assert [o.scope for o in memory.recall("S4 bypass", scope="line-4")] == ["line-4"]

    restored = ObservationalMemory(model)
    restored.load(memory.to_dict())
    assert restored.record == record and len(restored.journal) == 2

    statuses = ["resolved", "open", "info", "unconfirmed"] * 3
    lines = [f"[Mon] (line-{n}) {status}: item {n} with a few more words" for n, status in enumerate(statuses)]
    stubborn = ObservationalMemory(lambda *_: "\n".join(lines), budget_tokens=120)
    record = stubborn.reflect()  # never fits: code cuts the least important lines
    assert stubborn.cut_lines > 0 and "resolved" not in record and "unconfirmed" in record
    assert fit("short", 50) == ("short", 0)


def test_the_week_is_seeded_and_the_logs_are_read_once_then_sealed() -> None:
    assert build(3).logs == build(3).logs and build(3).logs != build(4).logs
    env = ShiftHandoverEnv(0, EnvOptions())
    with pytest.raises(ToolError, match="sealed"):
        env.list_questions()
    first = env.next_shift()
    assert first is not None and first["shift"] == LABELS[0] and first["shifts_left"] == 14
    with pytest.raises(ToolError, match="14 shifts"):
        env.seal()
    while env.next_shift() is not None:
        pass
    with pytest.raises(ToolError, match=f"budget is {HANDOVER_BUDGET}"):
        env.write_handover("word " * 500)
    env.seal()
    with pytest.raises(ToolError, match="sealed"):
        env.next_shift()
    assert len(env.list_questions()) == 7


def test_the_verifier_names_the_kind_of_mistake() -> None:
    env = ShiftHandoverEnv(0, EnvOptions())
    while env.next_shift() is not None:
        pass
    env.seal()
    fixes = next(q for q in env.questions if q.id == "temporary-fixes")
    answers = [
        {"question_id": "line3-bypass", "value": "yes"},
        {"question_id": "press2-limit", "value": next(q.stale for q in env.questions if q.id == "press2-limit")},
        {"question_id": "temporary-fixes", "value": [*fixes.value, "air compressor C2"]},
        {"question_id": "c3-gearbox", "value": "Yes, whining."},
        {"question_id": "scrap-rate", "value": "2%"},
    ]
    feedback = env.submit(answers).details["per_question"]
    assert feedback["line3-bypass"].startswith("outdated") and feedback["press2-limit"].startswith("outdated")
    assert feedback["temporary-fixes"] == "wrong: 0 machine(s) missing, 1 that should not be there"
    assert "confirmed" in feedback["c3-gearbox"] and "not recorded" in feedback["scrap-rate"]
    right = env.submit(
        [{"question_id": q.id, "value": list(q.value) if q.kind == "tags" else q.value} for q in env.questions]
    )
    assert right.accepted


@pytest.mark.parametrize("seed", range(6))
def test_keyword_rules_get_part_of_the_week(seed: int) -> None:
    env = ShiftHandoverEnv(seed, EnvOptions())
    baseline.solve(env, RunContext(tracer=Tracer()))
    assert 0.5 <= env.submissions[-1].details["score"] <= 1.0


def test_the_agent_builds_memory_then_answers_from_it() -> None:
    probe = ShiftHandoverEnv(2, EnvOptions())
    answers = [{"question_id": q.id, "value": list(q.value) if q.kind == "tags" else q.value} for q in probe.questions]
    seen_logs: list[bool] = []

    def policy(system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        if system.startswith("You keep the handover memory"):
            return text_turn(json.dumps({"observations": [{"scope": "plant", "status": "info", "text": "A shift."}]}))
        if system.startswith("You condense"):
            return text_turn("[week] (plant) info: shifts observed.")
        seen_logs.append(any("Shift log," in m.content for m in messages))
        last = messages[-1]
        if last.role == "user" or (last.tool_name == "observe_next_shift" and '"done"' not in last.content):
            return tool_turn("observe_next_shift")
        step = {"observe_next_shift": "reflect", "reflect": "seal_logs", "seal_logs": "list_questions"}
        if last.tool_name in step:
            return tool_turn(step[last.tool_name])
        if last.tool_name == "list_questions":
            return tool_turn("submit_answers", answers=answers)
        return text_turn("Done.")

    report = run_once(LABS["shift-handover"], "agent", 2, model=ScriptedModel(policy))
    assert report.accepted and not any(seen_logs)  # the agent itself never saw a log
    assert report.model_calls > 15  # one observer call per shift, the reflector, the agent's turns


def test_recall_treats_scope_as_a_soft_filter_and_repeats_are_kept_once() -> None:
    env = ShiftHandoverEnv(1, EnvOptions())
    kept = [
        {"when": "Mon morning", "scope": "asset:press 2", "status": "open", "text": "Press 2 limited to 750 cycles/h."},
        {"when": "Tue night", "scope": "asset:press 2", "status": "open", "text": "Press 2 limited to 750 cycles/h!"},
        {"when": "Wed night", "scope": "line-3", "status": "resolved", "text": "S4 on line 3 replaced, jumper out."},
    ]
    assert env.keep(kept) == {"kept": 2, "repeats_skipped": 1}  # the second says nothing new
    by_name = env.recall("cycle limit", scope="press 2")["observations"]
    assert [o["scope"] for o in by_name] == ["asset:press 2"]
    elsewhere = env.recall("S4 jumper", scope="plant")
    assert "searched every scope" in elsewhere["hint"] and elsewhere["observations"][0]["scope"] == "line-3"
    assert env.read_handover()["scopes_kept"] == {"asset:press 2": 1, "line-3": 1}


def test_a_scope_matches_the_whole_name_only() -> None:
    items = [{"scope": s} for s in ("line-1", "line-12", "asset:press 2", "asset:press 22")]
    assert [i["scope"] for i in in_scope(items, "line-1", key=lambda i: i["scope"])] == ["line-1"]
    assert [i["scope"] for i in in_scope(items, "press 2", key=lambda i: i["scope"])] == ["asset:press 2"]


def test_fitting_never_keeps_a_resolved_line_over_an_open_one() -> None:
    long_open = "(line-3) open: " + "the coolant pump on line 3 still leaks at the seal " * 3
    record = f"(line-1) resolved: fixed.\n{long_open}"
    kept, cut = fit(record, budget=12)
    assert "resolved" not in kept and cut == 2
