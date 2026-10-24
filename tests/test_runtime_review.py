"""Service failures found in review: two processes, a schedule that fails, a turn over budget,
an answer cut off, a call whose outcome is unknown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aisyslab.core.messages import ModelTurn, ToolCall
from aisyslab.core.model import ScriptedModel, text_turn
from aisyslab.core.tools import NoArgs, Tool, Toolbox
from aisyslab.runtime import ClaimLost, FanOut, ModelGateway, Runtime, RuntimeAgent, RuntimeConfig, Store
from aisyslab.runtime.engine import to_messages

ASSISTANT = RuntimeAgent("assistant", "Answers briefly.", "Answer briefly.")  # no standing task


class Clock:
    def __init__(self) -> None:
        self.now = 1_761_000_000.0

    def __call__(self) -> float:
        return self.now


def service(model: Any, clock: Clock, path: str | Path = ":memory:", **agents: RuntimeAgent) -> Runtime:
    store = Store(path, clock=clock)
    config = RuntimeConfig(heartbeat_every=None, claim_ttl=30.0)
    gateway = ModelGateway({"default": [model]}, store)
    return Runtime(store, gateway, {"assistant": ASSISTANT, **agents}, config=config)


def test_two_processes_with_the_same_worker_names_cannot_overwrite_each_other(tmp_path: Path) -> None:
    clock = Clock()
    first, second = Store(tmp_path / "s.db", clock=clock), Store(tmp_path / "s.db", clock=clock)
    tenant, _ = first.create_tenant("plant-a")
    session, thread = first.create_session(tenant, "assistant")
    run = first.create_run(
        tenant_id=tenant,
        session_id=session,
        thread_id=thread,
        agent="assistant",
        revision_id=first.revision("assistant", {}),
        text="Hi",
    )
    stale = first.claim("worker-0", ttl=30)["claimed_by"]
    clock.now += 31
    assert second.recover(max_retries=3) == [run["id"]]
    fresh = second.claim("worker-0", ttl=30)["claimed_by"]  # the same name, in the other process
    assert fresh != stale
    with pytest.raises(ClaimLost):
        first.record_turn(run["id"], stale, {"text": "late"})


def test_a_due_schedule_fires_once_even_when_two_schedulers_see_it(tmp_path: Path) -> None:
    clock = Clock()
    a = service(ScriptedModel([text_turn("ok")] * 4), clock, tmp_path / "s.db")
    b = service(ScriptedModel([text_turn("ok")] * 4), clock, tmp_path / "s.db")
    tenant, _ = a.store.create_tenant("plant-a")
    a.add_schedule(tenant, "line-check", "*/15 * * * *", "assistant", message="Is line 3 running?")
    clock.now += 15 * 60
    assert len(a.store.due_schedules(clock.now)) == len(b.store.due_schedules(clock.now)) == 1  # both see it
    assert len(a.tick()["fired"] + b.tick()["fired"]) == 1


def test_a_failing_schedule_records_its_error_and_the_scheduler_goes_on() -> None:
    clock = Clock()
    rt = service(ScriptedModel([text_turn("ok")] * 4), clock)
    tenant, _ = rt.store.create_tenant("plant-a")
    rt.add_schedule(tenant, "broken", "*/15 * * * *", "assistant", message="Check line 3.")
    rt.add_schedule(tenant, "fine", "*/15 * * * *", "assistant", message="Check line 4.")
    broken = next(s for s in rt.store.schedules(tenant) if s["name"] == "broken")
    action = json.dumps(broken["action"] | {"agent": "gone"})  # a row the scheduler cannot carry out
    with rt.store.tx() as db:
        db.execute("UPDATE schedules SET action = ? WHERE id = ?", (action, broken["id"]))
    clock.now += 15 * 60
    assert len(rt.tick()["fired"]) == 2
    errors = {s["name"]: s["last_error"] for s in rt.store.schedules(tenant)}
    assert errors["fine"] is None and "gone" in errors["broken"]
    assert all(s["next_at"] > clock.now for s in rt.store.schedules(tenant))  # both moved on


def test_a_fan_out_whose_collector_has_nothing_to_do_is_refused() -> None:
    rt = service(ScriptedModel([]), Clock())
    tenant, _ = rt.store.create_tenant("plant-a")
    briefing = FanOut("brief", ("assistant",), "assistant", collect_at="0 4 * * *", assemble_at="0 5 * * *")
    with pytest.raises(ValueError, match="no standing task"):
        rt.add_fanout(tenant, briefing)


def test_the_service_keeps_the_tool_budget_inside_a_turn() -> None:
    ran: list[int] = []

    def tools(_env: Any, _model: Any, _ctx: Any) -> Toolbox:
        return Toolbox([Tool("ping", "Ping.", NoArgs, lambda _: ran.append(1) or "pong")])

    pinger = RuntimeAgent("pinger", "Pings.", "Ping.", tools=tools, max_tool_calls=2)
    calls = [ToolCall(f"c{i}", "ping", {}) for i in range(5)]
    burst = ModelTurn(text="", tool_calls=calls, usage=text_turn("").usage, stop_reason="tool_use")
    rt = service(ScriptedModel([burst]), Clock(), pinger=pinger)
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "pinger", text="Go.")["run"]["id"]
    rt.run_until_idle()
    assert len(ran) == 2 and rt.store.run(run_id)["status"] == "failed"


def test_an_answer_cut_off_at_the_output_limit_is_not_a_completed_run() -> None:
    cut = text_turn("The blocked orders are")
    cut.stop_reason = "max_tokens"
    rt = service(ScriptedModel([cut]), Clock())
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "assistant", text="Which orders?")["run"]["id"]
    rt.run_until_idle()
    run = rt.store.run(run_id)
    assert run["status"] == "failed" and "cut off" in run["error"]


def test_a_call_that_started_and_never_finished_is_reported_as_unknown() -> None:
    calls = [{"id": "c1", "name": "post", "arguments": {}}, {"id": "c2", "name": "post", "arguments": {}}]
    items = [
        {"kind": "user", "data": {"text": "Post it."}},
        {"kind": "assistant", "data": {"text": "", "tool_calls": calls}},
        {"kind": "user", "data": {"text": "Did it work?"}},
    ]
    answered = {m.tool_call_id: json.loads(m.content)["error"] for m in to_messages(items, {"c1"}) if m.role == "tool"}
    assert answered["c1"].startswith("Outcome unknown") and answered["c2"].startswith("Not executed")
