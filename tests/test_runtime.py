"""The service runtime: durable runs, recovery, waiting states, the model gateway and time triggers."""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from aisyslab.core.lab import EnvOptions
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ModelError, ScriptedModel, text_turn, tool_turn
from aisyslab.core.tools import NoArgs, Tool, Toolbox
from aisyslab.labs.plant_database.baseline import BLOCKED_ORDERS
from aisyslab.labs.plant_database.env import PlantDatabaseEnv
from aisyslab.runtime import (
    BudgetExhausted,
    ClaimLost,
    Cron,
    FanOut,
    ModelGateway,
    ModelUnavailable,
    ModerationBlocked,
    NotFound,
    PhraseModerator,
    Route,
    Runtime,
    RuntimeAgent,
    RuntimeConfig,
    Store,
    idempotency_key,
)


class Clock:
    def __init__(self, start: float = 1_761_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Crash(BaseException):
    """Stands in for a worker process that dies: nothing below it gets to clean up."""


ASSISTANT = RuntimeAgent("assistant", "Answers briefly.", "Answer briefly.")


def service(
    model: Any,
    *,
    clock: Clock | None = None,
    agents: dict[str, RuntimeAgent] | None = None,
    path: str | Path = ":memory:",
    **config: Any,
) -> Runtime:
    store = Store(path, clock=clock or Clock())
    models = model if isinstance(model, list) else [model]
    settings = {"heartbeat_every": None, "claim_ttl": 30.0, "retry_delay": 10.0, **config}
    return Runtime(store, ModelGateway({"default": models}, store), agents, config=RuntimeConfig(**settings))


def blocked(seed: int) -> list[str]:
    return sorted(PlantDatabaseEnv(seed, EnvOptions()).expected)


def test_a_lab_agent_runs_as_a_durable_job() -> None:
    model = ScriptedModel(
        [tool_turn("run_query", sql=BLOCKED_ORDERS), tool_turn("submit_answer", work_order_ids=blocked(6))]
    )
    rt = service(model)
    tenant, _ = rt.store.create_tenant("plant-a")
    out = rt.start_session(tenant, "plant-database", seed=6)
    assert out["run"]["status"] == "pending"  # the request only records the work

    rt.run_until_idle()
    run = rt.store.run(out["run"]["id"])
    assert run["status"] == "completed" and run["result"]["verified"] and run["result"]["stop_reason"] == "goal_reached"
    assert run["result"]["answer"]["arguments"] == {"work_order_ids": blocked(6)}
    assert [i["kind"] for i in rt.store.history(out["thread_id"])] == ["user", "assistant", "tool", "assistant", "tool"]
    kinds = [e["kind"] for e in rt.store.events(run["id"])]
    assert kinds[:3] == ["run_created", "run_started", "generation"] and "verdict" in kinds and kinds[-1] == "completed"
    assert rt.store.revision_config(run["revision_id"])["models"] == ["scripted"]
    assert rt.store.usage(tenant)["tokens_used"] > 0


def test_a_crashed_worker_is_recovered_and_nothing_finished_is_repeated() -> None:
    crashed: list[bool] = []

    def policy(_system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        if not any(m.role == "tool" for m in messages):
            return tool_turn("run_query", sql=BLOCKED_ORDERS)
        if not crashed:
            crashed.append(True)
            raise Crash
        return tool_turn("submit_answer", work_order_ids=blocked(6))

    clock = Clock()
    rt = service(ScriptedModel(policy), clock=clock)
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "plant-database", seed=6)["run"]["id"]

    with pytest.raises(Crash):
        rt.worker("doomed").work_once()
    assert rt.store.run(run_id)["status"] == "running"  # the claim outlives the worker...
    assert not rt.tick()["recovered"]  # ...until it expires
    clock.advance(31)
    assert rt.tick()["recovered"] == [run_id]

    rt.run_until_idle()
    run = rt.store.run(run_id)
    assert run["status"] == "completed" and run["result"]["verified"] and run["retries"] == 1
    assert [e["attempts"] for e in rt.store.executions(run_id)] == [1, 1]  # the query ran once
    kinds = [e["kind"] for e in rt.store.events(run_id)]
    assert "recovered" in kinds and kinds.count("run_started") == 2


class Ledger:
    """An external system that keeps its own state and honours idempotency keys."""

    def __init__(self, _seed: int) -> None:
        self.postings: dict[str | None, int] = {}

    def post(self, key: str | None, amount: int) -> dict[str, int]:
        self.postings.setdefault(key, amount)
        return {"posted": amount, "postings": len(self.postings)}


class PostArgs(BaseModel):
    amount: int


def test_a_call_interrupted_mid_execution_is_retried_with_its_idempotency_key() -> None:
    ledgers: list[Ledger] = []
    crash_once = [True]

    def world(seed: int) -> Ledger:
        ledgers.append(Ledger(seed))
        return ledgers[-1]

    def tools(env: Ledger, _model: Any, _ctx: Any) -> Toolbox:
        def post(args: PostArgs) -> dict[str, int]:
            result = env.post(idempotency_key.get(), args.amount)
            if crash_once:
                crash_once.pop()
                raise Crash  # posted, but the result never reached the journal
            return result

        return Toolbox([Tool("post_entry", "Post a ledger entry.", PostArgs, post)])

    agent = RuntimeAgent("bookkeeper", "Posts entries.", "Post the entry.", world=world, tools=tools)
    clock = Clock()
    rt = service(
        ScriptedModel([tool_turn("post_entry", amount=120), text_turn("Posted.")]),
        clock=clock,
        agents={"bookkeeper": agent},
    )
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "bookkeeper", text="Post 120 for the pump repair.")["run"]["id"]

    with pytest.raises(Crash):
        rt.worker().work_once()
    clock.advance(31)
    rt.run_until_idle()

    assert rt.store.run(run_id)["status"] == "completed"
    assert len(ledgers) == 1 and len(ledgers[0].postings) == 1  # executed twice, applied once
    assert rt.store.executions(run_id)[0]["attempts"] == 2
    assert "tool_retry" in [e["kind"] for e in rt.store.events(run_id)]


def test_a_restarted_service_rebuilds_the_world_by_replaying_the_journal(tmp_path: Path) -> None:
    first = service(
        ScriptedModel([tool_turn("run_query", sql=BLOCKED_ORDERS)]), path=tmp_path / "s.db", yield_after_steps=2
    )
    tenant, _ = first.store.create_tenant("plant-a")
    run_id = first.start_session(tenant, "plant-database", seed=6)["run"]["id"]
    first.worker().work_once()  # a model turn and the query, then the run goes back to the queue
    assert first.store.run(run_id)["status"] == "pending"
    first.store.close()

    second = service(ScriptedModel([tool_turn("submit_answer", work_order_ids=blocked(6))]), path=tmp_path / "s.db")
    second.run_until_idle()
    run = second.store.run(run_id)
    assert run["status"] == "completed" and run["result"]["verified"]
    rebuilt = [e["data"] for e in second.store.events(run_id) if e["kind"] == "world_rebuilt"]
    assert rebuilt == [{"replayed_calls": 1}]


def test_a_replay_that_does_not_match_the_journal_fails_the_run(tmp_path: Path) -> None:
    readings = itertools.count()

    def tools(_env: Any, _model: Any, _ctx: Any) -> Toolbox:
        return Toolbox([Tool("read_meter", "Read the meter.", NoArgs, lambda _args: {"reading": next(readings)})])

    agents = {
        "meter": RuntimeAgent("meter", "Reads a meter.", "Read the meter.", world=lambda _s: object(), tools=tools)
    }
    first = service(
        ScriptedModel([tool_turn("read_meter")]), agents=agents, path=tmp_path / "s.db", yield_after_steps=2
    )
    tenant, _ = first.store.create_tenant("plant-a")
    run_id = first.start_session(tenant, "meter", text="Read it.")["run"]["id"]
    first.worker().work_once()
    first.store.close()

    second = service(ScriptedModel([text_turn("Done.")]), agents=agents, path=tmp_path / "s.db")
    second.run_until_idle()
    run = second.store.run(run_id)
    assert run["status"] == "failed" and "different result" in run["error"]


def test_a_worker_that_lost_its_claim_cannot_write() -> None:
    clock = Clock()
    store = Store(clock=clock)
    tenant, _ = store.create_tenant("plant-a")
    session_id, thread_id = store.create_session(tenant, "assistant")
    run = store.create_run(
        tenant_id=tenant,
        session_id=session_id,
        thread_id=thread_id,
        agent="assistant",
        revision_id=store.revision("assistant", {}),
        text="Hi",
    )
    assert store.claim("A", ttl=30)["id"] == run["id"]
    assert store.claim("B", ttl=30) is None  # one owner at a time
    clock.advance(31)
    assert store.recover(max_retries=3) == [run["id"]]
    assert store.claim("B", ttl=30)["id"] == run["id"]
    for late_write in (
        lambda: store.record_turn(run["id"], "A", {"text": "late"}),
        lambda: store.heartbeat(run["id"], "A", 30),
        lambda: store.finish(run["id"], "A", "completed"),
    ):
        with pytest.raises(ClaimLost):
            late_write()
    assert store.record_turn(run["id"], "B", {"text": "on time", "tool_calls": []}) == 2


def test_a_question_to_a_person_is_a_waiting_state_not_a_blocked_worker() -> None:
    def policy(_system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        last = messages[-1]
        if last.role == "user":
            return tool_turn("ask_human", question="Line 3 or line 4?")
        return text_turn(f"Checking {last.content}")

    rt = service(ScriptedModel(policy))
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "operations-assistant", text="Why did the line stop?")["run"]["id"]
    rt.run_until_idle()
    assert rt.store.run(run_id)["status"] == "waiting"
    assert [q["question"] for q in rt.store.pending_questions(run_id)] == ["Line 3 or line 4?"]

    rt.answer(tenant, run_id, "Line 3.")
    with pytest.raises(ValueError, match="not waiting"):
        rt.answer(tenant, run_id, "Line 4.")
    rt.run_until_idle()
    run = rt.store.run(run_id)
    assert run["status"] == "completed" and run["result"]["text"] == "Checking Line 3."
    kinds = [e["kind"] for e in rt.store.events(run_id)]
    assert kinds.index("waiting") < kinds.index("answered") < kinds.index("resumed") < kinds.index("completed")


def test_delegation_runs_a_child_and_returns_its_verified_result() -> None:
    def policy(system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        last = messages[-1]
        if system.startswith("You answer questions from the shift lead"):
            if last.role == "user":
                return tool_turn(
                    "delegate_to_agent", agent="plant-database", task="Which line-3 work orders are blocked?"
                )
            report = json.loads(last.content)
            ids = report["result"]["answer"]["arguments"]["work_order_ids"]
            return text_turn(f"Blocked: {', '.join(ids)} (verified: {report['result']['verified']}).")
        if not any(m.role == "tool" for m in messages):
            return tool_turn("run_query", sql=BLOCKED_ORDERS)
        return tool_turn("submit_answer", work_order_ids=blocked(6))

    rt = service(ScriptedModel(policy))
    tenant, _ = rt.store.create_tenant("plant-a")
    root_id = rt.start_session(tenant, "operations-assistant", text="Which orders are blocked?", seed=6)["run"]["id"]
    rt.run_until_idle()

    root = rt.store.run(root_id)
    assert root["status"] == "completed" and root["result"]["text"].endswith("(verified: True).")
    assert all(i in root["result"]["text"] for i in blocked(6))
    [child] = [r for r in rt.store.runs_for(root_id=root_id) if r["parent_id"] == root_id]
    assert child["agent"] == "plant-database" and child["depth"] == 1 and child["result"]["verified"]
    assert {e["agent"] for e in rt.store.events(root_id)} == {"operations-assistant", "plant-database"}
    with pytest.raises(NotFound):  # a child's thread belongs to its parent run
        rt.post_message(tenant, child["thread_id"], "Hello?")


def test_messages_sent_while_a_run_works_wait_in_the_thread_queue() -> None:
    rt = service(
        ScriptedModel([text_turn("First answer."), text_turn("Second answer.")]), agents={"assistant": ASSISTANT}
    )
    tenant, _ = rt.store.create_tenant("plant-a")
    out = rt.start_session(tenant, "assistant", text="First question?")
    queued = rt.post_message(tenant, out["thread_id"], "Second question?")
    assert queued["status"] == "queued"
    assert [i["kind"] for i in rt.store.history(out["thread_id"])] == ["user"]  # not in the conversation yet

    rt.run_until_idle()
    texts = [i["data"]["text"] for i in rt.store.history(out["thread_id"])]
    assert texts == ["First question?", "First answer.", "Second question?", "Second answer."]


def test_editing_an_earlier_message_branches_and_keeps_the_original() -> None:
    script = [text_turn("Pump 2."), text_turn("Use the spare."), text_turn("Pump 3 then.")]
    rt = service(ScriptedModel(script), agents={"assistant": ASSISTANT})
    tenant, _ = rt.store.create_tenant("plant-a")
    thread = rt.start_session(tenant, "assistant", text="Which pump?")["thread_id"]
    rt.run_until_idle()
    rt.post_message(tenant, thread, "What now?")
    rt.run_until_idle()

    branch = rt.branch(tenant, thread, keep_until=2)
    rt.post_message(tenant, branch, "And pump 3?")
    rt.run_until_idle()

    def texts(thread_id: str) -> list[str]:
        return [i["data"]["text"] for i in rt.store.history(thread_id)]

    assert texts(branch) == ["Which pump?", "Pump 2.", "And pump 3?", "Pump 3 then."]
    assert texts(thread) == ["Which pump?", "Pump 2.", "What now?", "Use the spare."]
    assert [i["seq"] for i in rt.store.history(branch)] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="keep_until"):
        rt.branch(tenant, thread, keep_until=9)


def test_the_answer_streams_as_text_events() -> None:
    text = "The pump on line 3 has a worn bearing; replace it before the next shift starts. " * 2
    rt = service(ScriptedModel([text_turn(text)]), agents={"assistant": ASSISTANT})
    tenant, _ = rt.store.create_tenant("plant-a")
    run_id = rt.start_session(tenant, "assistant", text="Status?")["run"]["id"]
    rt.run_until_idle()
    deltas = [e["data"]["text"] for e in rt.store.events(run_id) if e["kind"] == "text_delta"]
    assert "".join(deltas) == text and len(deltas) > 1


class Down:
    """A provider that is down."""

    name = "primary"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **_kwargs: Any) -> ModelTurn:
        self.calls += 1
        raise ModelError("503 from the provider")


def test_the_gateway_falls_back_and_stops_calling_a_model_that_keeps_failing() -> None:
    clock = Clock()
    store = Store(clock=clock)
    down, backup = Down(), ScriptedModel(lambda *_: text_turn("ok"), name="backup")
    gateway = ModelGateway({"default": [down, backup]}, store, failure_threshold=2, cooldown=60)
    tenant, _ = store.create_tenant("plant-a")
    events: list[str] = []
    model = gateway.bind(tenant_id=tenant, on_event=lambda kind, _data: events.append(kind))

    def ask() -> str:
        return model.complete(system="s", messages=[Message(role="user", content="hi")], tools=[]).model

    assert [ask() for _ in range(4)] == ["backup"] * 4
    assert down.calls == 2  # skipped while its breaker is open
    clock.advance(61)
    ask()
    assert down.calls == 3  # one trial after the cooldown; it fails and the breaker opens again
    ask()
    assert down.calls == 3 and events.count("model_failed") == 3
    calls = store.model_calls(tenant)
    assert sum(not c["ok"] for c in calls) == 3 and sum(c["ok"] for c in calls) == 6

    with pytest.raises(ModelUnavailable):
        ModelGateway({"default": [Down()]}, store).bind(tenant_id=tenant).complete(
            system="s", messages=[Message(role="user", content="hi")], tools=[]
        )


def test_budgets_moderation_and_the_cache_are_enforced_in_the_gateway() -> None:
    store = Store()
    model = ScriptedModel(lambda *_: text_turn('{"code": "HYD-LEAK"}'), name="classifier")
    gateway = ModelGateway(
        {"default": [model], "triage": Route((model,), cache=True)},
        store,
        moderator=PhraseModerator({"threats": ["burn the plant"]}),
    )
    tenant, _ = store.create_tenant("plant-a", token_budget=200)
    ticket = [Message(role="user", content="oil under the hydraulic hose")]

    first = gateway.bind(tenant_id=tenant, route="triage").complete(system="classify", messages=ticket, tools=[])
    again = gateway.bind(tenant_id=tenant, route="triage").complete(system="classify", messages=ticket, tools=[])
    assert model.calls == 1 and again.text == first.text and again.usage.input_tokens == 0

    other, _ = store.create_tenant("plant-b")
    gateway.bind(tenant_id=other, route="triage").complete(system="classify", messages=ticket, tools=[])
    assert model.calls == 2  # the cache is per tenant

    with pytest.raises(ModerationBlocked, match="threats"):
        gateway.bind(tenant_id=tenant).complete(
            system="s", messages=[Message(role="user", content="I will burn the plant down")], tools=[]
        )
    assert model.calls == 2  # nothing reached the model

    while store.budget_left(tenant) > 0:
        gateway.bind(tenant_id=tenant).complete(system="s" * 200, messages=ticket, tools=[])
    with pytest.raises(BudgetExhausted):
        gateway.bind(tenant_id=tenant).complete(system="s", messages=ticket, tools=[])


def test_reasoning_state_goes_back_only_to_the_model_that_wrote_it() -> None:
    seen: dict[str, list[Any]] = {}

    def recorder(name: str) -> ScriptedModel:
        def policy(_system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
            seen[name] = [m.provider_state for m in messages if m.role == "assistant"]
            turn = text_turn("ok")
            turn.provider_state = [f"{name}-reasoning"]
            return turn

        return ScriptedModel(policy, name=name)

    store = Store()
    tenant, _ = store.create_tenant("plant-a")
    via_a = ModelGateway({"default": [recorder("a")]}, store).bind(tenant_id=tenant)
    turn = via_a.complete(system="s", messages=[Message(role="user", content="hi")], tools=[])
    assert turn.provider_state == {"by": "a", "state": ["a-reasoning"]}

    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="ok", provider_state=turn.provider_state),
        Message(role="user", content="again"),
    ]
    via_a.complete(system="s", messages=history, tools=[])
    ModelGateway({"default": [recorder("b")]}, store).bind(tenant_id=tenant).complete(
        system="s", messages=history, tools=[]
    )
    assert seen["a"] == [["a-reasoning"]] and seen["b"] == [None]


def test_cron_expressions() -> None:
    friday = datetime(2026, 10, 23, 6, 0, tzinfo=UTC)
    assert Cron("0 5 * * 1-5").next_after(friday) == datetime(2026, 10, 26, 5, 0, tzinfo=UTC)
    assert Cron("*/15 * * * *").next_after(datetime(2026, 10, 23, 6, 7, tzinfo=UTC)) == datetime(
        2026, 10, 23, 6, 15, tzinfo=UTC
    )
    assert Cron("30 4 1,15 * *").next_after(datetime(2026, 10, 16, tzinfo=UTC)) == datetime(
        2026, 11, 1, 4, 30, tzinfo=UTC
    )
    assert Cron("0 9 * * 7").next_after(friday).isoweekday() == 7
    with pytest.raises(ValueError, match="never fires"):
        Cron("0 0 31 2 *").next_after(friday)
    for bad in ("* * * *", "61 * * * *", "a * * * *", "*/0 * * * *", "1-0 * * * *"):
        with pytest.raises(ValueError):
            Cron(bad)
    warsaw = Cron("0 6 * * *").next_epoch(friday.timestamp(), "Europe/Warsaw")
    assert datetime.fromtimestamp(warsaw, ZoneInfo("Europe/Warsaw")).hour == 6


def test_a_recurring_check_fires_on_its_schedule_and_only_once_after_a_gap() -> None:
    clock = Clock(datetime(2026, 10, 26, 3, 59, tzinfo=UTC).timestamp())
    rt = service(
        ScriptedModel(lambda *_: text_turn("Line 3 is running.")), clock=clock, agents={"assistant": ASSISTANT}
    )
    tenant, _ = rt.store.create_tenant("plant-a")
    rt.add_schedule(tenant, "line-check", "*/15 * * * *", "assistant", message="Is line 3 running?")
    with pytest.raises(ValueError, match="standing task"):
        rt.add_schedule(tenant, "empty", "*/15 * * * *", "assistant")

    clock.advance(60)  # 04:00
    assert len(rt.tick()["fired"]) == 1
    rt.run_until_idle()
    clock.advance(5 * 3600)  # the service was down for five hours
    assert len(rt.tick()["fired"]) == 1  # one firing, not twenty missed ones
    rt.run_until_idle()
    sessions = rt.store.sessions(tenant)
    assert len(sessions) == 2 and {s["last_status"] for s in sessions} == {"completed"}
    [schedule] = rt.store.schedules(tenant)
    assert datetime.fromtimestamp(schedule["next_at"], UTC).strftime("%H:%M") == "09:15"


def test_the_morning_briefing_takes_what_is_finished_and_names_what_is_missing() -> None:
    def policy(_system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        names = {t.name for t in tools}
        done = any(m.role == "tool" for m in messages)
        if "train_tree" in names:
            raise ModelError("the quality model is offline")
        if "run_query" in names:
            return (
                tool_turn("submit_answer", work_order_ids=blocked(0))
                if done
                else tool_turn("run_query", sql=BLOCKED_ORDERS)
            )
        if not done:
            return tool_turn("read_process", file="morning-briefing/assemble")
        return text_turn(
            "## Stoppage\nMISSING: line-incident\n## Blocked work orders\n...\n## Held batches\nMISSING: quality-gate"
        )

    monday_0359 = datetime(2026, 10, 26, 3, 59, tzinfo=UTC).timestamp()
    clock = Clock(monday_0359)
    rt = service(ScriptedModel(policy), clock=clock)
    tenant, _ = rt.store.create_tenant("plant-a")
    briefing = FanOut(
        name="morning-briefing",
        collectors=("line-incident", "plant-database", "quality-gate"),
        assembler="briefing-writer",
        collect_at="0 4 * * 1-5",
        assemble_at="0 5 * * 1-5",
        process="morning-briefing",
    )
    rt.add_fanout(tenant, briefing)
    # Today line-incident stops at once (no step budget), so its report fails.
    rt.agents["line-incident"] = RuntimeAgent("line-incident", "Out of service today.", "-", max_steps=0)

    clock.advance(60)  # 04:00: the collectors start together
    assert len(rt.tick()["fired"]) == 1
    rt.run_until_idle()
    clock.advance(3600)  # 05:00: the assembler takes what is there
    rt.tick()
    rt.run_until_idle()

    sessions = rt.store.sessions(tenant)
    writer = next(s for s in sessions if s["agent"] == "briefing-writer")
    thread = rt.store.session(tenant, writer["id"])["threads"][0]["id"]
    folder = rt.store.history(thread)[0]["data"]["text"]
    assert "## plant-database: completed, verified by its check" in folder
    assert "Missing at assembly time: line-incident (failed), quality-gate (pending)" in folder
    tool_result = json.loads(rt.store.history(thread)[2]["data"]["content"])
    assert tool_result["file"] == "morning-briefing/assemble" and "MISSING" in tool_result["text"]
    assert writer["last_status"] == "completed"
    assert {s["origin"] for s in sessions} == {"schedule"}
    next_runs = sorted(s["next_at"] for s in rt.store.schedules(tenant))
    assert [datetime.fromtimestamp(t, UTC).strftime("%a %H:%M") for t in next_runs] == ["Tue 04:00", "Tue 05:00"]
