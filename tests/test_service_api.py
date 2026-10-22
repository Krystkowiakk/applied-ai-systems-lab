"""The service's HTTP API: tokens and tenant isolation, streamed events, triage that degrades visibly."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from aisyslab.cli import main
from aisyslab.core.lab import EnvOptions
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ModelError, ScriptedModel, text_turn, tool_turn
from aisyslab.labs.plant_database.baseline import BLOCKED_ORDERS
from aisyslab.labs.plant_database.env import PlantDatabaseEnv
from aisyslab.runtime import ModelGateway, Route, Runtime, RuntimeConfig, Store
from aisyslab.runtime.api import create_app


def build(model: Any) -> tuple[Runtime, TestClient]:
    store = Store()
    gateway = ModelGateway({"default": [model], "triage": Route((model,), cache=True)}, store)
    runtime = Runtime(store, gateway, config=RuntimeConfig(heartbeat_every=None))
    return runtime, TestClient(create_app(runtime, poll_interval=0.01))


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def plant_database_model() -> ScriptedModel:
    expected = sorted(PlantDatabaseEnv(6, EnvOptions()).expected)
    return ScriptedModel(
        [tool_turn("run_query", sql=BLOCKED_ORDERS), tool_turn("submit_answer", work_order_ids=expected)]
    )


def events(client: TestClient, run_id: str, headers: dict[str, str]) -> list[tuple[int | None, str]]:
    out: list[tuple[int | None, str]] = []
    event_id: int | None = None
    with client.stream("GET", f"/v1/runs/{run_id}/events", headers=headers) as response:
        for line in response.iter_lines():
            if line.startswith("id: "):
                event_id = int(line[4:])
            elif line.startswith("event: "):
                out.append((event_id, line[7:]))
                event_id = None
    return out


def test_every_endpoint_needs_a_token_and_tenants_cannot_see_each_other() -> None:
    runtime, client = build(plant_database_model())
    _, token_a = runtime.store.create_tenant("plant-a")
    _, token_b = runtime.store.create_tenant("plant-b")
    assert client.get("/v1/sessions").status_code == 401
    assert client.get("/v1/sessions", headers=bearer("rt_wrong")).status_code == 401

    created = client.post("/v1/sessions", json={"agent": "plant-database", "seed": 6}, headers=bearer(token_a))
    assert created.status_code == 201
    run_id, thread_id = created.json()["run"]["id"], created.json()["thread_id"]
    runtime.run_until_idle()

    other = bearer(token_b)
    assert client.get(f"/v1/runs/{run_id}", headers=other).status_code == 404
    assert client.get(f"/v1/runs/{run_id}/events", headers=other).status_code == 404
    assert client.get(f"/v1/threads/{thread_id}/items", headers=other).status_code == 404
    assert client.post(f"/v1/threads/{thread_id}/messages", json={"text": "hi"}, headers=other).status_code == 404
    assert client.post(f"/v1/runs/{run_id}/cancel", headers=other).status_code == 404
    assert client.get("/v1/sessions", headers=other).json() == []
    assert client.get(f"/v1/runs/{run_id}", headers=bearer(token_a)).json()["status"] == "completed"


def test_a_run_streams_its_events_and_a_dropped_stream_resumes() -> None:
    runtime, client = build(plant_database_model())
    _, token = runtime.store.create_tenant("plant-a")
    headers = bearer(token)
    run_id = client.post("/v1/sessions", json={"agent": "plant-database", "seed": 6}, headers=headers).json()["run"][
        "id"
    ]
    runtime.run_until_idle()

    stream = events(client, run_id, headers)
    kinds = [kind for _, kind in stream]
    assert kinds[0] == "run_created" and kinds[-2:] == ["completed", "end"] and "verdict" in kinds
    resume_after = stream[2][0]
    assert resume_after is not None
    resumed = events(client, run_id, {**headers, "Last-Event-ID": str(resume_after)})
    assert resumed == stream[3:]  # exactly the events after the last one seen

    run = client.get(f"/v1/runs/{run_id}", headers=headers).json()
    assert run["result"]["verified"] and run["revision"]["agent"] == "plant-database"
    items = client.get(f"/v1/threads/{run['thread_id']}/items", headers=headers).json()
    assert [i["kind"] for i in items] == ["user", "assistant", "tool", "assistant", "tool"]


def test_triage_answers_with_a_fixed_shape_and_degrades_visibly() -> None:
    answers = iter(
        [
            text_turn('{"code": "HYD-LEAK", "priority": 1, "reason": "Oil under the hydraulic hose, line stopped."}'),
            text_turn("It is probably a leak somewhere."),
        ]
    )

    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        turn = next(answers, None)
        if turn is None:
            raise ModelError("provider down")
        return turn

    runtime, client = build(ScriptedModel(policy))
    _, token = runtime.store.create_tenant("plant-a")
    headers = bearer(token)

    def triage(text: str) -> dict[str, Any]:
        response = client.post("/v1/triage", json={"text": text}, headers=headers)
        assert response.status_code == 200
        return response.json()

    leak = "press 4: oil on the floor under the hydraulic hose. line stopped."
    assert triage(leak) == {
        "code": "HYD-LEAK",
        "priority": 1,
        "reason": "Oil under the hydraulic hose, line stopped.",
        "model": "scripted",
        "degraded": False,
        "note": "",
    }
    assert triage(leak)["degraded"] is False  # the same ticket again comes from the cache

    vague = triage("conveyor L3: proximity sensor flickers.")
    assert vague["degraded"] and vague["code"] == "ELEC-SENS" and "rejected" in vague["note"]
    down = triage("drive motor M7: motor trips on overload. safety risk.")
    assert down["degraded"] and down["code"] == "ELEC-MTR" and down["priority"] == 1 and "No model" in down["note"]
    unknown = triage("the forklift charger smells strange.")
    assert unknown["degraded"] and unknown["code"] is None


def test_waiting_runs_branches_cancellation_and_schedules_over_http() -> None:
    def policy(_s: str, messages: Sequence[Message], _t: Sequence[ToolSpec]) -> ModelTurn:
        last = messages[-1]
        return (
            tool_turn("ask_human", question="Which line?") if last.role == "user" else text_turn(f"On {last.content}")
        )

    runtime, client = build(ScriptedModel(policy))
    _, token = runtime.store.create_tenant("plant-a")
    headers = bearer(token)

    started = client.post(
        "/v1/sessions/bootstrap", json={"agent": "operations-assistant", "message": "Why the stop?"}, headers=headers
    ).json()
    runtime.run_until_idle()
    run = client.get(f"/v1/runs/{started['run']['id']}", headers=headers).json()
    assert run["status"] == "waiting" and run["questions"][0]["question"] == "Which line?"
    assert client.post(f"/v1/runs/{run['id']}/answer", json={"text": "line 3"}, headers=headers).status_code == 200
    assert client.post(f"/v1/runs/{run['id']}/answer", json={"text": "again"}, headers=headers).status_code == 409
    runtime.run_until_idle()
    assert client.get(f"/v1/runs/{run['id']}", headers=headers).json()["result"]["text"] == "On line 3"

    branch = client.post(f"/v1/threads/{started['thread_id']}/branch", json={"keep_until": 1}, headers=headers)
    assert branch.status_code == 201
    posted = client.post(f"/v1/threads/{branch.json()['thread_id']}/messages", json={"text": "Why?"}, headers=headers)
    assert posted.status_code == 202 and posted.json()["status"] == "pending"
    assert client.post(f"/v1/runs/{posted.json()['id']}/cancel", headers=headers).json() == {"cancelled": True}
    assert (
        client.post(f"/v1/threads/{started['thread_id']}/branch", json={"keep_until": 99}, headers=headers).status_code
        == 422
    )

    sessions = client.get("/v1/sessions", headers=headers).json()
    assert sessions[0]["origin"] == "bootstrap"
    assert client.post("/v1/sessions", json={"agent": "no-such-agent"}, headers=headers).status_code == 422
    assert client.post("/v1/sessions", json={"agent": "operations-assistant"}, headers=headers).status_code == 422

    fanout = {
        "name": "morning-briefing",
        "collectors": ["plant-database", "quality-gate"],
        "assembler": "briefing-writer",
        "collect_at": "0 4 * * 1-5",
        "assemble_at": "0 5 * * 1-5",
        "process": "morning-briefing",
    }
    assert len(client.post("/v1/fanouts", json=fanout, headers=headers).json()["schedules"]) == 2
    assert client.post("/v1/fanouts", json={**fanout, "collect_at": "0 25 * * *"}, headers=headers).status_code == 422
    assert client.post("/v1/fanouts", json={**fanout, "tz": "Mars/Base"}, headers=headers).status_code == 422
    check = {"name": "line-check", "cron": "*/15 * * * *", "agent": "line-incident"}
    assert client.post("/v1/schedules", json=check, headers=headers).status_code == 201
    assert client.post("/v1/schedules", json={**check, "cron": "every morning"}, headers=headers).status_code == 422
    assert (
        client.post("/v1/schedules", json={**check, "agent": "operations-assistant"}, headers=headers).status_code
        == 422
    )
    assert len(client.get("/v1/schedules", headers=headers).json()) == 3

    usage = client.get("/v1/usage", headers=headers).json()
    assert usage["tokens_used"] > 0 and usage["models"][0]["model"] == "scripted"
    page = client.get("/")
    assert page.status_code == 200 and "Maintenance request service" in page.text


def test_the_tenant_command_prints_a_token_that_works(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "service.db"
    assert main(["tenant", "plant-a", "--budget", "1000", "--db", str(db)]) == 0
    token = capsys.readouterr().out.strip().rsplit(" ", 1)[-1]
    store = Store(db)
    tenant = store.tenant_for_token(token)
    assert tenant is not None and tenant["name"] == "plant-a" and tenant["token_budget"] == 1000
    assert token not in db.read_bytes().decode("latin-1")  # only a hash is stored
