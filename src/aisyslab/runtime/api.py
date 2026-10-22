"""HTTP API and web UI of the maintenance request service (FastAPI).

Every endpoint has a fixed shape; there is no open chat endpoint that passes anything to a
model. Requests return as soon as the work is recorded: a message becomes a run, and its
progress arrives as server-sent events that can be resumed after a dropped connection
(``Last-Event-ID``). Each tenant authenticates with its own token and sees only its own data;
another tenant's objects answer 404, exactly like objects that do not exist.
"""

# No ``from __future__ import annotations`` here: FastAPI reads the ``Annotated`` dependencies
# defined inside ``create_app`` at definition time.

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..core.messages import Message
from ..core.model import ModelError
from ..evals.graders import parse, schema
from ..evals.triage import PROMPT_HEADER, SYMPTOMS, URGENT, Ticket
from .cron import FanOut
from .engine import Runtime, UnknownAgent
from .gateway import BudgetExhausted, ModerationBlocked
from .store import TERMINAL, NotFound

TRIAGE_PROMPT = PROMPT_HEADER


class SessionIn(BaseModel):
    agent: str
    message: str = Field("", max_length=20_000)
    title: str = Field("", max_length=200)
    seed: int = Field(0, ge=0, le=1_000_000)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class AnswerIn(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)
    call_id: str | None = None


class BranchIn(BaseModel):
    keep_until: int = Field(ge=0)


class TriageIn(BaseModel):
    text: str = Field(min_length=3, max_length=2_000)


class TriageOut(BaseModel):
    code: str | None
    priority: int
    reason: str
    model: str | None
    degraded: bool
    note: str = ""


class ScheduleIn(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    cron: str
    agent: str
    message: str = Field("", max_length=20_000)
    seed: int = Field(0, ge=0)
    tz: str = "UTC"


class FanOutIn(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    collectors: list[str] = Field(min_length=1, max_length=10)
    assembler: str
    collect_at: str
    assemble_at: str
    process: str = ""
    seed: int = Field(0, ge=0)
    tz: str = "UTC"


def rule_based_triage(text: str) -> dict[str, Any]:
    """The degraded path: rules on known symptom wordings. Unknown wording stays unclassified
    instead of getting a guessed code."""
    lowered = text.lower()
    code = next((c for c, phrases in SYMPTOMS.items() if any(p in lowered for p in phrases)), None)
    reason = (
        f"Matched a known symptom wording for {code}."
        if code
        else "No known symptom wording; a planner must classify it."
    )
    return {"code": code, "priority": 1 if any(u in lowered for u in URGENT) else 2, "reason": reason}


def _sse(event: dict[str, Any]) -> str:
    return f"id: {event['id']}\nevent: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _public(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if k not in ("tenant_id", "claimed_by", "claim_expires")}


def create_app(runtime: Runtime, *, workers: int = 0, tick_every: float = 0.5, poll_interval: float = 0.25) -> FastAPI:
    """The app. With ``workers`` it also runs the scheduler and workers in background threads."""
    store = runtime.store

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if workers:
            runtime.start(workers=workers, tick_every=tick_every)
        try:
            yield
        finally:
            if workers:
                runtime.stop()

    app = FastAPI(title="Maintenance request service", version="0.2.0", lifespan=lifespan)

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, _exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": "Not found."}, status_code=404)

    @app.exception_handler(UnknownAgent)
    async def unknown_agent(_request: Request, exc: UnknownAgent) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def auth(authorization: Annotated[str, Header()] = "") -> dict[str, Any]:
        scheme, _, token = authorization.partition(" ")
        tenant = store.tenant_for_token(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if tenant is None:
            raise HTTPException(401, "Missing or unknown bearer token.", headers={"WWW-Authenticate": "Bearer"})
        return tenant

    Tenant = Annotated[dict[str, Any], Depends(auth)]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return resources.files("aisyslab.runtime").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/v1/agents")
    def agents(tenant: Tenant) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "description": a.description,
                "standing_task": a.task is not None,
                "delegates": list(a.delegates),
                "asks_people": a.ask_human,
            }
            for a in sorted(runtime.agents.values(), key=lambda a: a.name)
        ]

    def _start(body: SessionIn, tenant: dict[str, Any], origin: str) -> dict[str, Any]:
        try:
            out = runtime.start_session(
                tenant["id"], body.agent, text=body.message, title=body.title, seed=body.seed, origin=origin
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**out, "run": _public(out["run"])}

    @app.post("/v1/sessions", status_code=201)
    def create_session(body: SessionIn, tenant: Tenant) -> dict[str, Any]:
        return _start(body, tenant, "api")

    @app.post("/v1/sessions/bootstrap", status_code=201)
    def bootstrap(body: SessionIn, tenant: Tenant) -> dict[str, Any]:
        """For an external trigger (a cron job, a webhook): a session and its first run in one call."""
        return _start(body, tenant, "bootstrap")

    @app.get("/v1/sessions")
    def sessions(tenant: Tenant) -> list[dict[str, Any]]:
        return store.sessions(tenant["id"])

    @app.get("/v1/sessions/{session_id}")
    def session(session_id: str, tenant: Tenant) -> dict[str, Any]:
        return store.session(tenant["id"], session_id)

    @app.get("/v1/threads/{thread_id}/items")
    def items(thread_id: str, tenant: Tenant) -> list[dict[str, Any]]:
        return [
            {**i, "data": {k: v for k, v in i["data"].items() if k != "provider_state"}}
            for i in runtime.items(tenant["id"], thread_id)
        ]

    @app.post("/v1/threads/{thread_id}/messages", status_code=202)
    def post_message(thread_id: str, body: MessageIn, tenant: Tenant) -> dict[str, Any]:
        try:
            return _public(runtime.post_message(tenant["id"], thread_id, body.text))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/threads/{thread_id}/branch", status_code=201)
    def branch(thread_id: str, body: BranchIn, tenant: Tenant) -> dict[str, str]:
        try:
            return {"thread_id": runtime.branch(tenant["id"], thread_id, body.keep_until)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str, tenant: Tenant) -> dict[str, Any]:
        found = store.run(run_id, tenant["id"])
        return {
            **_public(found),
            "questions": store.pending_questions(run_id),
            "revision": store.revision_config(found["revision_id"]),
            "children": [_public(r) for r in store.runs_for(root_id=found["root_id"]) if r["parent_id"] == run_id],
        }

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        tenant: Tenant,
        after: int = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        store.run(run_id, tenant["id"])
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

        async def stream() -> AsyncIterator[str]:
            cursor, quiet = start, 0.0
            while True:
                batch = store.events(run_id, cursor)
                for event in batch:
                    yield _sse(event)
                if batch:
                    cursor, quiet = batch[-1]["id"], 0.0
                if store.run(run_id)["status"] in TERMINAL:
                    for event in store.events(run_id, cursor):  # written between the two reads
                        yield _sse(event)
                    yield "event: end\ndata: {}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(poll_interval)
                quiet += poll_interval
                if quiet >= 15:
                    yield ": keep-alive\n\n"
                    quiet = 0.0

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.post("/v1/runs/{run_id}/answer")
    def answer(run_id: str, body: AnswerIn, tenant: Tenant) -> dict[str, str]:
        try:
            return {"answered": runtime.answer(tenant["id"], run_id, body.text, body.call_id)}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, tenant: Tenant) -> dict[str, bool]:
        return {"cancelled": runtime.cancel(tenant["id"], run_id)}

    @app.post("/v1/triage")
    def triage(body: TriageIn, tenant: Tenant) -> TriageOut:
        """A fixed-shape classification. If no model answers or the answer fails validation, a
        rule-based fallback answers instead and the response says so (``degraded``)."""
        text = body.text.strip()
        ticket = Ticket("api", text, "", 0, "api")
        model = runtime.gateway.bind(tenant_id=tenant["id"], route="triage")
        try:
            turn = model.complete(
                system=TRIAGE_PROMPT, messages=[Message(role="user", content=text)], tools=[], max_output_tokens=300
            )
            grade = schema(ticket, turn.text)
            if grade.passed:
                data = parse(turn.text) or {}
                return TriageOut(
                    code=data["code"],
                    priority=data["priority"],
                    reason=data["reason"],
                    model=turn.model,
                    degraded=False,
                )
            note = f"The model's answer was rejected: {grade.note}."
        except ModerationBlocked as exc:
            raise HTTPException(422, str(exc)) from exc
        except BudgetExhausted as exc:
            raise HTTPException(429, str(exc)) from exc
        except ModelError as exc:
            note = f"No model answered: {exc}"
        return TriageOut(**rule_based_triage(text), model=None, degraded=True, note=note[:300])

    @app.post("/v1/fanouts", status_code=201)
    def add_fanout(body: FanOutIn, tenant: Tenant) -> dict[str, Any]:
        try:
            fanout = FanOut(
                name=body.name,
                collectors=tuple(body.collectors),
                assembler=body.assembler,
                collect_at=body.collect_at,
                assemble_at=body.assemble_at,
                process=body.process,
                seed=body.seed,
                tz=body.tz,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"schedules": runtime.add_fanout(tenant["id"], fanout)}

    @app.post("/v1/schedules", status_code=201)
    def add_schedule(body: ScheduleIn, tenant: Tenant) -> dict[str, str]:
        """A recurring request for one agent, e.g. a check every 15 minutes."""
        try:
            schedule_id = runtime.add_schedule(
                tenant["id"], body.name, body.cron, body.agent, message=body.message, seed=body.seed, tz=body.tz
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"schedule": schedule_id}

    @app.get("/v1/schedules")
    def schedules(tenant: Tenant) -> list[dict[str, Any]]:
        return store.schedules(tenant["id"])

    @app.get("/v1/usage")
    def usage(tenant: Tenant) -> dict[str, Any]:
        return store.usage(tenant["id"])

    return app
