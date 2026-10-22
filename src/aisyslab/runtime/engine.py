"""The durable runtime: a message becomes a job, workers claim jobs and advance them step by step.

A request never runs inside the HTTP call that sent it. The API records the message and a run,
then returns. A worker claims the run and advances it one step at a time, either one model turn
or the tool calls that turn asked for, and writes every step before taking the next. A claim
expires unless a heartbeat extends it, so when a worker dies the scheduler hands the run to
another worker, which continues from the last written step: nothing finished is repeated.

Waiting is a state, not a blocked thread. A run that asks a person a question, or hands work to
another agent, stores what it waits for and leaves the worker; the scheduler resumes it when
the answer or the child's result is in. The scheduler's order is fixed: child results, resumed
runs, recovery of dead runs, queued messages, then timers.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from ..core.agent import AgentConfig
from ..core.lab import RunContext
from ..core.messages import Message, ModelTurn, ToolCall, ToolSpec
from ..core.model import ModelError
from ..core.tools import Toolbox
from ..core.trace import Tracer, preview
from .agents import (
    ASK_HUMAN,
    DELEGATE,
    ProcessLibrary,
    RuntimeAgent,
    ask_human_spec,
    default_agents,
    delegate_spec,
)
from .cron import Cron, FanOut
from .gateway import GatewayModel, ModelGateway, ModelUnavailable
from .store import ChildRun, ClaimLost, Dependency, NotFound, Store, new_id
from .worlds import ReplayDiverged, Worlds

# The id of the tool call being executed. A tool that changes an external system passes it on
# as an idempotency key, so a call repeated after a crash is applied once.
idempotency_key: ContextVar[str | None] = ContextVar("idempotency_key", default=None)


class UnknownAgent(LookupError):
    pass


@dataclass
class RuntimeConfig:
    claim_ttl: float = 60.0
    heartbeat_every: float | None = 15.0  # None: heartbeats only between steps (tests)
    max_retries: int = 3
    yield_after_steps: int = 8  # then the run goes back to the queue so others get a turn
    max_depth: int = 3  # how deep delegation may nest
    retry_delay: float = 30.0  # wait after every model on the route failed
    stream_flush_chars: int = 48


def jsonable(value: Any) -> Any:
    """Provider objects (e.g. SDK content blocks) as plain JSON, so they can be stored and replayed."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def turn_data(turn: ModelTurn) -> dict[str, Any]:
    return {
        "text": turn.text,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments, "parse_error": c.parse_error}
            for c in turn.tool_calls
        ],
        "model": turn.model,
        "stop_reason": turn.stop_reason,
        "usage": asdict(turn.usage),
        "provider_state": jsonable(turn.provider_state),
    }


def _calls(data: dict[str, Any]) -> list[ToolCall]:
    return [ToolCall(c["id"], c["name"], c["arguments"], c.get("parse_error")) for c in data["tool_calls"]]


def to_messages(items: Sequence[dict[str, Any]]) -> list[Message]:
    """The thread as the model sees it. A call left without a result by a run that stopped gets
    an explicit error result, because providers reject a call without one."""
    out: list[Message] = []
    open_calls: dict[str, str] = {}
    for item in items:
        data, kind = item["data"], item["kind"]
        if kind == "tool":
            open_calls.pop(data["call_id"], None)
            out.append(
                Message(
                    role="tool",
                    content=data["content"],
                    tool_call_id=data["call_id"],
                    tool_name=data["name"],
                    is_error=data["is_error"],
                )
            )
            continue
        for call_id, name in open_calls.items():
            out.append(
                Message(
                    role="tool",
                    content=json.dumps({"error": "Not executed: the run stopped before this call."}),
                    tool_call_id=call_id,
                    tool_name=name,
                    is_error=True,
                )
            )
        open_calls = {}
        if kind == "user":
            out.append(Message(role="user", content=data["text"]))
        else:
            calls = _calls(data)
            out.append(
                Message(
                    role="assistant", content=data["text"], tool_calls=calls, provider_state=data.get("provider_state")
                )
            )
            open_calls = {c.id: c.name for c in calls}
    return out


def unanswered(items: Sequence[dict[str, Any]]) -> list[ToolCall]:
    """Calls of the latest model turn that have no result yet."""
    for index in range(len(items) - 1, -1, -1):
        kind = items[index]["kind"]
        if kind == "user":
            return []
        if kind == "assistant":
            answered = {i["data"]["call_id"] for i in items[index + 1 :] if i["kind"] == "tool"}
            return [c for c in _calls(items[index]["data"]) if c.id not in answered]
    return []


class _RunTracer(Tracer):
    """Events raised inside tools (retries, verdicts, model calls in tools) join the run's stream."""

    def __init__(self, store: Store, run: dict[str, Any]) -> None:
        super().__init__()
        self._store, self._run = store, run

    def event(self, kind: str, **data: Any) -> None:
        super().event(kind, **data)
        self._store.emit(self._run, kind, jsonable(data))


class _TextSink:
    """Streamed text, written as events in chunks rather than one row per token."""

    def __init__(self, store: Store, run: dict[str, Any], flush_chars: int) -> None:
        self._store, self._run, self._flush_chars = store, run, flush_chars
        self._parts: list[str] = []
        self._size = 0

    def __call__(self, text: str) -> None:
        self._parts.append(text)
        self._size += len(text)
        if self._size >= self._flush_chars:
            self.flush()

    def flush(self) -> None:
        if self._parts:
            self._store.emit(self._run, "text_delta", {"text": "".join(self._parts)})
            self._parts, self._size = [], 0

    def reset(self) -> None:
        """Drop what a failed model streamed; the UI clears its draft on ``stream_reset``."""
        self._parts, self._size = [], 0
        self._store.emit(self._run, "stream_reset", {})


@dataclass
class _State:
    model: GatewayModel
    env: Any
    toolbox: Toolbox
    specs: list[ToolSpec]
    system: str


class Runtime:
    def __init__(
        self,
        store: Store,
        gateway: ModelGateway,
        agents: dict[str, RuntimeAgent] | None = None,
        *,
        config: RuntimeConfig | None = None,
        processes: ProcessLibrary | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.agents = dict(agents) if agents is not None else default_agents()
        self.config = config or RuntimeConfig()
        self.processes = processes if processes is not None else ProcessLibrary.default()
        self.worlds = Worlds(store)
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []

    # ----- requests -------------------------------------------------------------------------
    def agent(self, name: str) -> RuntimeAgent:
        if name not in self.agents:
            raise UnknownAgent(f"Unknown agent '{name}'. Available: {', '.join(sorted(self.agents))}.")
        return self.agents[name]

    def _revision(self, agent: RuntimeAgent) -> int:
        return self.store.revision(agent.name, agent.snapshot(self.gateway.model_names(agent.route)))

    def start_session(
        self,
        tenant_id: str,
        agent_name: str,
        *,
        text: str = "",
        title: str = "",
        seed: int = 0,
        origin: str = "api",
        group_key: str | None = None,
    ) -> dict[str, Any]:
        agent = self.agent(agent_name)
        message = agent.first_message(text, seed)
        if not message:
            raise ValueError(f"'{agent_name}' has no standing task; send a message.")
        session_id, thread_id = self.store.create_session(tenant_id, agent.name, title=title, seed=seed, origin=origin)
        run = self.store.create_run(
            tenant_id=tenant_id,
            session_id=session_id,
            thread_id=thread_id,
            agent=agent.name,
            revision_id=self._revision(agent),
            text=message,
            group_key=group_key,
        )
        return {"session_id": session_id, "thread_id": thread_id, "run": run}

    def post_message(self, tenant_id: str, thread_id: str, text: str) -> dict[str, Any]:
        thread = self._public_thread(tenant_id, thread_id)
        agent = self.agent(thread["agent"])
        if self.store.last_seq(thread_id) == 0:  # an empty branch starts over with the standing task
            text = agent.first_message(text, thread["seed"])
        if not text.strip():
            raise ValueError("The message is empty.")
        return self.store.create_run(
            tenant_id=tenant_id,
            session_id=thread["session_id"],
            thread_id=thread_id,
            agent=agent.name,
            revision_id=self._revision(agent),
            text=text,
        )

    def branch(self, tenant_id: str, thread_id: str, keep_until: int) -> str:
        self._public_thread(tenant_id, thread_id)
        return self.store.branch(thread_id, keep_until)

    def items(self, tenant_id: str, thread_id: str) -> list[dict[str, Any]]:
        self._public_thread(tenant_id, thread_id)
        return self.store.history(thread_id)

    def answer(self, tenant_id: str, run_id: str, text: str, call_id: str | None = None) -> str:
        return self.store.answer(run_id, tenant_id, text, call_id)

    def cancel(self, tenant_id: str, run_id: str) -> bool:
        return self.store.cancel(run_id, tenant_id)

    def _public_thread(self, tenant_id: str, thread_id: str) -> dict[str, Any]:
        thread = self.store.thread(thread_id, tenant_id)
        if thread["private"]:  # a child agent's thread belongs to its parent run
            raise NotFound(thread_id)
        return thread

    # ----- time triggers --------------------------------------------------------------------
    def add_fanout(self, tenant_id: str, fanout: FanOut) -> list[str]:
        for name in (*fanout.collectors, fanout.assembler):
            self.agent(name)
        now = self.store.clock()
        return [
            self.store.add_schedule(
                tenant_id,
                f"{fanout.name}:{kind}",
                cron,
                {"type": kind, **fanout.as_dict()},
                Cron(cron).next_epoch(now, fanout.tz),
            )
            for kind, cron in (("collect", fanout.collect_at), ("assemble", fanout.assemble_at))
        ]

    def add_schedule(
        self, tenant_id: str, name: str, cron: str, agent: str, *, message: str = "", seed: int = 0, tz: str = "UTC"
    ) -> str:
        """A recurring request, such as a check of the line every 15 minutes: a heartbeat."""
        if not self.agent(agent).first_message(message, seed):
            raise ValueError(f"'{agent}' has no standing task; give the schedule a message.")
        action = {"type": "bootstrap", "name": name, "agent": agent, "message": message, "seed": seed, "tz": tz}
        return self.store.add_schedule(tenant_id, name, cron, action, Cron(cron).next_epoch(self.store.clock(), tz))

    def _fire(self, schedule: dict[str, Any], now: float) -> None:
        action, tenant = schedule["action"], schedule["tenant_id"]
        name = action["name"]
        if action["type"] == "bootstrap":
            self.start_session(
                tenant, action["agent"], text=action["message"], title=name, seed=action["seed"], origin="schedule"
            )
        elif action["type"] == "collect":
            note = self._process_text(action["process"], "collect")
            for collector in action["collectors"]:
                self.start_session(
                    tenant,
                    collector,
                    text=note,
                    title=f"{name}: {collector}",
                    seed=action["seed"],
                    origin="schedule",
                    group_key=f"{name}@{int(now)}",
                )
        elif action["type"] == "assemble":
            group = self.store.latest_group(tenant, f"{name}@")
            reports = self.store.group_runs(tenant, group) if group else []
            self.start_session(
                tenant,
                action["assembler"],
                text=folder_message(name, action["collectors"], reports),
                title=f"{name}: briefing",
                origin="schedule",
            )
        self.store.reschedule(schedule["id"], Cron(schedule["cron"]).next_epoch(now, action["tz"]))

    def _process_text(self, process: str, step: str) -> str:
        name = f"{process}/{step}"
        return self.processes.read(name) if process and name in self.processes.entries() else ""

    # ----- scheduling -----------------------------------------------------------------------
    def tick(self) -> dict[str, list[str]]:
        """One scheduler pass, always in this order."""
        delivered = self.store.deliver_child_results()
        resumed = self.store.resume_ready()
        recovered = self.store.recover(self.config.max_retries)
        promoted = self.store.promote_queued()
        now = self.store.clock()
        fired = []
        for schedule in self.store.due_schedules(now):
            self._fire(schedule, now)
            fired.append(schedule["id"])
        return {
            "delivered": delivered,
            "resumed": resumed,
            "recovered": recovered,
            "promoted": promoted,
            "fired": fired,
        }

    def worker(self, name: str | None = None) -> Worker:
        return Worker(self, name or new_id("wrk"))

    def run_until_idle(self, *, workers: int = 1, max_rounds: int = 1000) -> int:
        """Ticks and workers in this thread until nothing moves. For tests and one-shot runs."""
        pool = [self.worker(f"worker-{i}") for i in range(workers)]
        steps = 0
        for _ in range(max_rounds):
            moved = any(self.tick().values())
            worked = False
            for w in pool:
                if w.work_once():
                    worked = True
                    steps += 1
            if not moved and not worked:
                return steps
        raise RuntimeError(f"Still busy after {max_rounds} rounds.")

    def start(self, *, workers: int = 2, tick_every: float = 0.5) -> None:
        """Background threads: one scheduler and ``workers`` workers, until ``stop``."""
        self._stopping.clear()

        def schedule() -> None:
            while not self._stopping.wait(tick_every):
                self.tick()

        def work(w: Worker) -> None:
            while not self._stopping.is_set():
                if w.work_once() is None:
                    self._stopping.wait(tick_every)

        self._threads = [threading.Thread(target=schedule, name="scheduler", daemon=True)]
        self._threads += [
            threading.Thread(target=work, args=(self.worker(f"worker-{i}"),), name=f"worker-{i}", daemon=True)
            for i in range(workers)
        ]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []

    # ----- one step of a run ----------------------------------------------------------------
    def _state(self, run: dict[str, Any], agent: RuntimeAgent) -> _State:
        model = self.gateway.bind(
            tenant_id=run["tenant_id"],
            run_id=run["id"],
            route=agent.route,
            on_event=lambda kind, data: self.store.emit(run, kind, data),
        )
        ctx = RunContext(
            tracer=_RunTracer(self.store, run),
            agent=AgentConfig(
                max_steps=agent.max_steps,
                max_tool_calls=agent.max_tool_calls,
                max_output_tokens=agent.max_output_tokens,
            ),
        )
        thread = self.store.thread(run["thread_id"])
        env = self.worlds.get(
            run["thread_id"],
            agent,
            thread["seed"],
            model,
            on_rebuilt=lambda n: self.store.emit(run, "world_rebuilt", {"replayed_calls": n}),
        )
        toolbox = agent.tools(env, model, ctx) if agent.tools is not None else Toolbox([])
        system = agent.system
        if agent.processes and self.processes.entries():
            toolbox = toolbox.extended([self.processes.tool()])
            system += "\n\nProcess files (read one with read_process when it applies):\n" + self.processes.listing()
        specs = toolbox.specs()
        if agent.ask_human:
            specs.append(ask_human_spec())
        if agent.delegates:
            specs.append(delegate_spec(self.agents, agent.delegates))
        return _State(model, env, toolbox, specs, system)

    def _step(self, run: dict[str, Any], worker: str) -> str:
        agent = self.agent(run["agent"])
        items = self.store.history(run["thread_id"])
        state = self._state(run, agent)
        calls = unanswered(items)
        if calls:
            return self._execute(run, worker, agent, state, calls)

        turns = sum(1 for i in items if i["kind"] == "assistant" and i["run_id"] == run["id"])
        if turns >= agent.max_steps:
            self.store.finish(
                run["id"], worker, "failed", error=f"Step budget used up ({agent.max_steps} model turns)."
            )
            return "failed"
        sink = _TextSink(self.store, run, self.config.stream_flush_chars)
        turn = state.model.complete_stream(
            system=state.system,
            messages=to_messages(items),
            tools=state.specs,
            on_text=sink,
            max_output_tokens=agent.max_output_tokens,
        )
        sink.flush()
        self.store.record_turn(run["id"], worker, turn_data(turn))
        self.store.emit(
            run,
            "generation",
            {
                "model": turn.model,
                "input_tokens": turn.usage.input_tokens,
                "output_tokens": turn.usage.output_tokens,
                "stop_reason": turn.stop_reason,
                "text": preview(turn.text),
                "tool_calls": [c.name for c in turn.tool_calls],
            },
        )
        if turn.tool_calls:
            return "continue"
        result = {"text": turn.text, "verified": agent.verified(state.env), "stop_reason": turn.stop_reason}
        if turn.stop_reason == "refusal":
            self.store.finish(run["id"], worker, "failed", result=result, error="The model refused.")
            return "failed"
        self.store.finish(run["id"], worker, "completed", result=result)
        return "completed"

    def _execute(
        self, run: dict[str, Any], worker: str, agent: RuntimeAgent, state: _State, calls: list[ToolCall]
    ) -> str:
        questions: list[Dependency] = []
        children: list[ChildRun] = []
        last: tuple[ToolCall, str] | None = None
        for call in calls:
            if call.name == ASK_HUMAN and agent.ask_human:
                question = str(call.arguments.get("question", "")).strip()
                problem = call.parse_error or ("" if question else "Write the question.")
                if problem:
                    self._refuse(run, worker, call, problem)
                else:
                    questions.append(Dependency(call.id, question))
            elif call.name == DELEGATE and agent.delegates:
                target = str(call.arguments.get("agent", ""))
                task = str(call.arguments.get("task", "")).strip()
                problem = call.parse_error or self._delegation_problem(run, agent, target, task)
                if problem:
                    self._refuse(run, worker, call, problem)
                else:
                    child = self.agent(target)
                    seed = self.store.thread(run["thread_id"])["seed"]
                    children.append(ChildRun(call.id, target, self._revision(child), child.first_message(task, seed)))
            else:
                last = (call, self._run_tool(run, worker, state, call))
        if questions or children:
            self.store.wait(run["id"], worker, questions, children)
            return "waiting"
        if agent.verified(state.env):
            answer = {"tool": last[0].name, "arguments": last[0].arguments, "result": last[1]} if last else None
            result = {"text": "", "verified": True, "stop_reason": "goal_reached", "answer": answer}
            self.store.finish(run["id"], worker, "completed", result=result)
            return "completed"
        if self.store.count_tool_items(run["id"]) >= agent.max_tool_calls:
            self.store.finish(run["id"], worker, "failed", error=f"Tool budget used up ({agent.max_tool_calls} calls).")
            return "failed"
        return "continue"

    def _delegation_problem(self, run: dict[str, Any], agent: RuntimeAgent, target: str, task: str) -> str:
        if target not in agent.delegates or target not in self.agents:
            return f"Unknown agent '{target}'. Choose one of: {', '.join(agent.delegates)}."
        if not task:
            return "Describe the task; the specialist cannot see this conversation."
        if run["depth"] + 1 > self.config.max_depth:
            return f"Delegation deeper than {self.config.max_depth} levels is not allowed."
        return ""

    def _refuse(self, run: dict[str, Any], worker: str, call: ToolCall, message: str) -> None:
        content = json.dumps({"error": message}, ensure_ascii=False)
        self.store.finish_execution(run["id"], worker, call.id, call.name, content, True)
        self.store.emit(run, "tool", {"name": call.name, "is_error": True, "result": preview(content, 400)})

    def _run_tool(self, run: dict[str, Any], worker: str, state: _State, call: ToolCall) -> str:
        before = self.store.start_execution(run["id"], worker, call.id, call.name, call.arguments)
        if before:
            self.store.emit(
                run,
                "tool_retry",
                {
                    "tool": call.name,
                    "call_id": call.id,
                    "note": "A previous worker may have run this call; its id goes to the tool as the idempotency key.",
                },
            )
        token = idempotency_key.set(call.id)
        try:
            result = state.toolbox.execute(call)
        finally:
            idempotency_key.reset(token)
        self.store.finish_execution(run["id"], worker, call.id, call.name, result.content, result.is_error)
        self.store.emit(
            run,
            "tool",
            {
                "name": call.name,
                "is_error": result.is_error,
                "duration_ms": round(result.duration_ms, 1),
                "result": preview(result.content, 400),
            },
        )
        return result.content


def folder_message(name: str, collectors: Sequence[str], reports: Sequence[dict[str, Any]]) -> str:
    """What the assembler receives: every finished report, and the collectors still missing."""
    started = min((r["created_at"] for r in reports), default=None)
    when = datetime.fromtimestamp(started, UTC).strftime("%Y-%m-%d %H:%M UTC") if started else "never"
    lines = [f"Reports collected for '{name}' (collection started {when}).", ""]
    finished = {r["agent"]: r for r in reports if r["status"] == "completed"}
    for collector in collectors:
        report = finished.get(collector)
        if report is None:
            continue
        result = report["result"] or {}
        verdict = result.get("verified")
        verified = (
            "no check" if verdict is None else ("verified by its check" if verdict else "not verified by its check")
        )
        lines.append(f"## {collector}: completed, {verified}")
        answer = result.get("answer")
        if answer:
            lines.append(f"Submitted {answer['tool']} with {json.dumps(answer['arguments'], ensure_ascii=False)}")
            lines.append(f"Check: {answer['result']}")
        if result.get("text"):
            lines.append(result["text"])
        lines.append("")
    states = {r["agent"]: r["status"] for r in reports}
    missing = [f"{c} ({states.get(c, 'not started')})" for c in collectors if c not in finished]
    lines.append("Missing at assembly time: " + (", ".join(missing) if missing else "none."))
    return "\n".join(lines)


class _Heartbeat(threading.Thread):
    """Extends the claim while a long step (a slow model call) is running."""

    def __init__(self, store: Store, run_id: str, worker: str, ttl: float, every: float) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{run_id}")
        self._store, self._run_id, self._worker, self._ttl, self._every = store, run_id, worker, ttl, every
        self._stopped = threading.Event()

    def run(self) -> None:
        while not self._stopped.wait(self._every):
            try:
                self._store.heartbeat(self._run_id, self._worker, self._ttl)
            except ClaimLost:
                return

    def stop(self) -> None:
        self._stopped.set()


class Worker:
    def __init__(self, runtime: Runtime, name: str) -> None:
        self.runtime = runtime
        self.name = name

    def work_once(self) -> str | None:
        """Claim one run and advance it until it finishes, waits or yields. Returns its id."""
        runtime, config, store = self.runtime, self.runtime.config, self.runtime.store
        run = store.claim(self.name, config.claim_ttl)
        if run is None:
            return None
        beat = None
        if config.heartbeat_every:
            beat = _Heartbeat(store, run["id"], self.name, config.claim_ttl, config.heartbeat_every)
            beat.start()
        try:
            for _ in range(config.yield_after_steps):
                if runtime._step(run, self.name) != "continue":
                    break
                store.heartbeat(run["id"], self.name, config.claim_ttl)
            else:
                store.release(run["id"], self.name)
        except ClaimLost:
            pass  # recovered, cancelled or finished elsewhere: this worker's step is void
        except ModelUnavailable as exc:
            _quietly(
                store.postpone,
                run["id"],
                self.name,
                delay=config.retry_delay,
                reason=str(exc),
                max_retries=config.max_retries,
            )
        except (ModelError, ReplayDiverged) as exc:
            _quietly(store.finish, run["id"], self.name, "failed", error=str(exc))
        except Exception as exc:  # a bug fails the run visibly instead of killing the worker
            _quietly(
                store.finish, run["id"], self.name, "failed", error=f"Internal error ({type(exc).__name__}): {exc}"
            )
        finally:
            if beat is not None:
                beat.stop()
        return str(run["id"])


def _quietly(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    with contextlib.suppress(ClaimLost):
        fn(*args, **kwargs)
