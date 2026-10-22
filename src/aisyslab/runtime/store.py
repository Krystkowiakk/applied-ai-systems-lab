"""Durable state of the service in SQLite: tenants, sessions, threads, items, runs and their journal.

Everything a run needs to continue lives here, so any worker can pick a run up: the
conversation as items, the journal of tool calls, what a waiting run waits for, and the claim
that says which worker owns the run until when. Every write a worker makes for a run first
checks that it still holds the claim (a fencing check), so a worker that lost its claim cannot
overwrite the work of the worker that recovered the run.

Nothing in a conversation is deleted or edited. A thread is append-only; changing an earlier
message means branching: a new thread that shares the history up to that point.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TERMINAL = ("completed", "failed", "cancelled")
# Claim order: runs that were waiting and can continue, recovered runs, child runs, new runs.
PRIORITY_NEW, PRIORITY_CHILD, PRIORITY_RECOVERED, PRIORITY_RESUMED = 0, 1, 2, 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE,
    token_budget INTEGER NOT NULL, tokens_used INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY, agent TEXT NOT NULL, digest TEXT NOT NULL, config TEXT NOT NULL,
    created_at REAL NOT NULL, UNIQUE (agent, digest));
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id), agent TEXT NOT NULL,
    title TEXT NOT NULL, seed INTEGER NOT NULL, origin TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), agent TEXT NOT NULL,
    parent_id TEXT REFERENCES threads(id), fork_seq INTEGER, private INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    thread_id TEXT NOT NULL REFERENCES threads(id), seq INTEGER NOT NULL, run_id TEXT,
    kind TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (thread_id, seq));
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, thread_id TEXT NOT NULL,
    agent TEXT NOT NULL, revision_id INTEGER NOT NULL, root_id TEXT NOT NULL, parent_id TEXT,
    depth INTEGER NOT NULL DEFAULT 0, input TEXT NOT NULL, status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0, retries INTEGER NOT NULL DEFAULT 0, claimed_by TEXT,
    claim_expires REAL, not_before REAL, group_key TEXT, result TEXT, error TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS runs_queue ON runs (status, priority, created_at);
CREATE INDEX IF NOT EXISTS runs_thread ON runs (thread_id, status);
CREATE TABLE IF NOT EXISTS dependencies (
    run_id TEXT NOT NULL, call_id TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT, question TEXT,
    answer TEXT, is_error INTEGER NOT NULL DEFAULT 0, resolved_at REAL,
    delivered INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (run_id, call_id));
CREATE TABLE IF NOT EXISTS executions (
    run_id TEXT NOT NULL, call_id TEXT NOT NULL, tool TEXT NOT NULL, arguments TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1, result TEXT, is_error INTEGER, started_at REAL NOT NULL,
    finished_at REAL, PRIMARY KEY (run_id, call_id));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, root_id TEXT NOT NULL, run_id TEXT NOT NULL,
    agent TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS events_root ON events (root_id, id);
CREATE INDEX IF NOT EXISTS events_run ON events (run_id, id);
CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, run_id TEXT, route TEXT NOT NULL,
    model TEXT NOT NULL, ok INTEGER NOT NULL, cached INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, latency_ms REAL NOT NULL,
    error TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT NOT NULL, cron TEXT NOT NULL,
    action TEXT NOT NULL, next_at REAL NOT NULL);
"""


class ClaimLost(Exception):
    """The worker no longer owns the run: it was recovered, cancelled or finished elsewhere."""


class NotFound(LookupError):
    """No such object for this tenant. Another tenant's objects look exactly the same."""


@dataclass(frozen=True)
class Dependency:
    """Something a waiting run waits for: a person's answer to a question."""

    call_id: str
    question: str


@dataclass(frozen=True)
class ChildRun:
    """A run started on behalf of a waiting run; its result comes back as that call's result."""

    call_id: str
    agent: str
    revision_id: int
    text: str


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _item(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "thread_id": row["thread_id"],
        "seq": row["seq"],
        "run_id": row["run_id"],
        "kind": row["kind"],
        "data": json.loads(row["data"]),
        "created_at": row["created_at"],
    }


def _run(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    run = dict(row)
    run["result"] = json.loads(run["result"]) if run.get("result") else None
    return run


class Store:
    def __init__(self, path: str | Path = ":memory:", *, clock: Callable[[], float] = time.time) -> None:
        self.path = str(path)
        self.clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. ``BEGIN IMMEDIATE`` takes the write lock first, so two
        workers, even in two processes, cannot claim the same run."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _all(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    def _one(self, sql: str, *args: Any) -> dict[str, Any] | None:
        rows = self._all(sql, *args)
        return rows[0] if rows else None

    # ----- tenants and agent revisions ------------------------------------------------------
    def create_tenant(self, name: str, token_budget: int = 500_000) -> tuple[str, str]:
        """A tenant and its API token. Only a hash of the token is stored; it is shown once."""
        tenant_id, token = new_id("ten"), "rt_" + secrets.token_urlsafe(24)
        with self.tx() as db:
            db.execute(
                "INSERT INTO tenants (id, name, token_hash, token_budget, created_at) VALUES (?, ?, ?, ?, ?)",
                (tenant_id, name, digest(token), token_budget, self.clock()),
            )
        return tenant_id, token

    def tenant_for_token(self, token: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM tenants WHERE token_hash = ?", digest(token))

    def tenant(self, tenant_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM tenants WHERE id = ?", tenant_id)
        if row is None:
            raise NotFound(tenant_id)
        return row

    def budget_left(self, tenant_id: str) -> int:
        tenant = self.tenant(tenant_id)
        return int(tenant["token_budget"]) - int(tenant["tokens_used"])

    def revision(self, agent: str, config: dict[str, Any]) -> int:
        """The id of this exact agent configuration, stored once and referenced by every run."""
        text = json.dumps(config, sort_keys=True, ensure_ascii=False)
        key = digest(text)[:16]
        with self.tx() as db:
            db.execute(
                "INSERT OR IGNORE INTO revisions (agent, digest, config, created_at) VALUES (?, ?, ?, ?)",
                (agent, key, text, self.clock()),
            )
            row = db.execute("SELECT id FROM revisions WHERE agent = ? AND digest = ?", (agent, key)).fetchone()
        return int(row["id"])

    def revision_config(self, revision_id: int) -> dict[str, Any]:
        row = self._one("SELECT * FROM revisions WHERE id = ?", revision_id)
        if row is None:
            raise NotFound(str(revision_id))
        return {"id": row["id"], "agent": row["agent"], "digest": row["digest"], **json.loads(row["config"])}

    # ----- sessions, threads and items ------------------------------------------------------
    def create_session(
        self, tenant_id: str, agent: str, *, title: str = "", seed: int = 0, origin: str = "api"
    ) -> tuple[str, str]:
        session_id, thread_id = new_id("ses"), new_id("thr")
        now = self.clock()
        with self.tx() as db:
            db.execute(
                "INSERT INTO sessions (id, tenant_id, agent, title, seed, origin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, tenant_id, agent, title or agent, seed, origin, now),
            )
            db.execute(
                "INSERT INTO threads (id, session_id, agent, private, created_at) VALUES (?, ?, ?, 0, ?)",
                (thread_id, session_id, agent, now),
            )
        return session_id, thread_id

    def sessions(self, tenant_id: str) -> list[dict[str, Any]]:
        return self._all(
            """SELECT s.*, (SELECT r.status FROM runs r WHERE r.session_id = s.id AND r.parent_id IS NULL
                           ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1) AS last_status
               FROM sessions s WHERE s.tenant_id = ? ORDER BY s.created_at DESC, s.rowid DESC""",
            tenant_id,
        )

    def session(self, tenant_id: str, session_id: str) -> dict[str, Any]:
        session = self._one("SELECT * FROM sessions WHERE id = ? AND tenant_id = ?", session_id, tenant_id)
        if session is None:
            raise NotFound(session_id)
        session["threads"] = self._all(
            "SELECT id, parent_id, fork_seq, created_at FROM threads WHERE session_id = ? AND private = 0"
            " ORDER BY created_at, rowid",
            session_id,
        )
        session["runs"] = self._all(
            "SELECT id, thread_id, agent, status, created_at FROM runs WHERE session_id = ? AND parent_id IS NULL"
            " ORDER BY created_at, rowid",
            session_id,
        )
        return session

    def thread(self, thread_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        row = self._one(
            "SELECT t.*, s.tenant_id, s.seed FROM threads t JOIN sessions s ON s.id = t.session_id WHERE t.id = ?",
            thread_id,
        )
        if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
            raise NotFound(thread_id)
        return row

    def branch(self, thread_id: str, keep_until: int) -> str:
        """A new thread sharing this one's history up to item ``keep_until``. Nothing is deleted."""
        with self.tx() as db:
            source = self._thread_row(db, thread_id)
            last = self._last_seq(db, source)
            if not 0 <= keep_until <= last:
                raise ValueError(f"keep_until must be between 0 and {last}.")
            new = new_id("thr")
            db.execute(
                "INSERT INTO threads (id, session_id, agent, parent_id, fork_seq, private, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (new, source["session_id"], source["agent"], thread_id, keep_until, source["private"], self.clock()),
            )
        return new

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's items, including those it shares with the threads it branched from."""
        with self._lock:
            return self._history(self._db, thread_id)

    def last_seq(self, thread_id: str) -> int:
        with self._lock:
            return self._last_seq(self._db, self._thread_row(self._db, thread_id))

    def _thread_row(self, db: sqlite3.Connection, thread_id: str) -> dict[str, Any]:
        row = db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row is None:
            raise NotFound(thread_id)
        return dict(row)

    def _history(self, db: sqlite3.Connection, thread_id: str) -> list[dict[str, Any]]:
        segments: list[tuple[str, int | None]] = []
        limit: int | None = None
        current: str | None = thread_id
        while current is not None:
            row = self._thread_row(db, current)
            segments.append((current, limit))
            if row["parent_id"] is not None:
                limit = row["fork_seq"] if limit is None else min(limit, row["fork_seq"])
            current = row["parent_id"]
        items: list[dict[str, Any]] = []
        for segment, upto in reversed(segments):
            rows = db.execute(
                "SELECT * FROM items WHERE thread_id = ? AND (? IS NULL OR seq <= ?) ORDER BY seq",
                (segment, upto, upto),
            )
            items.extend(_item(r) for r in rows)
        return items

    def _last_seq(self, db: sqlite3.Connection, thread: dict[str, Any]) -> int:
        own = db.execute("SELECT MAX(seq) FROM items WHERE thread_id = ?", (thread["id"],)).fetchone()[0]
        return int(own) if own is not None else int(thread["fork_seq"] or 0)

    def _append(self, db: sqlite3.Connection, thread_id: str, kind: str, data: Any, run_id: str | None) -> int:
        seq = self._last_seq(db, self._thread_row(db, thread_id)) + 1
        db.execute(
            "INSERT INTO items (thread_id, seq, run_id, kind, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, seq, run_id, kind, dumps(data), self.clock()),
        )
        return seq

    # ----- runs -----------------------------------------------------------------------------
    def create_run(
        self,
        *,
        tenant_id: str,
        session_id: str,
        thread_id: str,
        agent: str,
        revision_id: int,
        text: str,
        group_key: str | None = None,
    ) -> dict[str, Any]:
        """A run for a message. If the thread is busy, it waits in the thread's queue and its
        message joins the conversation only when its turn comes."""
        with self.tx() as db:
            return self._insert_run(
                db,
                tenant_id=tenant_id,
                session_id=session_id,
                thread_id=thread_id,
                agent=agent,
                revision_id=revision_id,
                text=text,
                parent=None,
                priority=PRIORITY_NEW,
                group_key=group_key,
            )

    def _insert_run(
        self,
        db: sqlite3.Connection,
        *,
        tenant_id: str,
        session_id: str,
        thread_id: str,
        agent: str,
        revision_id: int,
        text: str,
        parent: dict[str, Any] | None,
        priority: int,
        group_key: str | None,
    ) -> dict[str, Any]:
        run_id = new_id("run")
        busy = db.execute(
            "SELECT 1 FROM runs WHERE thread_id = ? AND status IN ('queued', 'pending', 'running', 'waiting') LIMIT 1",
            (thread_id,),
        ).fetchone()
        status = "queued" if busy else "pending"
        now = self.clock()
        db.execute(
            """INSERT INTO runs (id, tenant_id, session_id, thread_id, agent, revision_id, root_id, parent_id,
                                 depth, input, status, priority, group_key, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                tenant_id,
                session_id,
                thread_id,
                agent,
                revision_id,
                parent["root_id"] if parent else run_id,
                parent["id"] if parent else None,
                parent["depth"] + 1 if parent else 0,
                text,
                status,
                priority,
                group_key,
                now,
                now,
            ),
        )
        run = self._run_row(db, run_id)
        if status == "pending":
            self._append(db, thread_id, "user", {"text": text}, run_id)
        self._emit(
            db, run, "run_queued" if busy else "run_created", {"thread_id": thread_id, "parent": run["parent_id"]}
        )
        return run

    def _run_row(self, db: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFound(run_id)
        return _run(row)

    def run(self, run_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        row = self._one("SELECT * FROM runs WHERE id = ?", run_id)
        if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
            raise NotFound(run_id)
        return _run(row)

    def runs_for(self, *, root_id: str) -> list[dict[str, Any]]:
        return [_run(r) for r in self._all("SELECT * FROM runs WHERE root_id = ? ORDER BY created_at, rowid", root_id)]

    def count_tool_items(self, run_id: str) -> int:
        row = self._one("SELECT COUNT(*) AS n FROM items WHERE run_id = ? AND kind = 'tool'", run_id)
        return int(row["n"]) if row else 0

    def executions(self, run_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM executions WHERE run_id = ? ORDER BY started_at, rowid", run_id)

    def claim(self, worker: str, ttl: float) -> dict[str, Any] | None:
        """Take the next run off the queue for ``ttl`` seconds; a heartbeat extends the claim."""
        now = self.clock()
        with self.tx() as db:
            row = db.execute(
                """SELECT id FROM runs WHERE status = 'pending' AND (not_before IS NULL OR not_before <= ?)
                   ORDER BY priority DESC, created_at, rowid LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE runs SET status = 'running', claimed_by = ?, claim_expires = ?, updated_at = ? WHERE id = ?",
                (worker, now + ttl, now, row["id"]),
            )
            run = self._run_row(db, row["id"])
            self._emit(db, run, "run_started", {"worker": worker, "retries": run["retries"]})
            return run

    def _fence(self, db: sqlite3.Connection, run_id: str, worker: str) -> dict[str, Any]:
        row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row["status"] != "running" or row["claimed_by"] != worker:
            raise ClaimLost(run_id)
        return _run(row)

    def heartbeat(self, run_id: str, worker: str, ttl: float) -> None:
        with self.tx() as db:
            self._fence(db, run_id, worker)
            now = self.clock()
            db.execute("UPDATE runs SET claim_expires = ?, updated_at = ? WHERE id = ?", (now + ttl, now, run_id))

    def release(self, run_id: str, worker: str) -> None:
        """Give the run back to the queue between steps, so other runs get a turn."""
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            db.execute(
                "UPDATE runs SET status = 'pending', claimed_by = NULL, claim_expires = NULL, updated_at = ?"
                " WHERE id = ?",
                (self.clock(), run_id),
            )
            self._emit(db, run, "yielded", {"worker": worker})

    def postpone(self, run_id: str, worker: str, *, delay: float, reason: str, max_retries: int) -> bool:
        """Put the run back after an outage, to try again later; fail it when retries run out."""
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            if run["retries"] >= max_retries:
                self._finish(db, run, "failed", error=f"{reason} (gave up after {run['retries']} retries)")
                return False
            now = self.clock()
            db.execute(
                """UPDATE runs SET status = 'pending', claimed_by = NULL, claim_expires = NULL,
                   retries = retries + 1, not_before = ?, updated_at = ? WHERE id = ?""",
                (now + delay, now, run_id),
            )
            self._emit(db, run, "postponed", {"reason": reason, "retry_in_s": delay})
            return True

    def record_turn(self, run_id: str, worker: str, data: dict[str, Any]) -> int:
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            return self._append(db, run["thread_id"], "assistant", data, run_id)

    def start_execution(self, run_id: str, worker: str, call_id: str, tool: str, arguments: dict[str, Any]) -> int:
        """Journal a tool call before it runs. Returns how many times it was started before."""
        with self.tx() as db:
            self._fence(db, run_id, worker)
            row = db.execute(
                "SELECT attempts FROM executions WHERE run_id = ? AND call_id = ?", (run_id, call_id)
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO executions (run_id, call_id, tool, arguments, started_at) VALUES (?, ?, ?, ?, ?)",
                    (run_id, call_id, tool, dumps(arguments), self.clock()),
                )
                return 0
            db.execute(
                "UPDATE executions SET attempts = attempts + 1, started_at = ? WHERE run_id = ? AND call_id = ?",
                (self.clock(), run_id, call_id),
            )
            return int(row["attempts"])

    def finish_execution(self, run_id: str, worker: str, call_id: str, tool: str, content: str, is_error: bool) -> int:
        """Record a tool result in the journal and in the conversation, in one transaction."""
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            db.execute(
                "UPDATE executions SET result = ?, is_error = ?, finished_at = ? WHERE run_id = ? AND call_id = ?",
                (content, int(is_error), self.clock(), run_id, call_id),
            )
            data = {"call_id": call_id, "name": tool, "content": content, "is_error": is_error}
            return self._append(db, run["thread_id"], "tool", data, run_id)

    def finish(
        self, run_id: str, worker: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            self._finish(db, run, status, result=result, error=error)

    def _finish(
        self,
        db: sqlite3.Connection,
        run: dict[str, Any],
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        db.execute(
            """UPDATE runs SET status = ?, result = ?, error = ?, claimed_by = NULL, claim_expires = NULL,
               updated_at = ? WHERE id = ?""",
            (status, dumps(result) if result is not None else None, error, self.clock(), run["id"]),
        )
        self._emit(db, run, status, {"result": result, "error": error})
        self._promote(db, run["thread_id"])

    def wait(
        self, run_id: str, worker: str, questions: Sequence[Dependency], children: Sequence[ChildRun]
    ) -> list[str]:
        """The run waits for answers and child runs, created in the same transaction."""
        with self.tx() as db:
            run = self._fence(db, run_id, worker)
            child_ids: list[str] = []
            for child in children:
                thread_id = new_id("thr")
                db.execute(
                    "INSERT INTO threads (id, session_id, agent, private, created_at) VALUES (?, ?, ?, 1, ?)",
                    (thread_id, run["session_id"], child.agent, self.clock()),
                )
                created = self._insert_run(
                    db,
                    tenant_id=run["tenant_id"],
                    session_id=run["session_id"],
                    thread_id=thread_id,
                    agent=child.agent,
                    revision_id=child.revision_id,
                    text=child.text,
                    parent=run,
                    priority=PRIORITY_CHILD,
                    group_key=None,
                )
                db.execute(
                    "INSERT INTO dependencies (run_id, call_id, kind, ref, question) VALUES (?, ?, 'agent', ?, ?)",
                    (run_id, child.call_id, created["id"], child.text),
                )
                child_ids.append(created["id"])
            for question in questions:
                db.execute(
                    "INSERT INTO dependencies (run_id, call_id, kind, question) VALUES (?, ?, 'human', ?)",
                    (run_id, question.call_id, question.question),
                )
            db.execute(
                "UPDATE runs SET status = 'waiting', claimed_by = NULL, claim_expires = NULL, updated_at = ?"
                " WHERE id = ?",
                (self.clock(), run_id),
            )
            self._emit(db, run, "waiting", {"questions": [q.question for q in questions], "children": child_ids})
            return child_ids

    def pending_questions(self, run_id: str) -> list[dict[str, Any]]:
        return self._all(
            "SELECT call_id, question FROM dependencies WHERE run_id = ? AND kind = 'human' AND resolved_at IS NULL"
            " ORDER BY rowid",
            run_id,
        )

    def answer(self, run_id: str, tenant_id: str, text: str, call_id: str | None = None) -> str:
        """Record a person's answer; the scheduler resumes the run. Returns the call answered."""
        with self.tx() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None or row["tenant_id"] != tenant_id:
                raise NotFound(run_id)
            sql = "SELECT call_id FROM dependencies WHERE run_id = ? AND kind = 'human' AND resolved_at IS NULL"
            args: list[Any] = [run_id]
            if call_id:
                sql += " AND call_id = ?"
                args.append(call_id)
            open_question = db.execute(sql + " ORDER BY rowid LIMIT 1", args).fetchone()
            if open_question is None:
                raise ValueError("This run is not waiting for an answer.")
            db.execute(
                "UPDATE dependencies SET answer = ?, resolved_at = ? WHERE run_id = ? AND call_id = ?",
                (text, self.clock(), run_id, open_question["call_id"]),
            )
            self._emit(db, _run(row), "answered", {"call_id": open_question["call_id"]})
            return str(open_question["call_id"])

    def deliver_child_results(self) -> list[str]:
        """Resolve the dependencies of every finished child run. Returns the parents affected."""
        with self.tx() as db:
            rows = db.execute(
                """SELECT d.run_id, d.call_id, c.id AS child_id, c.agent, c.status, c.result, c.error
                   FROM dependencies d JOIN runs c ON c.id = d.ref
                   WHERE d.kind = 'agent' AND d.resolved_at IS NULL AND c.status IN ('completed', 'failed', 'cancelled')"""
            ).fetchall()
            for r in rows:
                answer = {
                    "agent": r["agent"],
                    "run_id": r["child_id"],
                    "status": r["status"],
                    "result": json.loads(r["result"]) if r["result"] else None,
                    "error": r["error"],
                }
                db.execute(
                    "UPDATE dependencies SET answer = ?, is_error = ?, resolved_at = ? WHERE run_id = ? AND call_id = ?",
                    (dumps(answer), int(r["status"] != "completed"), self.clock(), r["run_id"], r["call_id"]),
                )
                parent = self._run_row(db, r["run_id"])
                self._emit(
                    db, parent, "child_finished", {"child": r["child_id"], "agent": r["agent"], "status": r["status"]}
                )
            return [r["run_id"] for r in rows]

    def resume_ready(self) -> list[str]:
        """Waiting runs whose dependencies are all resolved get their answers and go back to the queue."""
        resumed: list[str] = []
        with self.tx() as db:
            for row in db.execute("SELECT * FROM runs WHERE status = 'waiting' ORDER BY updated_at, rowid").fetchall():
                deps = db.execute("SELECT * FROM dependencies WHERE run_id = ? ORDER BY rowid", (row["id"],)).fetchall()
                if any(d["resolved_at"] is None for d in deps):
                    continue
                for d in deps:
                    if d["delivered"]:
                        continue
                    name = "ask_human" if d["kind"] == "human" else "delegate_to_agent"
                    data = {
                        "call_id": d["call_id"],
                        "name": name,
                        "content": d["answer"],
                        "is_error": bool(d["is_error"]),
                    }
                    self._append(db, row["thread_id"], "tool", data, row["id"])
                    db.execute(
                        "UPDATE dependencies SET delivered = 1 WHERE run_id = ? AND call_id = ?",
                        (row["id"], d["call_id"]),
                    )
                db.execute(
                    "UPDATE runs SET status = 'pending', priority = ?, updated_at = ? WHERE id = ?",
                    (PRIORITY_RESUMED, self.clock(), row["id"]),
                )
                self._emit(db, _run(row), "resumed", {})
                resumed.append(row["id"])
        return resumed

    def recover(self, max_retries: int) -> list[str]:
        """Runs whose claim expired (the worker died or hung) go back to the queue, or fail."""
        now = self.clock()
        recovered: list[str] = []
        with self.tx() as db:
            for row in db.execute(
                "SELECT * FROM runs WHERE status = 'running' AND claim_expires < ?", (now,)
            ).fetchall():
                run = _run(row)
                if run["retries"] < max_retries:
                    db.execute(
                        """UPDATE runs SET status = 'pending', claimed_by = NULL, claim_expires = NULL,
                           retries = retries + 1, priority = MAX(priority, ?), updated_at = ? WHERE id = ?""",
                        (PRIORITY_RECOVERED, now, run["id"]),
                    )
                    self._emit(db, run, "recovered", {"worker": run["claimed_by"], "retry": run["retries"] + 1})
                else:
                    self._finish(
                        db, run, "failed", error=f"Abandoned: its worker stopped responding {run['retries'] + 1} times."
                    )
                recovered.append(run["id"])
        return recovered

    def promote_queued(self) -> list[str]:
        """Start the next queued message on every thread that has gone idle."""
        promoted: list[str] = []
        with self.tx() as db:
            threads = db.execute("SELECT DISTINCT thread_id FROM runs WHERE status = 'queued'").fetchall()
            for t in threads:
                started = self._promote(db, t["thread_id"])
                if started:
                    promoted.append(started)
        return promoted

    def _promote(self, db: sqlite3.Connection, thread_id: str) -> str | None:
        busy = db.execute(
            "SELECT 1 FROM runs WHERE thread_id = ? AND status IN ('pending', 'running', 'waiting') LIMIT 1",
            (thread_id,),
        ).fetchone()
        if busy:
            return None
        row = db.execute(
            "SELECT * FROM runs WHERE thread_id = ? AND status = 'queued' ORDER BY created_at, rowid LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        db.execute("UPDATE runs SET status = 'pending', updated_at = ? WHERE id = ?", (self.clock(), row["id"]))
        self._append(db, thread_id, "user", {"text": row["input"]}, row["id"])
        self._emit(db, _run(row), "run_dequeued", {})
        return str(row["id"])

    def cancel(self, run_id: str, tenant_id: str) -> bool:
        """Cancel a run and everything it started. A worker busy with it stops at its next write."""
        with self.tx() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None or row["tenant_id"] != tenant_id:
                raise NotFound(run_id)
            if row["status"] in TERMINAL:
                return False
            todo = [_run(row)]
            while todo:
                run = todo.pop()
                if run["status"] not in TERMINAL:
                    db.execute(
                        """UPDATE runs SET status = 'cancelled', claimed_by = NULL, claim_expires = NULL,
                           updated_at = ? WHERE id = ?""",
                        (self.clock(), run["id"]),
                    )
                    self._emit(db, run, "cancelled", {})
                todo.extend(_run(c) for c in db.execute("SELECT * FROM runs WHERE parent_id = ?", (run["id"],)))
            self._promote(db, row["thread_id"])
            return True

    def thread_runs(self, thread_id: str) -> list[dict[str, Any]]:
        return [
            _run(r) for r in self._all("SELECT * FROM runs WHERE thread_id = ? ORDER BY created_at, rowid", thread_id)
        ]

    # ----- events -----------------------------------------------------------------------------
    def _emit(self, db: sqlite3.Connection, run: dict[str, Any], kind: str, data: dict[str, Any]) -> int:
        cursor = db.execute(
            "INSERT INTO events (root_id, run_id, agent, kind, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run["root_id"], run["id"], run["agent"], kind, dumps(data), self.clock()),
        )
        return int(cursor.lastrowid or 0)

    def emit(self, run: dict[str, Any], kind: str, data: dict[str, Any]) -> int:
        with self.tx() as db:
            return self._emit(db, run, kind, data)

    def events(self, run_id: str, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        """Events of a run after a given event id. A root run's stream includes its child runs."""
        run = self.run(run_id)
        column = "root_id" if run["root_id"] == run_id else "run_id"
        rows = self._all(  # the column name is one of two constants, never user input
            f"SELECT * FROM events WHERE {column} = ? AND id > ? ORDER BY id LIMIT ?",
            run_id,
            after,
            limit,
        )
        return [{**r, "data": json.loads(r["data"])} for r in rows]

    # ----- model calls, usage and the response cache -----------------------------------------
    def log_model_call(
        self,
        *,
        tenant_id: str,
        run_id: str | None,
        route: str,
        model: str,
        ok: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        error: str | None = None,
        cached: bool = False,
    ) -> None:
        with self.tx() as db:
            db.execute(
                """INSERT INTO model_calls (tenant_id, run_id, route, model, ok, cached, input_tokens, output_tokens,
                                            latency_ms, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tenant_id,
                    run_id,
                    route,
                    model,
                    int(ok),
                    int(cached),
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    error,
                    self.clock(),
                ),
            )
            if ok and not cached:
                db.execute(
                    "UPDATE tenants SET tokens_used = tokens_used + ? WHERE id = ?",
                    (input_tokens + output_tokens, tenant_id),
                )

    def usage(self, tenant_id: str) -> dict[str, Any]:
        tenant = self.tenant(tenant_id)
        models = self._all(
            """SELECT route, model, COUNT(*) AS calls, SUM(1 - ok) AS failures, SUM(cached) AS cache_hits,
                      SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                      ROUND(AVG(latency_ms), 1) AS mean_latency_ms
               FROM model_calls WHERE tenant_id = ? GROUP BY route, model ORDER BY calls DESC""",
            tenant_id,
        )
        return {
            "tenant": tenant["name"],
            "token_budget": tenant["token_budget"],
            "tokens_used": tenant["tokens_used"],
            "models": models,
        }

    def model_calls(self, tenant_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM model_calls WHERE tenant_id = ? ORDER BY id", tenant_id)

    def cached_response(self, key: str) -> dict[str, Any] | None:
        row = self._one("SELECT value FROM response_cache WHERE key = ?", key)
        return json.loads(row["value"]) if row else None

    def cache_response(self, key: str, value: dict[str, Any]) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO response_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, dumps(value), self.clock()),
            )

    # ----- schedules ----------------------------------------------------------------------------
    def add_schedule(self, tenant_id: str, name: str, cron: str, action: dict[str, Any], next_at: float) -> str:
        schedule_id = new_id("sch")
        with self.tx() as db:
            db.execute(
                "INSERT INTO schedules (id, tenant_id, name, cron, action, next_at) VALUES (?, ?, ?, ?, ?, ?)",
                (schedule_id, tenant_id, name, cron, dumps(action), next_at),
            )
        return schedule_id

    def schedules(self, tenant_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM schedules WHERE tenant_id = ? ORDER BY next_at", tenant_id)
        return [{**r, "action": json.loads(r["action"])} for r in rows]

    def due_schedules(self, now: float) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM schedules WHERE next_at <= ? ORDER BY next_at, rowid", now)
        return [{**r, "action": json.loads(r["action"])} for r in rows]

    def reschedule(self, schedule_id: str, next_at: float) -> None:
        with self.tx() as db:
            db.execute("UPDATE schedules SET next_at = ? WHERE id = ?", (next_at, schedule_id))

    def latest_group(self, tenant_id: str, prefix: str) -> str | None:
        row = self._one(
            "SELECT group_key FROM runs WHERE tenant_id = ? AND group_key LIKE ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            tenant_id,
            prefix + "%",
        )
        return str(row["group_key"]) if row else None

    def group_runs(self, tenant_id: str, group_key: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM runs WHERE tenant_id = ? AND group_key = ? ORDER BY created_at, rowid", tenant_id, group_key
        )
        return [_run(r) for r in rows]
