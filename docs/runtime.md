# The maintenance request service: agents as a service

Code: [`src/aisyslab/runtime/`](../src/aisyslab/runtime/). CLI: `aisyslab serve`, `aisyslab tenant`.

The labs run one agent once, in one process. A service for a maintenance team has other
problems: requests arrive while others run, a run takes longer than an HTTP request, workers die,
people answer questions hours later, two plants share one deployment, a provider goes down, and
the morning briefing has to be ready at 05:00 without anyone asking. This module is that layer.

```bash
pip install -e ".[service,anthropic]"
aisyslab tenant plant-a --budget 500000        # prints a token once; only its hash is stored
aisyslab serve --provider anthropic            # API, workers and the web UI on http://127.0.0.1:8000/
aisyslab serve --demo                          # no model: an offline stand-in, to try the API and the UI
```

## Endpoints with a fixed shape

There is no open chat endpoint. Every request has a schema and goes to a named agent:

| Endpoint | What it does |
|---|---|
| `POST /v1/sessions` | Start a request with an agent; lab agents bring their standing task |
| `POST /v1/sessions/bootstrap` | The same, for an external trigger (a cron job, a webhook) |
| `POST /v1/threads/{id}/messages` | A follow-up message; while a run works, it waits in the thread's queue |
| `POST /v1/threads/{id}/branch` | Change an earlier message: a new thread that shares the history up to there |
| `GET /v1/threads/{id}/items` | The conversation as items: turns, tool calls, results |
| `GET /v1/runs/{id}` | Status, result, open questions, child runs, the agent's settings at the time |
| `GET /v1/runs/{id}/events` | Server-sent events, resumable with `Last-Event-ID` |
| `POST /v1/runs/{id}/answer` | Answer the question a waiting run asked |
| `POST /v1/runs/{id}/cancel` | Cancel a run and everything it started |
| `POST /v1/triage` | Classify a ticket: code, priority, reason, or a flagged rule-based fallback |
| `POST /v1/schedules` | A recurring request for one agent, e.g. a check of the line every 15 minutes |
| `POST /v1/fanouts` | Schedule agents that start together and an assembler that starts later |
| `GET /v1/usage` | Tokens used against the budget, per route and model |

Each tenant has its own token and sees only its own data. Another tenant's run answers 404,
exactly like a run that does not exist.

## A message is a job

`POST` returns as soon as the work is recorded. A worker claims the run, advances it one step
(a model turn, or the tool calls that turn asked for) and writes the step before the next one.

- **Claims expire.** A claim lasts `claim_ttl` seconds; a heartbeat thread extends it while a slow
  model call runs. When a worker dies, the claim runs out and the scheduler puts the run back in
  the queue. The next worker continues from the last written step.
- **Fencing.** Every write a worker makes for a run checks that the worker still holds the claim.
  A worker that froze, lost its claim and woke up again cannot overwrite the work of the worker
  that took over.
- **A journal of tool calls.** A call is journaled before it runs and its result after, together
  with the conversation item, in one transaction. A finished call is never run again. A call that
  started but has no result (the worker died in between) runs again, with its id as the
  idempotency key (`runtime.idempotency_key`), so a tool that changes an external system applies
  it once.
- **The world comes back from the journal.** The lab environments stand in for the plant's
  systems and live in the worker's process. When one is missing (a restart, another process, a new
  branch), it is rebuilt from the session's seed by running the thread's tool calls again, and
  every result must match the journal. A mismatch fails the run instead of continuing in a world
  its history does not describe. Tools that call a model (vision) or read the clock cannot be
  replayed this way; real systems keep their own state and do not need it.
- **Outages are postponed, not failed.** When every model on the route fails, the run goes back
  to the queue with a delay, up to `max_retries` times.

The scheduler works in a fixed order on every tick: results of child runs, runs that can resume,
runs whose worker died, queued messages, then timers.

## Waiting is a state

`ask_human` and `delegate_to_agent` do not block a worker. The run stores what it waits for and
leaves. An answer (`POST /answer`) or a finished child run resolves the dependency; the scheduler
appends the result to the conversation as the call's result and puts the run back in the queue.
Delegation nests like a call stack (up to `max_depth`): a child run has its own private thread,
agent, tools, budgets and check, and its events appear in the parent's event stream, tagged with
the child agent's name.

## Items, not messages; branches, not deletes

The conversation is stored as items: user messages, model turns with their tool calls, and tool
results, each with the run that wrote it. Nothing is edited or deleted. To change an earlier
message, the UI branches the thread at that point; the original stays as it was, and the
branch's world is rebuilt from the shared part of the journal.

## One gateway for every model call

`ModelGateway` is the only way the service calls a model:

- **Routes with fallbacks.** A route is a list of models tried in order (`--fallback-provider`).
  Text a failed model already streamed is dropped (`stream_reset`), so it does not run into the
  next model's answer.
- **A circuit breaker per model.** After `failure_threshold` failures in a row a model is skipped
  for `cooldown` seconds; then one trial call decides whether it comes back.
- **Tenant budgets.** Every call's tokens count against the tenant's budget; an exhausted budget
  fails the run with a clear reason.
- **A moderation hook.** Input can be checked before it reaches any provider; a blocked request
  costs nothing and says why.
- **A response cache** for routes where it is safe (classification): an identical request from the
  same tenant is answered from the cache. The cache is never shared between tenants.
- **A call log** with model, tokens, latency and errors, which `/v1/usage` summarises.
- **Reasoning stays with its model.** Provider state (reasoning blocks) is tagged with the model
  that produced it and replayed only to that model.

Streaming goes through the same path: providers that stream hand their text over as it arrives,
and the runtime writes it to the event stream in chunks (`text_delta`).

## Agent settings are recorded with every run

Each run references a revision: the agent's prompt, budgets, delegates and the models on its route,
stored once per distinct configuration. When someone reports a bad answer, the run says exactly
which settings produced it.

## Triage that degrades visibly

`POST /v1/triage` asks the triage route for a code, a priority and a reason, and validates the
answer. If no model answers or the answer fails validation, a rule-based fallback answers
instead, `"degraded": true`, with the reason in `note`. Wording the rules do not know stays
unclassified instead of getting a guessed code.

## Grouped in time, not linked by dependencies

`FanOut` starts several agents together and an assembler later. The morning briefing:

```python
FanOut(
    name="morning-briefing",
    collectors=("line-incident", "plant-database", "quality-gate"),
    assembler="briefing-writer",
    collect_at="0 4 * * 1-5",   # 04:00 on weekdays
    assemble_at="0 5 * * 1-5",  # an hour of slack
    process="morning-briefing",
)
```

There are no edges between the jobs. A collector that is slow or failed cannot block the others
or the assembler; at 05:00 the assembler gets every finished report and the list of missing ones,
which the briefing names instead of filling in. A recurring request for a single agent (`POST /v1/schedules`), such as a
check of the line every 15 minutes, is the same mechanism with one job. The schedule is a
five-field cron expression evaluated on the wall clock of a time zone. A missed firing (the service was down) fires once and
moves on, without a catch-up storm.

## Process files read on demand

How the briefing is collected and assembled is written in
[`processes/morning-briefing/`](../src/aisyslab/runtime/processes/morning-briefing/): one Markdown
file per step. The front matter's description, which says when a file applies, goes into the
agent's prompt; the body, which says how, is read with `read_process` only when needed. The
scheduler passes the collect step to each collector in its first message.

## The web UI

`GET /` serves one page without a build step: start a request, follow its events as they stream,
answer questions, send messages that queue, branch from any earlier message, cancel, classify a
ticket, and see the tenant's token use. Model output is inserted as text, never as HTML.

## What is not here

- One SQLite file with WAL, and polling for new events. Several processes can share it on one
  machine; across machines this would be Postgres with `LISTEN/NOTIFY` or a queue.
- Accounts and workspaces inside a tenant, and roles. A tenant's token is all-powerful.
- A retention policy: events and items grow until deleted by hand.
