# Orchestration: several agents on one task

Code: [`src/aisyslab/orchestration/`](../src/aisyslab/orchestration/). Lab:
[`shift-briefing`](labs.md#shift-briefing).

## When the plan is known: a graph, not an agent

`run_dag` runs steps as a dependency graph. Each step starts as soon as its inputs are ready,
independent steps run in parallel threads, and a cycle or an unknown dependency is refused before
anything runs. The `shift-briefing` baseline is three investigations and a report:

```python
run_dag([
    Step("root_cause", root_cause),
    Step("blocked", blocked),
    Step("held", held),
    Step("briefing", briefing, needs=("root_cause", "blocked", "held")),
])
```

## When it is not: an orchestrator with specialists

`delegate_tool` lets an orchestrator agent hand a task to a specialist that runs as its own agent:

- **Separate context.** The specialist sees only the task the orchestrator wrote and its own
  assignment, never the orchestrator's conversation or the other specialists'.
- **Its own tools, budgets and verifier.** A specialist is an `AgentSpec` (prompt, task, tool
  factory) bound to its own environment, so it can check its work before reporting.
- **Reports through documents.** The specialist writes its findings to the shared workspace and
  ends with a one-line summary; the orchestrator's context grows by a reference, not a transcript.
- **Accounted and traced.** The specialist's model calls count toward the run's cost, and its
  events land in the same trace, tagged with its name (`Tracer.child`).

## The workspace is the bus

`Workspace` holds documents with a version and an author. Every write names the version it was
based on; a write based on a stale version is refused with a hint to re-read and merge
(optimistic concurrency), so two agents cannot silently overwrite each other. Paths are confined
to the workspace.

## What the briefing lab measures

The baseline answers all three questions in five API calls, in parallel. The agent has to split
the work, write precise tasks for agents that cannot see each other, and assemble their results.
Its verifier names the wrong section (`root_cause`, `blocked_work_orders`, `held_batches`), so the
orchestrator knows which specialist to send back.
