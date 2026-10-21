"""Run steps as a dependency graph: each starts as soon as its inputs are ready.

Independent steps run in parallel threads; a step receives the results of the steps it
needs. This is the code-side answer to "orchestration": when the plan is known in advance, a
graph is cheaper, faster and more predictable than an agent deciding the order at run time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from ..core.trace import Tracer


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[dict[str, Any]], Any]
    needs: tuple[str, ...] = field(default_factory=tuple)


def check(steps: Sequence[Step]) -> list[str]:
    """Validate the graph and return a topological order; raise on unknown inputs or cycles."""
    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        raise ValueError("Step names must be unique.")
    known = set(names)
    for s in steps:
        missing = [n for n in s.needs if n not in known]
        if missing:
            raise ValueError(f"Step '{s.name}' needs unknown steps: {missing}")
    order: list[str] = []
    waiting = {s.name: set(s.needs) for s in steps}
    while waiting:
        ready = sorted(n for n, deps in waiting.items() if not deps)
        if not ready:
            raise ValueError(f"Cycle among steps: {sorted(waiting)}")
        for n in ready:
            order.append(n)
            del waiting[n]
        for deps in waiting.values():
            deps.difference_update(ready)
    return order


def run_dag(steps: Sequence[Step], *, max_workers: int = 4, tracer: Tracer | None = None) -> dict[str, Any]:
    check(steps)
    pending = {s.name: s for s in steps}
    results: dict[str, Any] = {}
    running: dict[Future[Any], Step] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dag") as pool:
        while pending or running:
            for step in [s for s in pending.values() if all(n in results for n in s.needs)]:
                del pending[step.name]
                if tracer:
                    tracer.event("dag_step", step=step.name, state="start", needs=list(step.needs))
                running[pool.submit(step.run, {n: results[n] for n in step.needs})] = step
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                results[step.name] = future.result()  # an exception in a step stops the graph
                if tracer:
                    tracer.event("dag_step", step=step.name, state="done")
    return results
