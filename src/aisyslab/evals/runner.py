"""Run an evaluation, compare two runs, and put several models side by side.

A run is identified by the prompt, the model and the dataset fingerprint, so two numbers are
only compared when they measure the same thing. ``compare`` lists regressions item by item:
an average that went up can still hide cases that broke.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from ..core.messages import Usage
from ..core.model import ChatModel, ModelError
from .graders import DETERMINISTIC, Grade, Grader, parse
from .triage import Ticket, classify


def fingerprint(tickets: Sequence[Ticket]) -> str:
    """Content hash of a dataset: the version that a score belongs to."""
    canonical = json.dumps([asdict(t) for t in tickets], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


@dataclass
class ItemResult:
    ticket_id: str
    expected: str
    output: str
    grades: list[Grade]
    error: str = ""  # the model call failed: the item counts as failed, the run goes on

    def passed(self, grader: str) -> bool:
        return next((g.passed for g in self.grades if g.name == grader), False)


@dataclass
class EvalReport:
    model: str
    prompt_id: str
    dataset: str
    items: list[ItemResult]
    usage: Usage = field(default_factory=Usage)

    def rate(self, grader: str) -> float:
        return round(sum(i.passed(grader) for i in self.items) / max(1, len(self.items)), 3)

    def has(self, grader: str) -> bool:
        return grader in self.graders()

    def graders(self) -> list[str]:
        return list(dict.fromkeys(g.name for i in self.items for g in i.grades))

    def confusion(self) -> dict[str, dict[str, int]]:
        table: dict[str, Counter[str]] = {}
        for item in self.items:
            got = str((parse(item.output) or {}).get("code"))
            table.setdefault(item.expected, Counter())[got] += 1
        return {expected: dict(row) for expected, row in sorted(table.items())}

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model,
            "prompt": self.prompt_id,
            "dataset": self.dataset,
            "n": len(self.items),
            **{g: self.rate(g) for g in self.graders()},
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }


def run_eval(
    model: ChatModel,
    prompt: str,
    tickets: Sequence[Ticket],
    graders: Sequence[Grader] = DETERMINISTIC,
    *,
    workers: int = 4,
) -> EvalReport:
    def one(ticket: Ticket) -> tuple[ItemResult, Usage]:
        try:
            output, usage = classify(model, prompt, ticket)
        except ModelError as exc:  # one failed call must not throw away a paid run
            return ItemResult(ticket.id, ticket.code, "", [g(ticket, "") for g in graders], str(exc)), Usage()
        return ItemResult(ticket.id, ticket.code, output, [g(ticket, output) for g in graders]), usage

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, tickets))  # map keeps the dataset order
    usage = Usage()
    for _, u in results:
        usage = usage + u
    return EvalReport(model.name, prompt_id(prompt), fingerprint(tickets), [r for r, _ in results], usage)


def compare(before: EvalReport, after: EvalReport, grader: str = "code") -> dict[str, object]:
    if before.dataset != after.dataset:
        raise ValueError("Runs on different datasets cannot be compared item by item.")
    was = {i.ticket_id: i.passed(grader) for i in before.items}
    now = {i.ticket_id: i.passed(grader) for i in after.items}
    return {
        "grader": grader,
        "before": before.rate(grader),
        "after": after.rate(grader),
        "fixed": sorted(t for t in now if now[t] and not was[t]),
        "regressed": sorted(t for t in now if was[t] and not now[t]),
    }


def to_markdown(reports: Sequence[EvalReport]) -> str:
    graders = list(dict.fromkeys(g for r in reports for g in r.graders()))
    head = "| Model | Prompt | Dataset | N | " + " | ".join(graders) + " | Tokens in | Tokens out |"
    rule = "|" + "---|" * (6 + len(graders))
    rows = [
        f"| {r.model} | {r.prompt_id} | {r.dataset} | {len(r.items)} | "
        + " | ".join(f"{r.rate(g):.2f}" if r.has(g) else "-" for g in graders)
        + f" | {r.usage.input_tokens:,} | {r.usage.output_tokens:,} |"
        for r in reports
    ]
    return "\n".join([head, rule, *rows])
