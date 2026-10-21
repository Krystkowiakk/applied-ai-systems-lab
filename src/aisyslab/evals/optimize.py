"""Automatic prompt optimisation, with a holdout that the loop never sees.

Round 0 measures the starting prompt. Each round shows a stronger optimiser model the current
prompt and the training cases it failed, and asks for an improved prompt. A candidate is kept
only if it scores higher on the training split; otherwise the round is recorded as rejected.
At the end both prompts are measured on the dev and holdout splits: the training score went up
by construction, so only the holdout says whether the prompt got better or just memorised.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.messages import Message, Usage
from ..core.model import ChatModel, ModelError
from .graders import parse
from .runner import EvalReport, run_eval
from .triage import Ticket

OPTIMIZER_SYSTEM = """You improve a classification prompt. You get the current prompt and cases it
got wrong (the ticket, the expected code, the code it gave). Return the complete improved prompt
and nothing else. Prefer general rules over copying the examples, and keep the answer format
section exactly as it is."""


@dataclass
class Round:
    number: int
    train: float
    kept: bool
    note: str = ""


@dataclass
class OptimizationResult:
    start_prompt: str
    best_prompt: str
    rounds: list[Round]
    final: dict[str, dict[str, float]]  # split -> {"start": .., "best": ..}
    usage: Usage = field(default_factory=Usage)


def _failures(report: EvalReport, tickets: Sequence[Ticket], limit: int) -> list[dict[str, str]]:
    by_id = {t.id: t for t in tickets}
    return [
        {"ticket": by_id[i.ticket_id].text, "expected": i.expected, "got": str((parse(i.output) or {}).get("code"))}
        for i in report.items
        if not i.passed("code")
    ][:limit]


def optimize_prompt(
    executor: ChatModel,
    optimizer: ChatModel,
    prompt: str,
    tickets: Sequence[Ticket],
    *,
    rounds: int = 5,
    max_failures: int = 8,
) -> OptimizationResult:
    split = {name: [t for t in tickets if t.split == name] for name in ("train", "dev", "holdout")}
    best_prompt = prompt
    best = run_eval(executor, prompt, split["train"])
    history = [Round(0, best.rate("code"), True, "starting prompt")]
    usage = best.usage
    for number in range(1, rounds + 1):
        failures = _failures(best, split["train"], max_failures)
        if not failures:
            history.append(Round(number, best.rate("code"), False, "no failures left on train"))
            break
        request = json.dumps({"prompt": best_prompt, "failures": failures}, ensure_ascii=False)
        try:
            turn = optimizer.complete(
                system=OPTIMIZER_SYSTEM,
                messages=[Message(role="user", content=request)],
                tools=[],
                max_output_tokens=4000,
            )
        except ModelError as exc:
            history.append(Round(number, best.rate("code"), False, f"optimiser failed: {exc}"))
            break
        usage = usage + turn.usage
        candidate = turn.text.strip()
        trial = run_eval(executor, candidate, split["train"])
        usage = usage + trial.usage
        kept = trial.rate("code") > best.rate("code")
        history.append(Round(number, trial.rate("code"), kept, "" if kept else "no gain on train, discarded"))
        if kept:
            best_prompt, best = candidate, trial
    final: dict[str, dict[str, float]] = {}
    for name in ("dev", "holdout"):
        start = run_eval(executor, prompt, split[name])
        end = run_eval(executor, best_prompt, split[name])
        usage = usage + start.usage + end.usage
        final[name] = {"start": start.rate("code"), "best": end.rate("code")}
    return OptimizationResult(prompt, best_prompt, history, final, usage)
