"""Fine-tuning compared with prompting, on the same holdout, with the cost per thousand tickets.

- ``examples``: the train split as chat examples: a one-line instruction, the ticket, the answer as
  JSON. The knowledge goes into the weights, so the instruction stays short. The dev split is the
  validation file; the holdout is never uploaded.
- ``validate``: before anything is paid for, every answer parses and has a known code, the codes are
  counted, no holdout ticket appears in the files, and the tokens are counted, so the training cost
  is known before the job starts.
- ``run_job``: upload, create, wait with a growing interval. The job id is journaled as soon as the
  job exists, under a key made of the data, the base model and the epochs: a run that is stopped
  and started again waits for the same job instead of paying for a second one.
- ``per_thousand``: inference for a thousand tickets from the measured tokens, and the training
  cost apart, since it is paid once.

``NearestTicketModel`` is the offline stand-in for a tuned model: it is trained on the examples in
the plainest way, by remembering them, and answers with the label of the most similar one. Like a
tuned model, it knows nothing the examples do not show.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.messages import Message, ModelTurn, ToolSpec, Usage
from ..core.model import ModelError
from ..core.tokens import estimate_tokens
from .graders import schema
from .runner import EvalReport, fingerprint
from .triage import URGENT, Ticket

TUNED_SYSTEM = "Triage the maintenance ticket. Answer with JSON: code, priority, reason."

# USD per million tokens: (input, output) for inference, and training. Checked on 24 October 2026.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "ft:gpt-4.1-nano": (0.20, 0.80),
}
TRAINING_PRICE = {"gpt-4.1-nano": 1.50}


def reason_for(ticket: Ticket) -> str:
    """The reason a planner would write: the reported symptom, and the urgency if the ticket states it."""
    symptom = next((part for part in ticket.text.rstrip(".").split(". ") if ":" in part), ticket.text)
    urgent = next((u for u in URGENT if u in ticket.text), None)
    return f"{symptom.strip()}" + (f"; the ticket says {urgent}." if urgent else ".")


def answer_for(ticket: Ticket) -> str:
    return json.dumps({"code": ticket.code, "priority": ticket.priority, "reason": reason_for(ticket)})


def examples(tickets: Sequence[Ticket]) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": TUNED_SYSTEM},
                {"role": "user", "content": t.text},
                {"role": "assistant", "content": answer_for(t)},
            ]
        }
        for t in tickets
    ]


def validate(
    train: Sequence[Ticket], dev: Sequence[Ticket], holdout: Sequence[Ticket], *, epochs: int = 3
) -> dict[str, Any]:
    problems = []
    for t in [*train, *dev]:
        grade = schema(t, answer_for(t))
        if not grade.passed:
            problems.append(f"{t.id}: {grade.note}")
    uploaded = {t.text for t in [*train, *dev]}
    leaks = sorted(t.id for t in holdout if t.text in uploaded)
    tokens = sum(estimate_tokens(m["content"]) + 4 for e in examples(train) for m in e["messages"])
    return {
        "train": len(train),
        "validation": len(dev),
        "per_code": dict(sorted(Counter(t.code for t in train).items())),
        "leaks": leaks,
        "problems": problems,
        "tokens_per_epoch": tokens,
        "training_tokens": tokens * epochs,
        "ok": not problems and not leaks and len(train) >= 10,
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


class TuningBackend(Protocol):
    def upload(self, path: Path) -> str: ...

    def create(self, *, training_file: str, validation_file: str, model: str, epochs: int, suffix: str) -> str: ...

    def status(self, job_id: str) -> dict[str, Any]: ...


class OpenAITuning:
    def __init__(self) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        self._openai: Any = openai
        self._client = openai.OpenAI(max_retries=3)

    def upload(self, path: Path) -> str:
        with path.open("rb") as handle:
            return str(self._call(self._client.files.create, file=handle, purpose="fine-tune").id)

    def create(self, *, training_file: str, validation_file: str, model: str, epochs: int, suffix: str) -> str:
        job = self._call(
            self._client.fine_tuning.jobs.create,
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            suffix=suffix,
            method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": epochs}}},
        )
        return str(job.id)

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._call(self._client.fine_tuning.jobs.retrieve, job_id)
        error = getattr(job, "error", None)
        return {
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "error": getattr(error, "message", None) if error else None,
        }

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc


class SimulatedTuning:
    """Offline: files and a job that succeeds after a few polls; counts what would have been paid for."""

    def __init__(self, polls_until_done: int = 2) -> None:
        self.polls_until_done = polls_until_done
        self.uploads: list[Path] = []
        self.created = 0
        self.polls = 0

    def upload(self, path: Path) -> str:
        self.uploads.append(path)
        return f"file-local-{len(self.uploads)}"

    def create(self, *, training_file: str, validation_file: str, model: str, epochs: int, suffix: str) -> str:
        self.created += 1
        return f"ftjob-local-{self.created}"

    def status(self, job_id: str) -> dict[str, Any]:
        self.polls += 1
        done = self.polls >= self.polls_until_done
        return {
            "status": "succeeded" if done else "running",
            "fine_tuned_model": "ft:nearest-ticket" if done else None,
            "trained_tokens": None,
            "error": None,
        }


def run_job(
    backend: TuningBackend,
    journal: Path,
    *,
    train: Sequence[Ticket],
    dev: Sequence[Ticket],
    model: str,
    epochs: int,
    workdir: Path,
    max_wait_s: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Start the job once per (data, model, epochs) and wait for it; a rerun resumes the same job.

    Every step is journaled before the next: the uploaded files (a rerun reuses them), the job
    (a rerun waits for it), or the provider's refusal (a rerun does not ask again)."""
    key = f"{fingerprint(train)}:{fingerprint(dev)}:{model}:{epochs}"
    entries = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else {}
    entry = entries.setdefault(key, {"model": model, "epochs": epochs, "status": "new"})
    if "files" not in entry:
        training = backend.upload(write_jsonl(workdir / "train.jsonl", examples(train)))
        validation = backend.upload(write_jsonl(workdir / "dev.jsonl", examples(dev)))
        entry.update(files=[training, validation], status="uploaded")
        _save(journal, entries)
    if entry["status"] == "uploaded":
        try:
            entry["job"] = backend.create(
                training_file=entry["files"][0],
                validation_file=entry["files"][1],
                model=model,
                epochs=epochs,
                suffix="triage",
            )
            entry["status"] = "created"
        except ModelError as exc:  # the client has already retried; what is left is a refusal
            entry.update(status="refused", error=str(exc))
        _save(journal, entries)  # before waiting: a crash from here on resumes this job
    started, delay = clock(), 5.0
    while entry["status"] not in ("succeeded", "failed", "cancelled", "refused"):
        entry.update(backend.status(entry["job"]))
        _save(journal, entries)
        if entry["status"] in ("succeeded", "failed", "cancelled") or clock() - started > max_wait_s:
            break
        sleep(delay)
        delay = min(delay * 2, 60.0)
    return entry


def _save(journal: Path, entries: dict[str, Any]) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Cost:
    per_thousand: float  # inference, USD per thousand tickets
    training: float  # USD, once


def per_thousand(
    report: EvalReport, price: tuple[float, float], *, training_tokens: int = 0, training_price: float = 0.0
) -> Cost:
    n = max(1, len(report.items))
    inference = (report.usage.input_tokens * price[0] + report.usage.output_tokens * price[1]) / 1e6 / n * 1000
    return Cost(round(inference, 4), round(training_tokens * training_price / 1e6, 4))


WORD = re.compile(r"[a-z]+")


class NearestTicketModel:
    """Offline stand-in for a tuned model, trained in the plainest way: by remembering the examples.

    The code is the label of the training ticket with the most words in common. Priority 1 comes
    from word pairs seen at least twice in priority-1 tickets and never in the others, so urgency is
    learned from the examples too, not from a list."""

    def __init__(self, train: Sequence[Ticket], name: str = "ft:nearest-ticket") -> None:
        self.memory = [(set(WORD.findall(t.text.lower())), t) for t in train]
        pairs = {p: Counter[str]() for p in (1, 2, 3)}
        for t in train:
            pairs[t.priority].update(set(_pairs(t.text)))
        self.urgent = {b for b, n in pairs[1].items() if n >= 2 and b not in pairs[2] and b not in pairs[3]}
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        text = messages[-1].content
        words = set(WORD.findall(text.lower()))
        _, nearest = max(self.memory, key=lambda m: (len(words & m[0]) / max(1, len(words | m[0])), m[1].id))
        priority = 1 if self.urgent & set(_pairs(text)) else 2
        reason = reason_for(Ticket("", text, nearest.code, priority, ""))
        answer = json.dumps({"code": nearest.code, "priority": priority, "reason": reason})
        usage = Usage(input_tokens=estimate_tokens(system + text), output_tokens=estimate_tokens(answer))
        return ModelTurn(text=answer, tool_calls=[], usage=usage, stop_reason="end_turn", model=self._name)


def _pairs(text: str) -> list[str]:
    words = WORD.findall(text.lower())
    return [f"{a} {b}" for a, b in itertools.pairwise(words)]
