"""Triage at scale and without a prompt engineer: batches under a schema, an interview, fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aisyslab.core.messages import Usage
from aisyslab.core.model import ModelError
from aisyslab.evals.batch import RuleFollowerBatch, batch_schema, classify_batches
from aisyslab.evals.finetune import (
    NearestTicketModel,
    SimulatedTuning,
    answer_for,
    examples,
    per_thousand,
    run_job,
    validate,
)
from aisyslab.evals.interview import NotesOwner, RuleDrafter, ScriptedInterviewer, draft_prompt, interview
from aisyslab.evals.runner import run_eval
from aisyslab.evals.triage import BASE_PROMPT, CODES, PROMPT_HEADER, RuleFollowerModel, Ticket, generate_tickets

TICKETS = generate_tickets(0)
SPLIT = {s: [t for t in TICKETS if t.split == s] for s in ("train", "dev", "holdout")}


def test_a_batch_answers_every_ticket_once_under_the_schema() -> None:
    schema = batch_schema(["T-1", "T-2"])
    item = schema["properties"]["items"]["items"]
    assert item["properties"]["id"]["enum"] == ["T-1", "T-2"] and item["properties"]["code"]["enum"] == list(CODES)
    assert item["additionalProperties"] is False and item["required"] == ["id", "code", "priority", "reason"]
    chosen = SPLIT["dev"] + SPLIT["holdout"]
    single = run_eval(RuleFollowerModel(), BASE_PROMPT, chosen)
    batched = classify_batches(RuleFollowerBatch(drop=1), BASE_PROMPT, chosen, size=12)
    assert batched.calls == 6 and batched.missing_first == 3  # three batches, each skipped one ticket
    assert [i.passed("code") for i in batched.report.items] == [i.passed("code") for i in single.items]
    assert batched.report.usage.input_tokens < single.usage.input_tokens  # the prompt is sent once per batch


class Sloppy:
    """Answers one ticket twice, invents an id and leaves the rest out, every time."""

    name = "sloppy"

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], schema_name: str) -> tuple[str, Usage]:
        first = json.loads(user)["tickets"][0]["id"]
        item = {"code": "MECH-BRG", "priority": 2, "reason": "r"}
        items = [{"id": first} | item, {"id": first} | item, {"id": "T-999"} | item]
        return json.dumps({"items": items}), Usage(10, 10)


def test_what_a_schema_cannot_promise_is_checked_in_code() -> None:
    batched = classify_batches(Sloppy(), BASE_PROMPT, SPLIT["dev"][:4], size=4)
    # The first answer covers one ticket, the second asks again for the three missing, and covers one.
    assert batched.calls == 2 and batched.missing_first == 3 and batched.repeated == 4
    outputs = [i.output for i in batched.report.items]
    assert sum(bool(o) for o in outputs) == 2 and not batched.report.items[2].passed("schema")


def test_an_interview_becomes_a_prompt_with_the_fixed_parts_kept() -> None:
    transcript, _usage = interview(ScriptedInterviewer(), NotesOwner(), questions=6)
    assert len(transcript) == 6 and transcript[0][0] == "How do operators describe a bearing problem?"
    prompt, _ = draft_prompt(RuleDrafter(), transcript)
    assert prompt.startswith(PROMPT_HEADER) and prompt.endswith("- if nothing else applies -> MECH-BRG\n")
    assert "- if the ticket mentions 'sounds like gravel' -> MECH-BRG" in prompt
    before = run_eval(RuleFollowerModel(), BASE_PROMPT, SPLIT["holdout"]).rate("code")
    after = run_eval(RuleFollowerModel(), prompt, SPLIT["holdout"]).rate("code")
    assert (before, after) == (0.167, 0.833)  # stand-ins: the pipeline works, not a measure of any model


def test_training_data_is_validated_before_anything_is_paid_for() -> None:
    check = validate(SPLIT["train"], SPLIT["dev"], SPLIT["holdout"], epochs=3)
    assert check["ok"] and check["leaks"] == [] and set(check["per_code"].values()) == {9}
    assert check["training_tokens"] == 3 * check["tokens_per_epoch"]
    leaked = validate([*SPLIT["train"], SPLIT["holdout"][0]], SPLIT["dev"], SPLIT["holdout"])
    assert not leaked["ok"] and leaked["leaks"] == [SPLIT["holdout"][0].id]
    first = examples(SPLIT["train"])[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    urgent = Ticket("T", "press 4: hissing at the pneumatic line. line stopped.", "PNEU-PRS", 1, "train")
    assert (
        json.loads(answer_for(urgent))["reason"]
        == "press 4: hissing at the pneumatic line; the ticket says line stopped."
    )


def test_a_restarted_run_resumes_the_same_job(tmp_path: Path) -> None:
    backend = SimulatedTuning(polls_until_done=3)
    journal = tmp_path / "journal.json"
    kwargs = {"train": SPLIT["train"], "dev": SPLIT["dev"], "model": "base", "epochs": 3, "workdir": tmp_path}
    clock = iter(range(100)).__next__
    first = run_job(backend, journal, max_wait_s=0, sleep=lambda _s: None, clock=lambda: float(clock()), **kwargs)
    assert first["status"] == "running" and backend.created == 1 and len(backend.uploads) == 2
    again = run_job(backend, journal, sleep=lambda _s: None, **kwargs)
    assert again["status"] == "succeeded" and backend.created == 1 and len(backend.uploads) == 2
    assert json.loads(journal.read_text())  # the job id was on disk before anyone waited
    other = run_job(backend, journal, sleep=lambda _s: None, **(kwargs | {"epochs": 2}))
    assert other["job"] == "ftjob-local-2"  # other settings, another job


def test_the_cost_per_thousand_and_the_tuned_stand_in() -> None:
    tuned = run_eval(NearestTicketModel(SPLIT["train"]), "x", SPLIT["holdout"])
    assert (tuned.rate("code"), tuned.rate("priority")) == (0.778, 1.0)  # unseen wordings cost it four tickets
    cost = per_thousand(tuned, (0.20, 0.80), training_tokens=14_000, training_price=1.50)
    expected = (tuned.usage.input_tokens * 0.20 + tuned.usage.output_tokens * 0.80) / 1e6 / 18 * 1000
    assert cost.per_thousand == round(expected, 4) and cost.training == 0.021


class Refusing(SimulatedTuning):
    """A provider that takes the files and refuses the job, as OpenAI does for new organisations since May 2026."""

    def create(self, *, training_file: str, validation_file: str, model: str, epochs: int, suffix: str) -> str:
        self.created += 1
        raise ModelError("PermissionDeniedError: Error code: 403 - training_not_available")


def test_a_refused_job_is_journaled_with_its_files_and_not_asked_again(tmp_path: Path) -> None:
    backend = Refusing()
    journal = tmp_path / "journal.json"
    kwargs = {"train": SPLIT["train"], "dev": SPLIT["dev"], "model": "base", "epochs": 3, "workdir": tmp_path}
    first = run_job(backend, journal, sleep=lambda _s: None, **kwargs)
    assert first["status"] == "refused" and "training_not_available" in first["error"]
    assert first["files"] == ["file-local-1", "file-local-2"]  # what was left behind is on record
    again = run_job(backend, journal, sleep=lambda _s: None, **kwargs)
    assert again["status"] == "refused" and backend.created == 1 and len(backend.uploads) == 2
