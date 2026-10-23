"""Shift handover: a week of shift logs in, a handover memory that fits a budget, answers out.

The logs can be read once, shift by shift, in order. Then they are sealed and the questions
open: from then on the only sources are what was kept, the observations (searchable) and the
handover record (at most ``HANDOVER_BUDGET`` tokens, the part that goes into every prompt). The
verifier grades each answer and names the kind of mistake: a value that a later shift changed,
missing or extra machines, a suspicion taken as fact, or a guess where the logs say nothing.
"""

from __future__ import annotations

import re
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.retrieval import HybridIndex
from ...core.tokens import estimate_tokens
from ...core.tools import ToolError
from ...core.verdict import Verdict
from ...memory.observational import SCOPE, STATUSES, in_scope
from .logs import LABELS, build

HANDOVER_BUDGET = 400


def _tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalise(kind: str, value: Any) -> Any:
    if kind == "tags":
        items = value if isinstance(value, list) else re.split(r",| and ", str(value))
        return frozenset(_tag(str(v)) for v in items if _tag(str(v)))
    text = " ".join(str(value).strip().lower().split())
    if text in ("not recorded", "not in the logs", "unknown"):
        return "not recorded"
    if kind == "yesno":
        return text.split()[0].strip(".,") if text else text
    if kind == "number":
        return "".join(ch for ch in text if ch.isdigit())
    if kind == "shift":
        match = re.match(r"^(mon|tue|wed|thu|fri)[a-z]*\s+(morning|afternoon|night)", text)
        return f"{match.group(1)} {match.group(2)}" if match else text
    return _tag(text)


class ShiftHandoverEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        week = build(seed)
        self.logs, self.questions, self.facts = week.logs, week.questions, week.facts
        self.position = 0
        self.sealed = False
        self.kept: list[dict[str, str]] = []
        self.handover = ""

    def next_shift(self) -> dict[str, Any] | None:
        """The next shift's log, once. None after the last shift."""
        self._api("next_shift")
        if self.sealed:
            raise ToolError("The logs are sealed; answer from what you kept.")
        if self.position >= len(self.logs):
            return None
        index = self.position
        self.position += 1
        return {"shift": LABELS[index], "log": self.logs[index], "shifts_left": len(self.logs) - self.position}

    def keep(self, observations: list[dict[str, str]]) -> dict[str, int]:
        """Store observations; one that repeats a kept one word for word is not stored again."""
        self._api("keep")
        seen = {(o["scope"], _tag(o["text"])) for o in self.kept}
        repeats = 0
        for o in observations:
            scope, status = str(o.get("scope", "")).lower(), str(o.get("status", "")).lower()
            if not SCOPE.match(scope) or status not in STATUSES or not str(o.get("text", "")).strip():
                raise ToolError(f"Invalid observation {o!r}.", hint="scope, status and text are required.")
            if (scope, _tag(str(o["text"]))) in seen:
                repeats += 1
                continue
            seen.add((scope, _tag(str(o["text"]))))
            self.kept.append(
                {"when": str(o.get("when", "")), "scope": scope, "status": status, "text": str(o["text"])[:300]}
            )
        return {"kept": len(self.kept), "repeats_skipped": repeats}

    def write_handover(self, text: str) -> dict[str, int]:
        self._api("write_handover")
        size = estimate_tokens(text)
        if size > HANDOVER_BUDGET:
            raise ToolError(
                f"The handover record is {size} tokens; the budget is {HANDOVER_BUDGET}.",
                hint="Condense it: drop what nobody has to act on first.",
            )
        self.handover = text
        return {"tokens": size, "budget": HANDOVER_BUDGET}

    def seal(self) -> dict[str, Any]:
        self._api("seal")
        if self.position < len(self.logs):
            raise ToolError(f"{len(self.logs) - self.position} shifts have not been read yet.")
        self.sealed = True
        return {"sealed": True, "questions": len(self.questions)}

    def read_handover(self) -> dict[str, Any]:
        self._api("read_handover")
        scopes: dict[str, int] = {}
        for o in self.kept:
            scopes[o["scope"]] = scopes.get(o["scope"], 0) + 1
        return {"record": self.handover or "(empty)", "tokens": estimate_tokens(self.handover), "scopes_kept": scopes}

    def recall(self, query: str, scope: str | None = None, k: int = 5) -> dict[str, Any]:
        self._api("recall")
        if not self.kept:
            return {"observations": [], "hint": "Nothing kept yet."}
        pool = in_scope(self.kept, scope, key=lambda o: o["scope"])
        result: dict[str, Any] = {}
        if not pool:
            pool = list(self.kept)
            result["hint"] = f"Nothing kept under {scope!r}; searched every scope. read_handover lists the scopes."
        hits = HybridIndex([f"[{o['when']}] ({o['scope']}) {o['status']}: {o['text']}" for o in pool]).search(
            query, query, k=k
        )
        result["observations"] = [pool[int(h["index"])] for h in hits]
        return result

    def list_questions(self) -> list[dict[str, str]]:
        self._api("list_questions")
        if not self.sealed:
            raise ToolError("Questions open once the logs are sealed: the next shift asks after the week.")
        return [{"question_id": q.id, "question": q.text} for q in self.questions]

    def submit(self, answers: list[dict[str, Any]]) -> Verdict:
        self._api("submit")
        if not self.sealed:
            raise ToolError("Seal the logs before answering.")
        given = {a.get("question_id"): a.get("value") for a in answers}
        per_question: dict[str, str] = {}
        for q in self.questions:
            if q.id not in given:
                per_question[q.id] = "missing"
                continue
            got = normalise(q.kind, given[q.id])
            want = normalise("tags" if q.kind == "tags" else q.kind, list(q.value) if q.kind == "tags" else q.value)
            if got == want:
                per_question[q.id] = "ok"
            elif q.kind == "none":
                per_question[q.id] = "wrong: the logs do not say this; answer 'not recorded'"
            elif q.stale is not None and got == normalise(q.kind, q.stale):
                per_question[q.id] = "outdated: that was true before a later shift changed it"
            elif q.kind == "tags":
                missing, extra = len(want - got), len(got - want)
                per_question[q.id] = f"wrong: {missing} machine(s) missing, {extra} that should not be there"
            elif q.id == "c3-gearbox":
                per_question[q.id] = "wrong: check whether anyone actually confirmed it"
            else:
                per_question[q.id] = "wrong value"
        correct = sum(v == "ok" for v in per_question.values())
        accepted = correct == len(self.questions)
        feedback = (
            "All answers correct." if accepted else "; ".join(f"{k}: {v}" for k, v in per_question.items() if v != "ok")
        )
        details = {"score": round(correct / len(self.questions), 3), "per_question": per_question}
        return self._record(Verdict(accepted, "OK" if accepted else "PARTIAL", feedback, details))
