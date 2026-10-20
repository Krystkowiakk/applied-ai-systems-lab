"""Relocation assistant: answer from the handbook, cite the section, say when it is not there.

The verifier grades each answer on its normalised value and on whether a cited section
actually supports it. A value taken from the superseded edition is reported as outdated.
"""

from __future__ import annotations

from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.retrieval import HybridIndex
from ...core.tools import ToolError
from ...core.verdict import Verdict
from .corpus import build, normalise


class RelocationRagEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.sections, self.questions = build(seed)
        self._by_id = {s.id: s for s in self.sections}
        self.index = HybridIndex([f"{s.title}. {s.text}" for s in self.sections], options.embedder)

    def list_questions(self) -> list[dict[str, str]]:
        self._api("list_questions")
        return [{"question_id": q.id, "question": q.text} for q in self.questions]

    def search(self, keywords: str, question: str | None = None, k: int = 5) -> dict[str, Any]:
        self._api("search")
        if not keywords.strip():
            raise ToolError("Give at least one keyword.")
        hits: list[dict[str, Any]] = []
        for hit in self.index.search(keywords, question, k=min(max(k, 1), 8)):
            s = self.sections[int(hit["index"])]
            hits.append(
                {
                    "section_id": s.id,
                    "title": s.title,
                    "edition": s.edition,
                    "status": s.status,
                    "snippet": s.text[:160],
                    "scores": {key: hit[key] for key in ("rrf", "bm25", "cosine")},
                }
            )
        result: dict[str, Any] = {"results": hits}
        if all(h["scores"]["bm25"] == 0 for h in hits):
            result["hint"] = "No keyword matched; results are ranked by similarity only. The handbook is in English."
        return result

    def read_section(self, section_id: str) -> dict[str, str]:
        self._api("read_section")
        s = self._by_id.get(section_id)
        if s is None:
            raise ToolError(f"No section '{section_id}'.", hint="Use section_id values returned by search.")
        return {"section_id": s.id, "title": s.title, "edition": s.edition, "status": s.status, "text": s.text}

    def submit(self, answers: list[dict[str, Any]]) -> Verdict:
        self._api("submit")
        given = {a.get("question_id"): a for a in answers}
        per_question: dict[str, str] = {}
        for q in self.questions:
            a = given.get(q.id)
            if a is None:
                per_question[q.id] = "missing"
                continue
            value = normalise(str(a.get("value", "")))
            cites = {str(c) for c in a.get("citations", [])}
            if q.value == "not covered":
                if value == "not covered":
                    per_question[q.id] = "ok"
                else:
                    per_question[q.id] = (
                        "wrong: the handbook does not answer this; say 'not covered' instead of guessing"
                    )
            elif value == q.value and cites & set(q.cites):
                per_question[q.id] = "ok"
            elif value == q.value:
                per_question[q.id] = "value ok, but no cited section supports it"
            elif q.stale is not None and value == q.stale:
                per_question[q.id] = "outdated: that value comes from a superseded edition"
            else:
                per_question[q.id] = "wrong value"
        correct = sum(v == "ok" for v in per_question.values())
        score = round(correct / len(self.questions), 3)
        accepted = correct == len(self.questions)
        feedback = (
            "All answers correct and supported."
            if accepted
            else "; ".join(f"{k}: {v}" for k, v in per_question.items() if v != "ok")
        )
        return self._record(
            Verdict(accepted, "OK" if accepted else "PARTIAL", feedback, {"score": score, "per_question": per_question})
        )
