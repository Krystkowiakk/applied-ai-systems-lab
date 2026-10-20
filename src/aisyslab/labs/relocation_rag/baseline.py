"""Deterministic solver: classic search plus pattern extraction, no model.

This is the honest ceiling of code without a model on open questions: it retrieves well
but cannot read. It takes the first figure of the best section, cannot apply exceptions or
date arithmetic, misses a question asked in Polish, and only says "not covered" when
keyword overlap is low. Expect a partial score, not acceptance.
"""

from __future__ import annotations

import re

from ...core.lab import RunContext
from .env import RelocationRagEnv

NUMBER = re.compile(r"\d[\d,]*")
MIN_BM25 = 2.0


def solve(env: RelocationRagEnv, ctx: RunContext) -> None:
    answers = []
    for q in ctx.call(env.list_questions):
        result = ctx.call(env.search, q["question"], q["question"], 3)
        top = result["results"][0]
        if top["scores"]["bm25"] < MIN_BM25:
            answers.append(
                {"question_id": q["question_id"], "value": "not covered", "answer": "Not covered.", "citations": []}
            )
            continue
        text = ctx.call(env.read_section, top["section_id"])["text"]
        if q["question"].lower().startswith(("can ", "may ", "do ", "does ", "is ", "are ")):
            value = "no" if re.search(r"\b(not|cannot|may not)\b", text) else "yes"
        else:
            match = NUMBER.search(text)
            value = match.group(0) if match else "not covered"
        answers.append(
            {"question_id": q["question_id"], "value": value, "answer": value, "citations": [top["section_id"]]}
        )
    ctx.submit(env.submit, answers)
