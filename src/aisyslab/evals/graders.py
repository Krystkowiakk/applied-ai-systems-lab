"""Graders: a mix of exact checks and a model judge, each returning a named grade.

Deterministic graders check what code can check (valid JSON, the right code, the right
priority). The judge grades only what code cannot: whether the free-text reason is faithful
to the ticket. Mixing them keeps the expensive, noisy grader on the narrowest question.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .judge import LLMJudge
from .triage import CODES, Ticket


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    note: str = ""


Grader = Callable[[Ticket, str], Grade]


def parse(output: str) -> dict[str, Any] | None:
    try:
        value = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def schema(ticket: Ticket, output: str) -> Grade:
    data = parse(output)
    if data is None:
        return Grade("schema", False, "not a JSON object")
    problems = []
    if data.get("code") not in CODES:
        problems.append(f"unknown code {data.get('code')!r}")
    if data.get("priority") not in (1, 2, 3):
        problems.append(f"priority {data.get('priority')!r}")
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        problems.append("no reason")
    return Grade("schema", not problems, "; ".join(problems))


def code(ticket: Ticket, output: str) -> Grade:
    data = parse(output) or {}
    got = data.get("code")
    return Grade("code", got == ticket.code, "" if got == ticket.code else f"expected {ticket.code}, got {got}")


def priority(ticket: Ticket, output: str) -> Grade:
    data = parse(output) or {}
    got = data.get("priority")
    return Grade(
        "priority", got == ticket.priority, "" if got == ticket.priority else f"expected {ticket.priority}, got {got}"
    )


def judged_reason(judge: LLMJudge) -> Grader:
    def grade(ticket: Ticket, output: str) -> Grade:
        data = parse(output) or {}
        passed, note = judge.grade(ticket.text, str(data.get("reason", "")))
        return Grade("reason", passed, note)

    return grade


DETERMINISTIC: list[Grader] = [schema, code, priority]
