"""Deterministic solver: three narrow workers write to a shared blackboard.

Each worker owns one fact, one fixed query and one extraction rule. The loop around them
re-runs the workers when the verifier reports a stale value, because the mailbox is live.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from ...core.lab import RunContext
from .env import DOMAIN, InboxEnv

HUMAN_DATE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\d{1,2}) "
    r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})"
)
VPN = re.compile(r"\bVX-\d{4}-[A-Z]{2}\b")
CAB = re.compile(r"(CHG-\d{6}) \(([^)]*)\) (?:is|was) (approved|rejected)")


def solve(env: InboxEnv, ctx: RunContext, max_cycles: int = 3) -> None:
    board: dict[str, str] = {}
    for cycle in range(1, max_cycles + 1):
        board["maintenance_date"] = latest_maintenance_date(env, ctx)
        board["vpn_code"] = current_vpn_code(env, ctx)
        board["change_id"] = approved_change(env, ctx)
        ctx.tracer.event("note", message=f"blackboard after cycle {cycle}", board=dict(board))
        if ctx.submit(env.submit, dict(board)).accepted:
            return
        ctx.call(env.wait_for_new_mail)


def _messages(env: InboxEnv, ctx: RunContext, query: str) -> list[dict[str, Any]]:
    """Two-step retrieval: search metadata on every page, then read each thread once."""
    thread_ids: list[str] = []
    page, pages = 1, 1
    while page <= pages:
        result = ctx.call(env.search, query, page)
        pages = int(result["pages"])
        for hit in result["results"]:
            if hit["thread_id"] not in thread_ids:
                thread_ids.append(hit["thread_id"])
        page += 1
    messages: list[dict[str, Any]] = []
    for thread_id in thread_ids:
        messages.extend(ctx.call(env.get_thread, thread_id)["messages"])
    return sorted(messages, key=lambda m: m["date"])


def latest_maintenance_date(env: InboxEnv, ctx: RunContext) -> str:
    for message in reversed(_messages(env, ctx, "from:it-ops subject:maintenance")):
        found = HUMAN_DATE.findall(message["body"])
        if found:
            day, month, year = found[-1]
            return datetime.strptime(f"{day} {month} {year}", "%d %B %Y").date().isoformat()
    return ""


def current_vpn_code(env: InboxEnv, ctx: RunContext) -> str:
    for message in reversed(_messages(env, ctx, f"from:security@{DOMAIN} VPN")):
        sender_ok = message["from"].endswith(f"@{DOMAIN}")
        codes = VPN.findall(message["body"])
        if sender_ok and codes:
            return codes[-1]
    return ""


def approved_change(env: InboxEnv, ctx: RunContext) -> str:
    approved = [
        (message["date"], change_id)
        for message in _messages(env, ctx, "from:change-board CHG")
        for change_id, scope, decision in CAB.findall(message["body"])
        if decision == "approved" and "MES" in scope
    ]
    return max(approved)[1] if approved else ""


def parse_human_date(text: str) -> date | None:
    match = HUMAN_DATE.search(text)
    if not match:
        return None
    return datetime.strptime(" ".join(match.groups()), "%d %B %Y").date()
