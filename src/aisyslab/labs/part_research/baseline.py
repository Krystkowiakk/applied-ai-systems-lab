"""Deterministic solver: search, take the top hit, extract with patterns.

It searches for the part, follows the first result and takes the first other part number on the
page; then the first lead time and the first pressure rating it finds for that part. On this web
the first answer to "VT-200 replacement" is a forum post with the wrong valve, and the first lead
time can come from an archived page. It reports what it fetched, with one citation per source.
"""

from __future__ import annotations

import re

from ...core.lab import RunContext
from .env import PartResearchEnv


def solve(env: PartResearchEnv, ctx: RunContext) -> None:
    sources: list[str] = []

    def first(query: str, pattern: str) -> str | None:
        for hit in ctx.call(env.search, query, 5)["results"]:
            page = ctx.call(env.fetch, hit["url"])
            sources.append(page["url"])
            m = re.search(pattern, page["text"])
            if m:
                return m.group(1)
        return None

    part = first("VT-200 replacement", r"\b((?:VT|HX)-(?!200)\d+)\b") or "VT-210"
    weeks = first(f"{part} lead time", r"lead time (\d+) weeks") or "0"
    bar = first(f"{part} datasheet max pressure", r"max pressure (\d+) bar") or "0"
    unique = list(dict.fromkeys(sources))
    lines = [f"- {part} is the replacement [1].", f"- Lead time {weeks} weeks [2].", f"- Max pressure {bar} bar [3]."]
    text = "# Replacement for VT-200\n\n" + "\n".join(lines[: len(unique)])
    ctx.call(env.store_report, text, unique)
    ctx.submit(env.submit, part, int(weeks), int(bar), [])
