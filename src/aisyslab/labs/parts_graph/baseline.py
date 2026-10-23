"""Deterministic solver: the same graph, with the notes read by patterns instead of a model.

Tables load exactly as in the agent's graph. In the notes, patterns find a machine that "runs on
line" somewhere, every assembly a logbook names, and every assembly an alternate is "approved" in,
each dated by the note. That holds for a plain changeover. It does not for a move planned for next
month (dated today), an assembly that went to stores or one a machine does not use (both named in a
logbook), or an approval withdrawn in other words. Then it asks the graph the planner's question in
code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime

from ...core.lab import RunContext
from ...graph import Fact, GraphStore
from .env import PartsGraphEnv
from .plant import SCHEMA, impact, load_tables

MOVE = re.compile(r"(MC-\d{2})[^.]*?\bruns? on line (L\d)")
ALTERNATE = re.compile(r"(P-\d{4})[^.]*?approved as an alternate for (P-\d{4})([^.]*)")
LOGBOOK = re.compile(r"Maintenance logbook, (MC-\d{2})")
DELAY = re.compile(r"not ship (P-\d{4}) before (\d{1,2} \w+ \d{4})")


def facts_by_pattern(doc: Mapping[str, str]) -> list[Fact]:
    day = doc["date"]
    facts = [Fact(m[1], "RUNS_ON", m[2], since=day) for m in MOVE.finditer(doc["text"])]
    for m in ALTERNATE.finditer(doc["text"]):
        facts += [Fact(m[1], "ALTERNATE_FOR", m[2], scope=a, since=day) for a in re.findall(r"A-\d{2}", m[3])]
    logbook = LOGBOOK.match(doc["title"])
    if logbook:
        named = dict.fromkeys(re.findall(r"A-\d{2}", doc["text"]))
        facts += [Fact(a, "INSTALLED_ON", logbook[1], since=day) for a in named]
    return facts


def solve(env: PartsGraphEnv, ctx: RunContext) -> None:
    listing = ctx.call(env.list_documents)
    docs = [ctx.call(env.read_document, d["id"]) for d in listing["documents"]]
    graph = GraphStore(SCHEMA)
    load_tables(graph, docs)
    part, arrival = "", "9999-12-31"
    for doc in docs:
        if doc["kind"] != "note":
            continue
        for fact in facts_by_pattern(doc):
            graph.add(fact, doc["id"])
        delay = DELAY.search(doc["text"])
        if delay:
            part, arrival = delay[1], datetime.strptime(delay[2], "%d %B %Y").date().isoformat()
    answer = impact(graph, part, arrival, listing["today"]) if part else {"lines": [], "at_risk": [], "saved": []}
    ctx.submit(env.submit, answer["lines"], answer["at_risk"], answer["saved"])
