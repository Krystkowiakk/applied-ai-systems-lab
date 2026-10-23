"""Extraction from documents into the graph, validated by a parser.

A model reads one document and writes facts in a line format. The parser checks every line: the
grammar, the relation, the node types at both ends, the dates, and that every ID appears in the
document and, when a registry is given, exists. Rejected lines go back to the model once, with the
reasons; what still fails is reported and never stored. Tables do not go through a model at all:
what code can parse, code parses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import date

from .store import Fact, GraphSchema

# (system, user, purpose) -> the model's text. The caller decides how calls are made and costed.
ModelCall = Callable[[str, str, str], str]

FORMAT = "SRC RELATION DST [in SCOPE] [since YYYY-MM-DD] [until YYYY-MM-DD]"
FACT_LINE = re.compile(
    r"^(?P<src>[A-Z][A-Z0-9-]*)\s+(?P<rel>[A-Z][A-Z_]+)\s+(?P<dst>[A-Z][A-Z0-9-]*)"
    r"(?:\s+in\s+(?P<scope>[A-Z][A-Z0-9-]*))?"
    r"(?:\s+since\s+(?P<since>\S+))?"
    r"(?:\s+until\s+(?P<until>\S+))?$"
)
BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+")

EXTRACTION_SYSTEM = """You turn one plant document into facts for a graph: one fact per line, nothing else.

Format: {format}
A line holds IDs and these keywords only, never type names or other words: write
{plain}, not {typed}.

Relations, each with an example line:
{relations}

- Write only what the document states as true. What it says is not the case is not a fact.
- A change has two sides: the old fact ends (until the day of the change) and the new one starts
  (since that day). A plan for a later day is written the same way, with that later day.
- An approval that is withdrawn or suspended ends on the day it was withdrawn: write it with until
  that day.
- Use the IDs exactly as the document writes them. If the document states none of these facts, write
  NONE."""


@dataclass
class Parsed:
    facts: list[Fact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Extraction:
    source: str
    facts: list[Fact]
    rejected: list[str]
    calls: int


def extraction_prompt(schema: GraphSchema) -> str:
    relations = [f"- {r.name}: {r.meaning}. Example: {schema.example(r)}" for r in schema.relations]
    first = schema.relations[0]
    plain = schema.example(first)
    typed = f"{first.src} {plain.split()[0]} {first.name} {first.dst} {plain.split()[2]}"
    return EXTRACTION_SYSTEM.format(format=FORMAT, plain=plain, typed=typed, relations="\n".join(relations))


def parse_facts(text: str, schema: GraphSchema, *, document: str, known: Collection[str] | None = None) -> Parsed:
    """Facts from a model's lines; every line that is not a valid, grounded fact becomes an error."""
    parsed = Parsed()
    for number, raw in enumerate(text.strip().splitlines(), 1):
        line = BULLET.sub("", raw.strip()).strip("`").strip()
        if not line or line.upper() == "NONE":
            continue
        match = FACT_LINE.match(line)
        if match is None:
            parsed.errors.append(f"line {number} ({line[:70]!r}): {_format_problem(line, schema)}")
            continue
        fact = Fact(match["src"], match["rel"], match["dst"], match["scope"] or "", match["since"], match["until"])
        problem = _problem(fact, schema, document, known)
        if problem:
            parsed.errors.append(f"line {number} ({line[:70]!r}): {problem}")
        elif fact not in parsed.facts:
            parsed.facts.append(fact)
    return parsed


def extract(
    call: ModelCall,
    schema: GraphSchema,
    *,
    source: str,
    document: str,
    known: Collection[str] | None = None,
    repairs: int = 1,
) -> Extraction:
    """One model call per document, plus at most ``repairs`` calls to fix rejected lines."""
    system = extraction_prompt(schema)
    user = f"Document {source}:\n\n{document}"
    text = call(system, user, "graph extraction")
    parsed = parse_facts(text, schema, document=document, known=known)
    calls = 1
    while parsed.errors and calls <= repairs:
        rejected = "\n".join(parsed.errors)
        retry = f"{user}\n\nYou wrote:\n{text}\n\nThe parser rejected these lines:\n{rejected}\n\nWrite the whole list again, corrected."
        text = call(system, retry, "graph extraction repair")
        parsed = parse_facts(text, schema, document=document, known=known)
        calls += 1
    return Extraction(source, parsed.facts, parsed.errors, calls)


def _problem(fact: Fact, schema: GraphSchema, document: str, known: Collection[str] | None) -> str | None:
    relation = schema.relation(fact.rel)
    if relation is None:
        return f"unknown relation {fact.rel}; use one of {', '.join(r.name for r in schema.relations)}"
    shape = f"{relation.name} goes from a {relation.src} to a {relation.dst}" + (
        f" in a {relation.scope}" if relation.scope else ""
    )
    if fact.scope and not relation.scope:
        return f"{relation.name} takes no 'in' part"
    if relation.scope and not fact.scope:
        return f"{relation.name} needs 'in <{relation.scope}>': {shape}"
    for node_id, expected in ((fact.src, relation.src), (fact.dst, relation.dst), (fact.scope, relation.scope)):
        if not node_id:
            continue
        actual = schema.type_of(node_id)
        if actual is None:
            return f"{node_id} is not an ID this graph holds"
        if actual != expected:
            return f"{shape}; {node_id} is a {actual}"
    for label, value in (("since", fact.since), ("until", fact.until)):
        if value is not None and not _iso_date(value):
            return f"{label} {value!r} is not a date in the form YYYY-MM-DD"
    if fact.since and fact.until and fact.since >= fact.until:
        return "since must come before until"
    for node_id in filter(None, (fact.src, fact.dst, fact.scope)):
        if not re.search(rf"(?<![\w-]){re.escape(node_id)}(?![\w-])", document):
            return f"{node_id} does not appear in the document"
        if known is not None and node_id not in known:
            return f"{node_id} is not a known {schema.type_of(node_id)}"
    return None


def _format_problem(line: str, schema: GraphSchema) -> str:
    """Say how to fix a line that does not parse, when that can be seen."""
    names = {t.lower() for t in schema.types}
    words = line.split()
    plain = " ".join(word for word in words if word.lower() not in names)
    if plain != line and FACT_LINE.match(plain):
        return f"leave out the type names and write {plain!r}"
    if len(words) > 1 and re.fullmatch(r"[A-Z][A-Z_]+", words[1]) and schema.relation(words[1]) is None:
        relations = ", ".join(r.name for r in schema.relations)
        return f"the graph holds no relation {words[1]}; it holds {relations}"
    return f"not in the format {FORMAT}"


def _iso_date(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True
