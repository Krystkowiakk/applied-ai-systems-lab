"""A small property graph on SQLite: typed nodes, typed edges with dates, and traversal.

Nodes have a type, read from the pattern of their ID, and properties. Edges have a relation, an
optional scope (the assembly an alternate part is approved in), the dates between which they hold,
and the documents they came from. The schema says which relations exist, between which node types,
and which are functional: a machine stands on one line at a time, so a dated move ends the fact it
replaces, and a changeover note supersedes the layout table without anyone deleting a row.

Traversal is what similarity search cannot do. ``walk`` follows a chain of relations from a node
and returns every path, with the documents behind each hop; ``edges`` lists a node's facts as
current, ended and planned. Production graphs live in a graph database (Neo4j) or in Postgres; this
store is small enough to read in one sitting and runs anywhere.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DDL = """
CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT NOT NULL, props TEXT NOT NULL DEFAULT '{}');
CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    src TEXT NOT NULL REFERENCES nodes(id),
    rel TEXT NOT NULL,
    dst TEXT NOT NULL REFERENCES nodes(id),
    scope TEXT NOT NULL DEFAULT '',
    since TEXT,
    until TEXT,
    sources TEXT NOT NULL
);
CREATE INDEX edges_out ON edges(src, rel);
CREATE INDEX edges_in ON edges(dst, rel);
"""

# An edge holds on a day when it has started and not yet ended. Dates are ISO strings.
CURRENT = "(since IS NULL OR since <= :day) AND (until IS NULL OR until > :day)"


@dataclass(frozen=True)
class Relation:
    name: str
    src: str
    dst: str
    meaning: str
    functional: bool = False  # one current target per source: a dated new fact ends the old one
    scope: str | None = None  # the node type that qualifies the fact ("in A-21"); required when set
    example: str = ""  # one line in the parser's format, for prompts


@dataclass(frozen=True)
class GraphSchema:
    types: dict[str, str]  # node type -> ID pattern
    examples: dict[str, str]  # node type -> an example ID, for prompts
    relations: tuple[Relation, ...]

    def type_of(self, node_id: str) -> str | None:
        return next((t for t, pattern in self.types.items() if re.fullmatch(pattern, node_id)), None)

    def relation(self, name: str) -> Relation | None:
        return next((r for r in self.relations if r.name == name), None)

    def example(self, relation: Relation) -> str:
        if relation.example:
            return relation.example
        line = f"{self.examples[relation.src]} {relation.name} {self.examples[relation.dst]}"
        return line + (f" in {self.examples[relation.scope]}" if relation.scope else "")


@dataclass(frozen=True)
class Fact:
    src: str
    rel: str
    dst: str
    scope: str = ""
    since: str | None = None
    until: str | None = None

    def __str__(self) -> str:
        text = f"{self.src} {self.rel} {self.dst}"
        text += f" in {self.scope}" if self.scope else ""
        text += f" since {self.since}" if self.since else ""
        return text + (f" until {self.until}" if self.until else "")


@dataclass(frozen=True)
class Edge:
    fact: Fact
    sources: tuple[str, ...]
    state: str  # "current", "ended" or "planned", on the day asked about

    def __str__(self) -> str:
        return f"{self.fact}  [{', '.join(self.sources)}]"


class GraphStore:
    def __init__(self, schema: GraphSchema) -> None:
        self.schema = schema
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(DDL)

    # Nodes

    def add_node(self, node_id: str, **props: Any) -> None:
        node_type = self.schema.type_of(node_id)
        if node_type is None:
            raise ValueError(f"{node_id} matches no node type")
        row = self.db.execute("SELECT props FROM nodes WHERE id = ?", (node_id,)).fetchone()
        merged = (json.loads(row["props"]) if row else {}) | props
        self.db.execute(
            "INSERT INTO nodes (id, type, props) VALUES (?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET props = excluded.props",
            (node_id, node_type, json.dumps(merged)),
        )

    def node(self, node_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return {"id": row["id"], "type": row["type"], "props": json.loads(row["props"])} if row else None

    def node_ids(self) -> set[str]:
        return {row["id"] for row in self.db.execute("SELECT id FROM nodes")}

    # Facts

    def add(self, fact: Fact, source: str) -> list[str]:
        """Store a fact from a source. Returns what it changed beyond itself, in words."""
        relation = self.schema.relation(fact.rel)
        if relation is None:
            raise ValueError(f"unknown relation {fact.rel}")
        for node_id in (fact.src, fact.dst, fact.scope):
            if node_id and self.node(node_id) is None:
                self.add_node(node_id)
        same = self._rows("src = ? AND rel = ? AND dst = ? AND scope = ?", fact.src, fact.rel, fact.dst, fact.scope)
        for row in same:  # already known, or known with a date this fact fills in (in either order)
            if (
                (row["since"], row["until"]) == (fact.since, fact.until)
                or (fact.until and row["until"] == fact.until and row["since"] in (None, fact.since))
                or (fact.since and not fact.until and row["since"] is None and (row["until"] or "") > fact.since)
            ):
                self._update(row, source, since=row["since"] or fact.since)
                return []
        current = next(
            (r for r in same if r["until"] is None and (fact.since is None or r["since"] in (None, fact.since))), None
        )
        if current is not None:  # this fact dates or ends one we hold
            self._update(current, source, since=current["since"] or fact.since, until=fact.until)
            return [f"{_fact(current)} ended {fact.until}"] if fact.until else []
        changes = []
        if relation.functional and fact.since and not fact.until:
            for row in self._rows(
                "src = ? AND rel = ? AND scope = ? AND dst != ? AND until IS NULL AND (since IS NULL OR since < ?)",
                fact.src,
                fact.rel,
                fact.scope,
                fact.dst,
                fact.since,
            ):
                self._update(row, source, until=fact.since)
                changes.append(f"{_fact(row)} ends {fact.since}: superseded by {fact}")
        self.db.execute(
            "INSERT INTO edges (src, rel, dst, scope, since, until, sources) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fact.src, fact.rel, fact.dst, fact.scope, fact.since, fact.until, json.dumps([source])),
        )
        return changes

    def remove(self, fact: Fact, source: str) -> bool:
        """Withdraw what one source said about a fact; the fact goes when no source is left."""
        removed = False
        for row in self._rows(
            "src = ? AND rel = ? AND dst = ? AND scope = ?", fact.src, fact.rel, fact.dst, fact.scope
        ):
            sources = json.loads(row["sources"])
            if source not in sources:
                continue
            removed = True
            if sources == [source]:
                self.db.execute("DELETE FROM edges WHERE id = ?", (row["id"],))
            else:
                sources.remove(source)
                self.db.execute("UPDATE edges SET sources = ? WHERE id = ?", (json.dumps(sources), row["id"]))
        return removed

    # Reading

    def edges(self, node_id: str, *, day: str, rel: str | None = None, direction: str = "both") -> list[Edge]:
        """A node's facts, each marked current, ended or planned on ``day``."""
        ends = {"out": ["src = :node"], "in": ["dst = :node"], "both": ["src = :node", "dst = :node"]}[direction]
        sql = f"SELECT * FROM edges WHERE ({' OR '.join(ends)})" + (" AND rel = :rel" if rel else "") + " ORDER BY id"
        found = []
        for row in self.db.execute(sql, {"node": node_id, "rel": rel}):
            if row["until"] is not None and row["until"] <= day:
                state = "ended"
            elif row["since"] is not None and row["since"] > day:
                state = "planned"
            else:
                state = "current"
            found.append(Edge(_fact(row), tuple(json.loads(row["sources"])), state))
        return found

    def walk(self, start: str, steps: Sequence[str], *, day: str, limit: int = 60) -> list[dict[str, Any]]:
        """Every path from ``start`` along a chain of relations that hold on ``day``.

        A step is a relation name, followed from source to target; with a leading ``<`` it is
        followed backwards (``<FOR_PRODUCT`` goes from a product to its orders)."""
        paths: list[dict[str, Any]] = [{"nodes": [start], "via": []}]
        for step in steps:
            rel = step.lstrip("<")
            if self.schema.relation(rel) is None:
                raise ValueError(f"unknown relation {rel}; the relations are {self._relation_names()}")
            near, far = ("dst", "src") if step.startswith("<") else ("src", "dst")
            grown = []
            for path in paths:
                rows = self.db.execute(
                    f"SELECT * FROM edges WHERE {near} = :node AND rel = :rel AND {CURRENT} ORDER BY {far}",
                    {"node": path["nodes"][-1], "rel": rel, "day": day},
                ).fetchall()
                for row in rows:
                    if row[far] not in path["nodes"]:
                        grown.append(
                            {"nodes": [*path["nodes"], row[far]], "via": [*path["via"], json.loads(row["sources"])]}
                        )
            paths = grown
            if len(paths) > limit:
                raise ValueError(f"{len(paths)} paths after {step}: start further down the chain or walk fewer steps")
        return paths

    def conflicts(self, *, day: str) -> list[str]:
        """Functional relations with more than one current target: facts that need a date or a fix."""
        found = []
        for relation in self.schema.relations:
            if not relation.functional:
                continue
            groups = self.db.execute(
                f"SELECT src, scope, COUNT(*) AS n FROM edges WHERE rel = :rel AND {CURRENT} "
                "GROUP BY src, scope HAVING n > 1",
                {"rel": relation.name, "day": day},
            ).fetchall()
            for group in groups:
                edges = self.edges(group["src"], day=day, rel=relation.name, direction="out")
                held = [str(e) for e in edges if e.state == "current"]
                found.append(f"{group['src']} {relation.name} has {group['n']} current targets: " + "; ".join(held))
        return found

    def stats(self) -> dict[str, dict[str, int]]:
        nodes = self.db.execute("SELECT type, COUNT(*) AS n FROM nodes GROUP BY type ORDER BY type")
        edges = self.db.execute("SELECT rel, COUNT(*) AS n FROM edges GROUP BY rel ORDER BY rel")
        return {"nodes": {r["type"]: r["n"] for r in nodes}, "edges": {r["rel"]: r["n"] for r in edges}}

    def _rows(self, where: str, *params: Any) -> list[sqlite3.Row]:
        return self.db.execute(f"SELECT * FROM edges WHERE {where} ORDER BY id", params).fetchall()

    def _update(self, row: sqlite3.Row, source: str, **dates: str | None) -> None:
        sources = json.loads(row["sources"])
        if source not in sources:
            sources.append(source)
        since = dates.get("since", row["since"])
        until = dates.get("until", row["until"])
        self.db.execute(
            "UPDATE edges SET since = ?, until = ?, sources = ? WHERE id = ?",
            (since, until, json.dumps(sources), row["id"]),
        )

    def _relation_names(self) -> str:
        return ", ".join(r.name for r in self.schema.relations)


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(row["src"], row["rel"], row["dst"], row["scope"], row["since"], row["until"])
