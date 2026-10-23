"""The plant's graph: its schema, the tables loaded by code, and the impact question asked in code."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping

from ...graph import Fact, GraphSchema, GraphStore, Relation

SCHEMA = GraphSchema(
    types={
        "Part": r"P-\d{4}",
        "Assembly": r"A-\d{2}",
        "Machine": r"MC-\d{2}",
        "Line": r"L\d",
        "Product": r"FG-\d{3}",
        "Order": r"SO-\d{4}",
    },
    examples={  # IDs no plant document uses, so a copied example fails the parser's source check
        "Part": "P-9999",
        "Assembly": "A-99",
        "Machine": "MC-99",
        "Line": "L9",
        "Product": "FG-999",
        "Order": "SO-9999",
    },
    relations=(
        Relation("PART_OF", "Part", "Assembly", "the part is in the assembly's bill of materials"),
        Relation(
            "INSTALLED_ON",
            "Assembly",
            "Machine",
            "the assembly is fitted on the machine",
            example="A-99 INSTALLED_ON MC-99 until 2026-03-01",
        ),
        Relation(
            "RUNS_ON",
            "Machine",
            "Line",
            "the machine stands on the line (one line at a time)",
            functional=True,
            example="MC-99 RUNS_ON L9 since 2026-03-01",
        ),
        Relation("MAKES", "Line", "Product", "the line makes the product"),
        Relation("FOR_PRODUCT", "Order", "Product", "the customer order is for the product"),
        Relation(
            "ALTERNATE_FOR",
            "Part",
            "Part",
            "the first part is approved to replace the second part in the assembly",
            scope="Assembly",
            example="P-9998 ALTERNATE_FOR P-9999 in A-99 since 2026-02-01",
        ),
    ),
)

# From a part to the orders it can hold up.
CHAIN = ("PART_OF", "INSTALLED_ON", "RUNS_ON", "MAKES", "<FOR_PRODUCT")

Loader = Callable[[GraphStore, dict[str, str], str], None]


def _bom(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add_node(row["assembly"], description=row["assembly_description"])
    g.add(Fact(row["part"], "PART_OF", row["assembly"]), source)


def _register(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add(Fact(row["assembly"], "INSTALLED_ON", row["machine"]), source)


def _layout(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add_node(row["machine"], description=row["machine_description"])
    g.add(Fact(row["machine"], "RUNS_ON", row["line"]), source)


def _line_plan(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add_node(row["product"], description=row["product_description"])
    g.add(Fact(row["line"], "MAKES", row["product"]), source)


def _orders(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add_node(row["order"], quantity=int(row["quantity"]), due=row["due"], status=row["status"])
    g.add(Fact(row["order"], "FOR_PRODUCT", row["product"]), source)


def _parts(g: GraphStore, row: dict[str, str], source: str) -> None:
    g.add_node(row["part"], description=row["description"], supplier=row["supplier"])


LOADERS: dict[str, Loader] = {
    "bom": _bom,
    "register": _register,
    "layout": _layout,
    "line-plan": _line_plan,
    "orders": _orders,
    "parts": _parts,
}


def load_tables(g: GraphStore, documents: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """What code can parse, code parses: every table row becomes nodes and facts, no model involved."""
    counts = {}
    for doc in documents:
        if doc["kind"] != "table":
            continue
        rows = list(csv.DictReader(doc["text"].splitlines()[1:]))
        for row in rows:
            LOADERS[doc["id"]](g, row, doc["id"])
        counts[doc["id"]] = len(rows)
    return counts


def impact(g: GraphStore, part: str, arrival: str, day: str) -> dict[str, list[str]]:
    """The planner's question asked of a graph in code: lines held up, orders at risk, orders saved."""
    approved = {
        e.fact.scope for e in g.edges(part, day=day, rel="ALTERNATE_FOR", direction="in") if e.state == "current"
    }
    lines = {path["nodes"][-1] for path in g.walk(part, CHAIN[:3], day=day)}
    risk: set[str] = set()
    covered: set[str] = set()
    for path in g.walk(part, CHAIN, day=day, limit=1000):
        assembly, order = path["nodes"][1], path["nodes"][-1]
        node = g.node(order)
        props = node["props"] if node else {}
        if props.get("status") == "open" and props.get("due", "9999-12-31") < arrival:
            (covered if assembly in approved else risk).add(order)
    return {"lines": sorted(lines), "at_risk": sorted(risk), "saved": sorted(covered - risk)}
