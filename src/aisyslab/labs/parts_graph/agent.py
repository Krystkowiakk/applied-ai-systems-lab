"""Agent solver: build the graph once, then answer by traversal, not by similarity.

``build_graph`` is an ingestion step, not a conversation: tables load by code, and each note is
read into facts by one model call with a fresh context, checked by the parser, repaired once if a
line is rejected. The agent then walks the chain from the part to the orders, looks at the facts
behind a node (current, ended, planned), reads a document when a fact looks wrong, and corrects the
graph through the same parser.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.lab import RunContext
from ...core.messages import Message
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox, ToolError
from ...graph import GraphStore, extract, parse_facts
from .env import PartsGraphEnv
from .plant import SCHEMA, load_tables

SYSTEM = """You answer a planner's question about a late part from the plant's documents.

The relations are the knowledge here, and no single document connects a part to an order: the bill
of materials puts parts in assemblies, the installation register and logbooks put assemblies on
machines, the layout and changeover notes put machines on lines, the line plan says what each line
makes, and the order book says which product each order is for.

- Call build_graph first. Tables are loaded by code; each note is read into facts by a model and
  checked by a parser. It reports what each note added, what it changed and what it could not read.
  A rejected line is a fact missing from the graph: read that note and add what it states with
  correct_graph before you answer.
- Follow the chain with walk, and look at one node's facts with edges: current, ended and planned
  facts are listed apart, with the documents behind them. Only current facts count today.
- When a fact looks wrong or missing, read the document behind it and fix the graph with
  correct_graph.
- An approved alternate saves an order only if it is approved now, in the assembly on that order's
  path. Orders count when they are open and due before the part arrives.
- Your answer counts only through submit_impact. When it is not accepted, find what the feedback
  names, fix the graph and submit again."""

TASK = "Answer the planner's question from the question tool."


class ReadArgs(BaseModel):
    doc_id: str


class SearchArgs(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=8)


class EdgesArgs(BaseModel):
    node: str = Field(description="An ID: part, assembly, machine, line, product or order.")
    relation: str | None = Field(default=None, description="Only this relation.")
    direction: Literal["out", "in", "both"] = "both"


class WalkArgs(BaseModel):
    start: str
    path: list[str] = Field(min_length=1, max_length=6, description="Relation names; '<' in front goes backwards.")


class CorrectArgs(BaseModel):
    doc_id: str = Field(description="The document that states the facts.")
    add: list[str] = Field(
        default_factory=list, description="Facts it states, e.g. 'MC-99 RUNS_ON L9 since 2026-10-01'."
    )
    remove: list[str] = Field(default_factory=list, description="Facts stored from it that it does not state.")


class ImpactArgs(BaseModel):
    lines: list[str] = Field(description="Lines the late part holds up today.")
    at_risk_orders: list[str] = Field(
        description="Open orders due before the part arrives, with no approved alternate."
    )
    saved_orders: list[str] = Field(description="Open orders due before then that an approved alternate saves.")


def toolbox(env: PartsGraphEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    graph = GraphStore(SCHEMA)
    built: dict[str, Any] = {}

    def call(system: str, user: str, purpose: str) -> str:
        assert model is not None
        messages = [Message(role="user", content=user)]
        return ctx.side_call(model, system=system, messages=messages, purpose=purpose, max_output_tokens=1500)

    def build_graph(_: NoArgs) -> dict[str, Any]:
        if built:
            return built | {"note": "The graph is already built; this is the same report."}
        if model is None:
            raise ToolError("No model is configured here to read the notes.")
        listing = ctx.tool_call(env.list_documents)["documents"]
        docs = [ctx.tool_call(env.read_document, d["id"]) for d in listing]
        tables = load_tables(graph, docs)
        known = graph.node_ids()
        notes: dict[str, Any] = {}
        changes: list[str] = []
        for doc in docs:
            if doc["kind"] != "note":
                continue
            result = extract(call, SCHEMA, source=doc["id"], document=doc["text"], known=known)
            for fact in result.facts:
                changes += [f"{c} ({doc['id']})" for c in graph.add(fact, doc["id"])]
            notes[doc["id"]] = {"facts": [str(f) for f in result.facts]} | (
                {"rejected": result.rejected} if result.rejected else {}
            )
        rejected = {doc_id: len(n["rejected"]) for doc_id, n in notes.items() if "rejected" in n}
        built.update(
            today=env.today,
            fact_format=[SCHEMA.example(r) for r in SCHEMA.relations],
            tables=tables,
            notes=notes,
            changes=changes,
            conflicts=graph.conflicts(day=env.today),
        )
        if rejected:
            built["warning"] = (
                f"{sum(rejected.values())} lines from {len(rejected)} notes were rejected and are not in the graph "
                f"({', '.join(rejected)}). Read those notes and add their facts with correct_graph."
            )
        return built

    def need_graph() -> None:
        if not built:
            raise ToolError("Build the graph first with build_graph.")

    def edges(args: EdgesArgs) -> dict[str, Any]:
        need_graph()
        node = graph.node(args.node.strip())
        if node is None:
            raise ToolError(f"No node '{args.node}' in the graph.", hint="Use IDs as the documents write them.")
        found = graph.edges(node["id"], day=env.today, rel=args.relation, direction=args.direction)
        listed: dict[str, Any] = {
            state: [str(e) for e in found if e.state == state][:40] for state in ("current", "ended", "planned")
        }
        return {"node": node["id"], "type": node["type"], "details": node["props"]} | listed

    def walk(args: WalkArgs) -> dict[str, Any]:
        need_graph()
        try:
            paths = graph.walk(args.start.strip(), args.path, day=env.today)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        results = []
        for path in paths:
            hops = [args.start.strip()]
            for step, nxt in zip(args.path, path["nodes"][1:], strict=True):
                hops.append(f"<-{step[1:]}- {nxt}" if step.startswith("<") else f"-{step}-> {nxt}")
            end = graph.node(path["nodes"][-1])
            sources = list(dict.fromkeys(s for via in path["via"] for s in via))
            results.append({"path": " ".join(hops), "end": end["props"] if end else {}, "documents": sources})
        return {"today": env.today, "paths": len(results), "results": results}

    def correct_graph(args: CorrectArgs) -> dict[str, Any]:
        need_graph()
        doc = ctx.tool_call(env.read_document, args.doc_id)
        known = graph.node_ids()
        added = parse_facts("\n".join(args.add), SCHEMA, document=doc["text"], known=known)
        removed = parse_facts("\n".join(args.remove), SCHEMA, document=doc["text"], known=known)
        changes: list[str] = []
        for fact in added.facts:
            changes += graph.add(fact, doc["id"])
        not_found = [str(f) for f in removed.facts if not graph.remove(f, doc["id"])]
        return {
            "added": [str(f) for f in added.facts],
            "removed": [str(f) for f in removed.facts if str(f) not in not_found],
            "changes": changes,
            "rejected": added.errors + removed.errors + [f"{f}: not stored from {doc['id']}" for f in not_found],
            "conflicts": graph.conflicts(day=env.today),
        }

    def submit(args: ImpactArgs) -> dict[str, Any]:
        verdict = ctx.submit(env.submit, args.lines, args.at_risk_orders, args.saved_orders, as_tool=True)
        return verdict.as_dict()

    return Toolbox(
        [
            Tool("question", "The planner's question.", NoArgs, lambda _: ctx.tool_call(env.question)),
            Tool(
                "list_documents",
                "The plant's documents: id, title, kind (table or note) and date; also today's date.",
                NoArgs,
                lambda _: ctx.tool_call(env.list_documents),
            ),
            Tool(
                "read_document",
                "One document's full text.",
                ReadArgs,
                lambda a: ctx.tool_call(env.read_document, a.doc_id),
            ),
            Tool(
                "search_documents",
                "Search the documents by similarity (keywords and meaning); returns ids, titles and snippets.",
                SearchArgs,
                lambda a: ctx.tool_call(env.search, a.query, a.k),
            ),
            Tool(
                "build_graph",
                "Build the plant graph: tables by code, notes by a model checked by a parser. Reports what was "
                "loaded, what each note added, what changed and what was rejected.",
                NoArgs,
                build_graph,
            ),
            Tool(
                "edges",
                "One node's facts, split into current, ended and planned, with the documents behind them.",
                EdgesArgs,
                edges,
            ),
            Tool(
                "walk",
                "Every path from a node along a chain of relations, as of today, with the end node's details and "
                "the documents behind each path. A step is a relation name; '<' in front follows it backwards: "
                "['INSTALLED_ON', 'RUNS_ON'] goes from an assembly to its lines, ['<FOR_PRODUCT'] from a product "
                "to its orders.",
                WalkArgs,
                walk,
            ),
            Tool(
                "correct_graph",
                "Add facts a document states, or remove facts stored from it that it does not state, in the "
                "graph's line format. Checked by the same parser as the extraction.",
                CorrectArgs,
                correct_graph,
            ),
            Tool(
                "submit_impact",
                "Submit the lines held up, the orders at risk and the orders an approved alternate saves. Returns "
                "what is right and what to fix.",
                ImpactArgs,
                submit,
            ),
        ]
    )


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
