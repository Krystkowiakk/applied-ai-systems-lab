"""The graph store, the extraction parser, and the parts-graph lab that answers by traversal."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from aisyslab.core.lab import EnvOptions, RunContext
from aisyslab.core.messages import Message, ModelTurn, ToolCall, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.resilience import RetryPolicy
from aisyslab.core.runner import run_once
from aisyslab.core.tools import Toolbox
from aisyslab.core.trace import Tracer
from aisyslab.graph import Fact, GraphStore, extract, parse_facts
from aisyslab.labs import LABS
from aisyslab.labs.parts_graph import baseline
from aisyslab.labs.parts_graph.env import PartsGraphEnv
from aisyslab.labs.parts_graph.plant import CHAIN, SCHEMA, impact, load_tables
from aisyslab.labs.parts_graph.world import TODAY, build

DAY = TODAY.isoformat()


def plant() -> GraphStore:
    g = GraphStore(SCHEMA)
    g.add(Fact("MC-14", "RUNS_ON", "L2"), "layout")
    g.add(Fact("A-21", "INSTALLED_ON", "MC-14"), "register")
    g.add(Fact("P-1040", "PART_OF", "A-21"), "bom")
    for line, product in (("L1", "FG-201"), ("L2", "FG-202"), ("L3", "FG-203")):
        g.add(Fact(line, "MAKES", product), "line-plan")
    for order, product in (("SO-5501", "FG-202"), ("SO-5502", "FG-203"), ("SO-5503", "FG-203")):
        g.add_node(order, due="2026-11-05", status="open")
        g.add(Fact(order, "FOR_PRODUCT", product), "orders")
    return g


def lines_of(g: GraphStore, machine: str, day: str = DAY) -> list[str]:
    return [p["nodes"][-1] for p in g.walk(machine, ["RUNS_ON"], day=day)]


def test_a_dated_move_ends_the_layout_fact_and_a_planned_one_waits() -> None:
    g = plant()
    changes = g.add(Fact("MC-14", "RUNS_ON", "L3", since="2026-10-14"), "changeover")
    assert changes == ["MC-14 RUNS_ON L2 ends 2026-10-14: superseded by MC-14 RUNS_ON L3 since 2026-10-14"]
    assert lines_of(g, "MC-14") == ["L3"] and lines_of(g, "MC-14", "2026-10-01") == ["L2"]
    g.add(Fact("MC-14", "RUNS_ON", "L1", since="2026-11-30"), "minutes")
    assert lines_of(g, "MC-14") == ["L3"] and lines_of(g, "MC-14", "2026-12-01") == ["L1"]
    states = {str(e.fact): e.state for e in g.edges("MC-14", day=DAY, rel="RUNS_ON")}
    assert states == {
        "MC-14 RUNS_ON L2 until 2026-10-14": "ended",
        "MC-14 RUNS_ON L3 since 2026-10-14 until 2026-11-30": "current",
        "MC-14 RUNS_ON L1 since 2026-11-30": "planned",
    }
    orders = [p["nodes"][-1] for p in g.walk("P-1040", CHAIN, day=DAY)]
    assert orders == ["SO-5502", "SO-5503"]  # through L3 now, not L2
    assert g.edges("MC-14", day=DAY, rel="RUNS_ON")[0].sources == ("layout", "changeover")


def test_an_ending_and_its_start_merge_in_either_order() -> None:
    approve = Fact("P-1123", "ALTERNATE_FOR", "P-1040", scope="A-21", since="2026-09-02")
    withdraw = Fact("P-1123", "ALTERNATE_FOR", "P-1040", scope="A-21", until="2026-10-01")
    for first, second in ((approve, withdraw), (withdraw, approve)):
        g = plant()
        g.add(first, "a")
        g.add(second, "b")
        found = g.edges("P-1040", day=DAY, rel="ALTERNATE_FOR", direction="in")
        assert [(str(e.fact), e.state) for e in found] == [
            ("P-1123 ALTERNATE_FOR P-1040 in A-21 since 2026-09-02 until 2026-10-01", "ended")
        ]


def test_undated_conflicts_are_reported_and_a_source_can_be_withdrawn() -> None:
    g = plant()
    g.add(Fact("MC-14", "RUNS_ON", "L3"), "note-x")
    assert len(g.conflicts(day=DAY)) == 1 and "L2" in g.conflicts(day=DAY)[0] and "L3" in g.conflicts(day=DAY)[0]
    assert not g.remove(Fact("MC-14", "RUNS_ON", "L3"), "another-note")
    assert g.remove(Fact("MC-14", "RUNS_ON", "L3"), "note-x")
    assert g.conflicts(day=DAY) == [] and lines_of(g, "MC-14") == ["L2"]


def test_walk_goes_backwards_and_refuses_to_explode() -> None:
    g = plant()
    assert [p["nodes"] for p in g.walk("FG-203", ["<FOR_PRODUCT"], day=DAY)] == [
        ["FG-203", "SO-5502"],
        ["FG-203", "SO-5503"],
    ]
    with pytest.raises(ValueError, match="2 paths"):
        g.walk("FG-203", ["<FOR_PRODUCT"], day=DAY, limit=1)
    with pytest.raises(ValueError, match="unknown relation"):
        g.walk("FG-203", ["ORDERED_BY"], day=DAY)


DOC = "Changeover, 14 October 2026: MC-14 runs on line L3 from today; it no longer feeds L2. A-21 is fitted."


def test_the_parser_says_what_is_wrong_with_each_line() -> None:
    text = "\n".join(
        [
            "- MC-14 RUNS_ON L3 since 2026-10-14",
            "2. MC-14 RUNS_ON L2 until 2026-10-14",
            "MC-14 moved to L3",
            "MC-14 STANDS_ON L3",
            "L3 RUNS_ON MC-14",
            "A-21 INSTALLED_ON MC-14 in L3",
            "MC-14 RUNS_ON L3 since 14.10.2026",
            "MC-14 RUNS_ON L4",
            "MC-99 RUNS_ON L3",
            "Machine MC-14 RUNS_ON Line L3",
            "P-1040 DELAYED_UNTIL 2026-11-15",
            "NONE",
        ]
    )
    parsed = parse_facts(text, SCHEMA, document=DOC + " MC-99", known={"MC-14", "L2", "L3", "L4", "A-21"})
    assert [str(f) for f in parsed.facts] == ["MC-14 RUNS_ON L3 since 2026-10-14", "MC-14 RUNS_ON L2 until 2026-10-14"]
    reasons = [e.split("): ", 1)[1] for e in parsed.errors]
    assert reasons == [
        "not in the format SRC RELATION DST [in SCOPE] [since YYYY-MM-DD] [until YYYY-MM-DD]",
        "unknown relation STANDS_ON; use one of PART_OF, INSTALLED_ON, RUNS_ON, MAKES, FOR_PRODUCT, ALTERNATE_FOR",
        "RUNS_ON goes from a Machine to a Line; L3 is a Line",
        "INSTALLED_ON takes no 'in' part",
        "since '14.10.2026' is not a date in the form YYYY-MM-DD",
        "L4 does not appear in the document",
        "MC-99 is not a known Machine",
        "leave out the type names and write 'MC-14 RUNS_ON L3'",
        "the graph holds no relation DELAYED_UNTIL; it holds PART_OF, INSTALLED_ON, RUNS_ON, MAKES, FOR_PRODUCT, "
        "ALTERNATE_FOR",
    ]
    scoped = parse_facts("P-1123 ALTERNATE_FOR P-1040", SCHEMA, document="P-1123 P-1040")
    assert "needs 'in <Assembly>'" in scoped.errors[0]


def test_extraction_sends_rejected_lines_back_once() -> None:
    answers = iter(
        ["MC-14 RUNS_ON L3 since 14 October", "MC-14 RUNS_ON L3 since 2026-10-14\nMC-14 RUNS_ON L2 until 2026-10-14"]
    )
    prompts: list[tuple[str, str]] = []

    def call(system: str, user: str, purpose: str) -> str:
        prompts.append((user, purpose))
        return next(answers)

    result = extract(call, SCHEMA, source="changeover", document=DOC)
    assert result.calls == 2 and result.rejected == [] and len(result.facts) == 2
    assert prompts[1][1] == "graph extraction repair" and "not in the format" in prompts[1][0]
    stubborn = extract(lambda *_: "MC-14 RUNS_ON L9", SCHEMA, source="changeover", document=DOC)
    assert stubborn.calls == 2 and stubborn.facts == [] and "L9 does not appear" in stubborn.rejected[0]


@pytest.mark.parametrize("seed", range(20))
def test_the_notes_facts_rebuild_the_answer_and_no_document_holds_it(seed: int) -> None:
    world = build(seed)
    g = GraphStore(SCHEMA)
    load_tables(g, [d.as_dict() for d in world.documents])
    known = g.node_ids()
    for doc in world.documents:
        parsed = parse_facts("\n".join(doc.facts), SCHEMA, document=doc.text, known=known)
        assert parsed.errors == []
        for fact in parsed.facts:
            g.add(fact, doc.id)
    assert impact(g, world.part, world.arrival, DAY) == world.answer
    assert g.conflicts(day=DAY) == []
    orders = set(world.answer["at_risk"]) | set(world.answer["saved"])
    # No single document names the part and an order it holds up: similarity cannot find the answer in one hop.
    assert not any(world.part in d.text and any(o in d.text for o in orders) for d in world.documents)


def test_patterns_fall_for_every_trap_but_a_plain_move() -> None:
    for seed in range(20):
        env = PartsGraphEnv(seed, EnvOptions(flaky=0.2 if seed % 2 else 0.0))
        baseline.solve(env, RunContext(tracer=Tracer(), retry=RetryPolicy(sleep=lambda _s: None)))
        assert env.submissions[-1].accepted == (env.world.traps == ("moved",)), (seed, env.world.traps)


def test_the_verifier_counts_without_naming() -> None:
    env = PartsGraphEnv(0, EnvOptions())
    a = env.world.answer
    verdict = env.submit(a["lines"][:1], a["at_risk"] + ["SO-0000"], a["saved"] + a["at_risk"][:1])
    assert not verdict.accepted
    assert verdict.details["checks"] == {
        "lines": "1 line missing",
        "at_risk": "1 listed that should not be",
        "saved": "1 order(s) listed as both at risk and saved",
    }
    assert not re.search(r"SO-\d{4}|L\d", verdict.feedback)


def machine_in(text: str) -> str:
    found = re.search(r"MC-\d{2}", text)
    assert found is not None
    return found[0]


def extraction_policy(seed: int, plan: list[ModelTurn], skip: str = "") -> Callable[..., ModelTurn]:
    """Plays the model: extraction calls get the note's reference facts, the agent's turns follow the plan."""
    facts = {d.id: d.facts for d in build(seed).documents}

    def policy(system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        if system.startswith("You turn one plant document"):
            doc_id = (messages[-1].content or "").split(":", 1)[0].removeprefix("Document ")
            return text_turn("NONE" if doc_id == skip else "\n".join(facts[doc_id]) or "NONE")
        return plan.pop(0) if plan else text_turn("Done.")

    return policy


def call_tool(tools: Toolbox, name: str, **arguments: object) -> dict[str, Any]:
    result = tools.execute(ToolCall(id=name, name=name, arguments=arguments))
    assert not result.is_error, result.content
    return json.loads(result.content)


def test_the_agent_builds_the_graph_walks_it_and_submits() -> None:
    world = build(1)
    a = world.answer
    plan = [
        tool_turn("question"),
        tool_turn("build_graph"),
        tool_turn("walk", start=world.part, path=list(CHAIN)),
        tool_turn("edges", node=world.part, relation="ALTERNATE_FOR", direction="in"),
        tool_turn("submit_impact", lines=a["lines"], at_risk_orders=a["at_risk"], saved_orders=a["saved"]),
    ]
    report = run_once(LABS["parts-graph"], "agent", 1, model=ScriptedModel(extraction_policy(1, plan)))
    assert report.accepted, report.error
    notes = sum(d.kind == "note" for d in world.documents)
    assert report.model_calls == 5 + notes  # five agent turns, and one extraction call per note


def test_a_missed_fact_is_corrected_through_the_parser() -> None:
    world = build(1)  # its one trap: a machine on the part's path moved to another line
    tables = GraphStore(SCHEMA)
    load_tables(tables, [d.as_dict() for d in world.documents])
    carriers = {p["nodes"][-1] for p in tables.walk(world.part, ["PART_OF", "INSTALLED_ON"], day=DAY)}
    move = next(d for d in world.documents if d.id.startswith("changeover") and machine_in(d.title) in carriers)
    machine = machine_in(move.title)
    spec = LABS["parts-graph"].spec
    assert spec is not None
    model = ScriptedModel(extraction_policy(1, [], skip=move.id))
    tools = spec.toolbox(PartsGraphEnv(1, EnvOptions()), model, RunContext(tracer=Tracer()))
    assert call_tool(tools, "build_graph")["notes"][move.id] == {"facts": []}
    before = call_tool(tools, "walk", start=machine, path=["RUNS_ON"])
    fixed = call_tool(tools, "correct_graph", doc_id=move.id, add=[*move.facts, f"{machine} RUNS_ON L9"])
    after = call_tool(tools, "walk", start=machine, path=["RUNS_ON"])
    assert fixed["added"] == list(move.facts) and "L9 does not appear" in fixed["rejected"][0]
    assert before["results"][0]["documents"] == ["layout"] and after["results"][0]["documents"] == [move.id]
    assert before["results"][0]["path"] != after["results"][0]["path"]


def timeline(g: GraphStore, machine: str) -> list[str]:
    return sorted(str(e.fact) for e in g.edges(machine, day=DAY, rel="RUNS_ON", direction="out"))


def test_withdrawing_a_fact_undoes_what_it_ended() -> None:
    g = plant()
    move = Fact("MC-14", "RUNS_ON", "L3", since="2026-10-14")
    g.add(move, "note-x")
    assert lines_of(g, "MC-14") == ["L3"]
    assert g.remove(move, "note-x")
    assert lines_of(g, "MC-14") == ["L2"] and g.conflicts(day=DAY) == []  # the layout holds again
    ending = Fact("MC-14", "RUNS_ON", "L2", until="2026-10-14")
    g.add(ending, "note-y")
    assert lines_of(g, "MC-14") == []
    g.remove(ending, "note-y")
    assert lines_of(g, "MC-14") == ["L2"]


def test_facts_give_the_same_timeline_in_any_order() -> None:
    move = Fact("MC-14", "RUNS_ON", "L3", since="2026-10-14")
    planned = Fact("MC-14", "RUNS_ON", "L1", since="2026-11-30")
    first, second = plant(), plant()
    first.add(move, "changeover")
    first.add(planned, "minutes")
    second.add(planned, "minutes")
    second.add(move, "changeover")
    assert (
        timeline(first, "MC-14")
        == timeline(second, "MC-14")
        == [
            "MC-14 RUNS_ON L1 since 2026-11-30",
            "MC-14 RUNS_ON L2 until 2026-10-14",
            "MC-14 RUNS_ON L3 since 2026-10-14 until 2026-11-30",
        ]
    )
    assert first.conflicts(day=DAY) == second.conflicts(day=DAY) == []
    second.remove(planned, "minutes")  # the plan is cancelled: the move holds with no end
    assert "MC-14 RUNS_ON L3 since 2026-10-14" in timeline(second, "MC-14")
