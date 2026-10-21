"""MCP: labs served as MCP servers, an MCP host merging servers, and tool discovery."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import Client

from aisyslab.core.lab import EnvOptions
from aisyslab.core.messages import ToolCall
from aisyslab.core.model import ScriptedModel, tool_turn
from aisyslab.core.runner import run_once
from aisyslab.labs import LABS
from aisyslab.labs.plant_database.baseline import BLOCKED_ORDERS
from aisyslab.labs.plant_database.env import PlantDatabaseEnv
from aisyslab.mcp.host import McpHost
from aisyslab.mcp.route import routed
from aisyslab.mcp.server import lab_server


def test_lab_server_speaks_mcp() -> None:
    served = lab_server(LABS["plant-database"], 4)

    async def session() -> None:
        async with Client(served.server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == set(served.toolbox.names())
            resource = await client.read_resource(served.task_uri)
            assert "line 3" in resource.contents[0].text
            ok = await client.call_tool("list_tables", {})
            assert not ok.is_error and "work_orders" in ok.content[0].text
            refused = await client.call_tool("run_query", {"sql": "DELETE FROM stock"})
            assert refused.is_error and "Read-only" in json.loads(refused.content[0].text)["error"]

    asyncio.run(session())
    assert served.env.write_attempts == 1


def test_host_merges_servers_with_prefixes_and_audits_them() -> None:
    plant = lab_server(LABS["plant-database"], 1)
    gate = lab_server(LABS["quality-gate"], 1)
    with McpHost({"plant": plant.server, "gate": gate.server}) as host:
        audit = {a.prefix: a for a in host.audit()}
        assert set(audit) == {"plant", "gate"} and "run_query" in audit["plant"].tools
        assert audit["gate"].resources == ["lab://quality-gate/task"]
        box = host.toolbox()
        assert "plant__run_query" in box.names() and "gate__train_tree" in box.names()
        ok = box.execute(ToolCall(id="1", name="plant__list_tables", arguments={}))
        assert not ok.is_error and "work_orders" in ok.content
        bad = box.execute(ToolCall(id="2", name="gate__train_tree", arguments={"features": ["no_such_column"]}))
        assert bad.is_error and json.loads(bad.content)["hint"].startswith("Features:")
        assert host.read_resource("plant", "lab://plant-database/task").startswith("Planner's question")


def test_agent_solves_a_lab_through_mcp() -> None:
    expected = sorted(PlantDatabaseEnv(4, EnvOptions()).expected)
    script = [
        tool_turn("plant_database__run_query", sql=BLOCKED_ORDERS),
        tool_turn("plant_database__submit_answer", work_order_ids=expected),
    ]
    report = run_once(routed(LABS["plant-database"], via_mcp=True), "agent", 4, model=ScriptedModel(script))
    assert report.accepted and report.status == "goal_reached"


def test_discovery_gives_the_model_two_tools() -> None:
    expected = sorted(PlantDatabaseEnv(5, EnvOptions()).expected)
    seen: list[list[str]] = []

    def policy(_s, messages, tools):  # type: ignore[no-untyped-def]
        seen.append([t.name for t in tools])
        done = [m.tool_name for m in messages if m.role == "tool"]
        if not done:
            return tool_turn("search_tools", query="run a SQL query on the database")
        if len(done) == 1:
            found = [t["name"] for t in json.loads(messages[-1].content)["tools"]]
            assert "plant_database__run_query" in found
            return tool_turn("call_tool", tool="plant_database__run_query", arguments={"sql": BLOCKED_ORDERS})
        return tool_turn("call_tool", tool="plant_database__submit_answer", arguments={"work_order_ids": expected})

    lab = routed(LABS["plant-database"], via_mcp=True, discover=True)
    report = run_once(lab, "agent", 5, model=ScriptedModel(policy))
    assert report.accepted and all(names == ["search_tools", "call_tool"] for names in seen)


def test_stdio_transport_end_to_end() -> None:
    """The way a desktop host runs a local server: a subprocess talking JSON-RPC over stdio."""
    import sys

    from mcp import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "aisyslab.cli", "mcp", "quality-gate", "--seed", "2"]
    )
    with McpHost({"gate": params}) as host:
        box = host.toolbox()
        assert "gate__train_tree" in box.names()
        trained = box.execute(
            ToolCall(id="1", name="gate__train_tree", arguments={"features": ["torque_nm", "cure_temp_c"]})
        )
        assert not trained.is_error and json.loads(trained.content)["model_id"] == "M1"
