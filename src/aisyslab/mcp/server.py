"""Serve a lab over the Model Context Protocol.

Any MCP host (Claude Desktop, an IDE, another agent) can then work the lab with its own model:
the server lists the lab's tools with their JSON schemas, runs them against one seeded
environment, and publishes the task as a resource. Tool errors come back as results with
``is_error`` set, so the host's model can read the hint and recover, exactly as in the local
agent loop.

Transports: stdio for a desktop host on the same machine (one process per user), streamable
HTTP for anything remote or shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server

from ..core.lab import EnvOptions, Lab, RunContext
from ..core.messages import ToolCall
from ..core.model import ChatModel
from ..core.tools import Toolbox
from ..core.trace import Tracer


@dataclass
class LabServer:
    """An MCP server bound to one environment, with the toolbox it serves."""

    lab: Lab
    env: Any
    toolbox: Toolbox
    server: Server[Any]
    task: str
    calls: list[str] = field(default_factory=list)

    @property
    def task_uri(self) -> str:
        return f"lab://{self.lab.id}/task"


def lab_server(
    lab: Lab,
    seed: int = 0,
    *,
    env: Any = None,
    options: EnvOptions | None = None,
    model: ChatModel | None = None,
    ctx: RunContext | None = None,
) -> LabServer:
    """Wrap a lab's agent tools in an MCP server.

    Pass ``env`` and ``ctx`` to serve an environment that is already running (the in-process
    route used by ``--via-mcp``); otherwise a fresh environment is created from ``seed``.
    ``model`` is only needed by tools that make model calls themselves (vision).
    """
    if lab.spec is None:
        raise ValueError(f"Lab '{lab.id}' has no agent spec to serve.")
    spec = lab.spec
    env = env if env is not None else lab.make_env(seed, options or EnvOptions())
    ctx = ctx or RunContext(tracer=Tracer())
    box = spec.toolbox(env, model, ctx)
    task = spec.task(env)
    holder: dict[str, LabServer] = {}

    async def list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[types.Tool(name=s.name, description=s.description, input_schema=s.parameters) for s in box.specs()]
        )

    async def call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        holder["server"].calls.append(params.name)
        result = box.execute(ToolCall(id="mcp", name=params.name, arguments=dict(params.arguments or {})))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)], is_error=result.is_error
        )

    async def list_resources(_ctx: Any, _params: Any) -> types.ListResourcesResult:
        resource = types.Resource(
            name="task", uri=f"lab://{lab.id}/task", description=lab.title, mime_type="text/plain"
        )
        return types.ListResourcesResult(resources=[resource])

    async def read_resource(_ctx: Any, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        if str(params.uri) != f"lab://{lab.id}/task":
            raise ValueError(f"Unknown resource {params.uri}")
        contents = types.TextResourceContents(uri=str(params.uri), mime_type="text/plain", text=task)
        return types.ReadResourceResult(contents=[contents])

    server: Server[Any] = Server(
        f"aisyslab-{lab.id}",
        version="0.2.0",
        title=lab.title,
        instructions=f"{spec.system}\n\nTask:\n{task}",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )
    holder["server"] = LabServer(lab=lab, env=env, toolbox=box, server=server, task=task)
    return holder["server"]


async def serve_stdio(lab_server: LabServer) -> None:
    """Serve one host over stdin/stdout (how desktop hosts launch local servers)."""
    from mcp.server.stdio import stdio_server

    server = lab_server.server
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def http_app(lab_server: LabServer, *, host: str = "127.0.0.1") -> Any:
    """A Starlette app serving the lab over streamable HTTP at ``/mcp``."""
    return lab_server.server.streamable_http_app(host=host)
