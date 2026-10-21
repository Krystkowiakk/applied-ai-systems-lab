"""An MCP host for the synchronous agent loop.

The host connects to several MCP servers at once and merges their tools into one toolbox.
Tool names get the server's prefix (``plant__run_query``), because two servers may both offer
``search`` and the model must not confuse them. The host also audits what each server offers
(tools, resources) before any of it reaches a model.

MCP clients are asynchronous; the agent loop is not. The host runs one event loop in a
background thread and keeps every client session inside a single long-lived task, which is
what the SDK's cancel scopes require. Tool calls cross the thread boundary through a queue.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import Client

from ..core.tools import Tool, Toolbox, ToolError

SEPARATOR = "__"


@dataclass(frozen=True)
class ServerAudit:
    """What one server exposes, recorded before its tools are given to a model."""

    prefix: str
    server_name: str
    tools: list[str]
    resources: list[str]
    instructions: str | None


class McpHost:
    """Connect to MCP servers (in-process ``Server``, stdio parameters or an HTTP URL)."""

    def __init__(self, servers: Mapping[str, Any], *, timeout: float = 60.0) -> None:
        bad = [p for p in servers if not p.replace("_", "").isalnum()]
        if bad:
            raise ValueError(f"Server prefixes must be alphanumeric or '_': {bad}")
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-host", daemon=True)
        self._thread.start()
        self._requests: asyncio.Queue[Any] | None = None
        ready: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        self._session = asyncio.run_coroutine_threadsafe(self._run(dict(servers), ready), self._loop)
        try:
            self._listing = ready.result(timeout)
        except Exception:
            self._stop_loop()
            raise
        self.calls: list[str] = []

    async def _run(self, servers: dict[str, Any], ready: concurrent.futures.Future[dict[str, Any]]) -> None:
        self._requests = asyncio.Queue()
        try:
            async with AsyncExitStack() as stack:
                clients = {p: await stack.enter_async_context(Client(target)) for p, target in servers.items()}
                listing: dict[str, Any] = {}
                for prefix, client in clients.items():
                    tools = (await client.list_tools()).tools
                    try:
                        resources = [str(r.uri) for r in (await client.list_resources()).resources]
                    except Exception:  # a server without resources is fine
                        resources = []
                    info = client.server_info
                    listing[prefix] = {
                        "tools": tools,
                        "audit": ServerAudit(
                            prefix=prefix,
                            server_name=info.name if info else prefix,
                            tools=[t.name for t in tools],
                            resources=resources,
                            instructions=getattr(info, "instructions", None),
                        ),
                    }
                ready.set_result(listing)
                while True:
                    item = await self._requests.get()
                    if item is None:
                        break
                    kind, prefix, payload, done = item
                    try:
                        if kind == "call":
                            name, arguments = payload
                            done.set_result(await clients[prefix].call_tool(name, arguments))
                        else:
                            done.set_result(await clients[prefix].read_resource(payload))
                    except Exception as exc:
                        done.set_exception(exc)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    # ----- what the servers offer -------------------------------------------------------------
    def audit(self) -> list[ServerAudit]:
        return [entry["audit"] for entry in self._listing.values()]

    def toolbox(self, *, max_output_chars: int = 6000) -> Toolbox:
        tools = []
        for prefix, entry in self._listing.items():
            for remote in entry["tools"]:
                tools.append(
                    Tool(
                        name=f"{prefix}{SEPARATOR}{remote.name}",
                        description=remote.description or "",
                        args=None,
                        schema=dict(remote.input_schema),
                        handler=self._caller(prefix, remote.name),
                    )
                )
        return Toolbox(tools, max_output_chars=max_output_chars)

    def _caller(self, prefix: str, name: str) -> Callable[[dict[str, Any]], Any]:
        return lambda arguments: self.call(prefix, name, arguments)

    def read_resource(self, prefix: str, uri: str) -> str:
        result = self._submit("read", prefix, uri)
        return "\n".join(getattr(c, "text", "") for c in result.contents)

    # ----- calls ----------------------------------------------------------------------------
    def call(self, prefix: str, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on one server; MCP error results become ToolErrors the model can read."""
        self.calls.append(f"{prefix}{SEPARATOR}{name}")
        result = self._submit("call", prefix, (name, arguments))
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            raise ToolError.from_payload(text)
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _submit(self, kind: str, prefix: str, payload: Any) -> Any:
        if prefix not in self._listing:
            raise ToolError(f"No MCP server '{prefix}'.", hint=f"Servers: {', '.join(self._listing)}.")
        done: concurrent.futures.Future[Any] = concurrent.futures.Future()
        requests = self._requests
        assert requests is not None
        self._loop.call_soon_threadsafe(requests.put_nowait, (kind, prefix, payload, done))
        return done.result(self._timeout)

    # ----- lifecycle --------------------------------------------------------------------------
    def close(self) -> None:
        if self._requests is not None and not self._session.done():
            self._loop.call_soon_threadsafe(self._requests.put_nowait, None)
            self._session.result(self._timeout)
        self._stop_loop()

    def _stop_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(self._timeout)
        if not self._loop.is_closed():
            self._loop.close()

    def __enter__(self) -> McpHost:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
