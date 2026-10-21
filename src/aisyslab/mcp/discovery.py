"""Progressive tool discovery: two tools instead of a list.

With tools merged from several servers, sending every schema in every request costs tokens
and attention, and models choose worse from long lists. Here the model gets two tools:
``search_tools`` finds candidates by what they do (hybrid search over names and
descriptions) and returns their exact argument schemas; ``call_tool`` runs one by name. The
underlying toolbox still validates the arguments and runs the call, so nothing is lost in
safety; only the list the model reads at each step gets shorter.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.messages import ToolCall
from ..core.retrieval import HybridIndex
from ..core.tools import Tool, Toolbox, ToolError


class SearchToolsArgs(BaseModel):
    query: str = Field(description="What you want to do, in a few words, e.g. 'read a database table'.")
    limit: int = Field(default=5, ge=1, le=10)


class CallToolArgs(BaseModel):
    tool: str = Field(description="Exact tool name returned by search_tools.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments matching the tool's schema.")


def discovery_toolbox(inner: Toolbox, *, max_output_chars: int = 6000) -> Toolbox:
    """Wrap a toolbox so the model searches for tools before calling them."""
    specs = {s.name: s for s in inner.specs()}
    names = list(specs)
    index = HybridIndex([f"{name.replace('__', ' ').replace('_', ' ')}: {specs[name].description}" for name in names])

    def search(args: SearchToolsArgs) -> dict[str, Any]:
        hits = index.search(args.query, args.query, k=args.limit)
        return {
            "tools": [
                {
                    "name": names[h["index"]],
                    "description": specs[names[h["index"]]].description,
                    "schema": specs[names[h["index"]]].parameters,
                }
                for h in hits
            ],
            "hint": "Call one with call_tool(tool, arguments). Search again if none fits.",
        }

    def call(args: CallToolArgs) -> Any:
        if args.tool not in specs:
            raise ToolError(f"No tool named '{args.tool}'.", hint="Use search_tools to find the exact name.")
        result = inner.execute(ToolCall(id="discovered", name=args.tool, arguments=args.arguments))
        if result.is_error:
            raise ToolError.from_payload(result.content)
        return result.content

    return Toolbox(
        [
            Tool(
                "search_tools",
                "Find tools by what they do; returns names and argument schemas.",
                SearchToolsArgs,
                search,
            ),
            Tool("call_tool", "Call a tool found with search_tools.", CallToolArgs, call),
        ],
        max_output_chars=max_output_chars,
    )
