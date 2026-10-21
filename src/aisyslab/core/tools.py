"""Tools as the agent's API: typed arguments, errors returned as data, bounded output."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .messages import ToolCall, ToolSpec

TRUNCATION_HINT = (
    "Output truncated. Narrow the query instead of retrying it: add filters, lower the limit, or ask for the next page."
)


class ToolError(Exception):
    """An expected failure the model should see and recover from.

    Raised for bad input, missing records or policy refusals. The toolbox returns it to
    the model as a structured error result; it never ends the run.
    """

    def __init__(self, message: str, *, hint: str | None = None, **data: Any) -> None:
        super().__init__(message)
        self.hint = hint
        self.data = data

    @classmethod
    def from_payload(cls, text: str) -> ToolError:
        """Rebuild the error from the JSON payload a toolbox (or an MCP server) returned."""
        try:
            payload = json.loads(text)
        except ValueError:
            return cls(text)
        if isinstance(payload, dict) and "error" in payload:
            data = dict(payload)
            message = str(data.pop("error"))
            hint = data.pop("hint", None)
            return cls(message, hint=hint, **data)
        return cls(text)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel] | None
    handler: Callable[[Any], Any]
    # A raw JSON schema instead of a pydantic model, for tools defined elsewhere (an MCP
    # server validates its own arguments); the handler then receives the argument dict.
    schema: dict[str, Any] | None = None

    def spec(self) -> ToolSpec:
        if self.args is None:
            return ToolSpec(self.name, self.description, self.schema or {"type": "object", "properties": {}})
        return ToolSpec(self.name, self.description, _clean_schema(_inline_refs(self.args.model_json_schema())))


def tool(name: str, description: str, args: type[BaseModel]) -> Callable[[Callable[[Any], Any]], Tool]:
    """Decorator turning a handler into a ``Tool``; the handler receives the validated model."""

    def wrap(handler: Callable[[Any], Any]) -> Tool:
        return Tool(name=name, description=description, args=args, handler=handler)

    return wrap


class NoArgs(BaseModel):
    """Argument model for tools that take no parameters."""


@dataclass
class ToolResult:
    content: str
    is_error: bool
    truncated: bool
    duration_ms: float
    internal_error: bool = False


class Toolbox:
    def __init__(self, tools: Iterable[Tool], *, max_output_chars: int = 6000) -> None:
        self._tools = {t.name: t for t in tools}
        self.max_output_chars = max_output_chars

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()

        def done(content: str, *, error: bool, internal: bool = False) -> ToolResult:
            text, truncated = bounded(content, self.max_output_chars)
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(text, error, truncated, elapsed, internal_error=internal)

        tool = self._tools.get(call.name)
        if tool is None:
            return done(
                _error(f"Unknown tool '{call.name}'.", hint=f"Available: {', '.join(self._tools)}."), error=True
            )
        if call.parse_error:
            return done(_error(f"Arguments were not valid JSON: {call.parse_error}"), error=True)
        try:
            args = call.arguments if tool.args is None else tool.args.model_validate(call.arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in exc.errors()
            )
            return done(_error(f"Invalid arguments for {call.name}: {problems}"), error=True)
        try:
            value = tool.handler(args)
        except ToolError as exc:
            return done(_error(str(exc), hint=exc.hint, **exc.data), error=True)
        except Exception as exc:  # a bug in a tool must not take the whole run down
            return done(_error(f"Internal tool failure ({type(exc).__name__}): {exc}"), error=True, internal=True)
        return done(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), error=False)


def bounded(text: str, limit: int) -> tuple[str, bool]:
    """Cap tool output so one call cannot flood the context window."""
    if len(text) <= limit:
        return text, False
    return f"{text[:limit]}\n...[{len(text) - limit} more characters. {TRUNCATION_HINT}]", True


def _error(message: str, *, hint: str | None = None, **data: Any) -> str:
    payload: dict[str, Any] = {"error": message}
    if hint:
        payload["hint"] = hint
    payload.update(data)
    return json.dumps(payload, ensure_ascii=False)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace "$ref" pointers with their definitions; not every server resolves them."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(defs[ref.split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _clean_schema(node: Any) -> Any:
    """Drop pydantic's cosmetic titles and close objects to extra keys."""
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title":
                continue
            if key in ("properties", "$defs"):  # keys here are field names, not keywords
                cleaned[key] = {name: _clean_schema(sub) for name, sub in value.items()}
            else:
                cleaned[key] = _clean_schema(value)
        if cleaned.get("type") == "object":
            cleaned.setdefault("properties", {})
            cleaned["additionalProperties"] = False
        return cleaned
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node
