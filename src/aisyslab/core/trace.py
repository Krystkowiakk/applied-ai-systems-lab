"""Append-only run trace: every model call, tool call, retry and verdict as one JSON line.

The hierarchy mirrors what observability tools such as Langfuse expect (run -> step ->
generation / tool), so a JSONL trace can be exported later without changing call sites.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

PREVIEW_CHARS = 240


class Tracer:
    def __init__(self, path: Path | None = None, *, echo: Callable[[str], None] | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._echo = echo
        self._started = time.perf_counter()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        record = {"seq": len(self.events), "t": round(time.perf_counter() - self._started, 3), "kind": kind}
        record.update(data)
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self._echo is not None:
            self._echo(format_event(record))

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["kind"] == kind)


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_event(e: dict[str, Any]) -> str:
    kind = e["kind"]
    head = f"{e['t']:>8.3f}s  {kind:<10}"
    if kind == "generation":
        calls = ", ".join(e.get("tool_calls", [])) or "-"
        return f"{head} step {e['step']}: {e['input_tokens']}→{e['output_tokens']} tok, tools: {calls}"
    if kind == "tool":
        flag = " ERROR" if e.get("is_error") else ""
        return f"{head} {e['name']}({preview(json.dumps(e.get('arguments', {}), ensure_ascii=False), 80)}){flag} → {preview(e.get('result', ''), 100)}"
    if kind == "verdict":
        status = "ACCEPTED" if e["accepted"] else "rejected"
        return f"{head} {status} [{e.get('code', '')}] {preview(e.get('feedback', ''), 140)}"
    if kind == "retry":
        return f"{head} attempt {e['attempt']} failed ({e['reason']}), waiting {e['delay']:.2f}s"
    rest = {k: v for k, v in e.items() if k not in ("seq", "t", "kind")}
    return f"{head} {preview(json.dumps(rest, ensure_ascii=False, default=str), 160)}"


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_trace(events: Iterable[dict[str, Any]]) -> str:
    return "\n".join(format_event(e) for e in events)
