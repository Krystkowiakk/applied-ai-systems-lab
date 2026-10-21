"""A shared workspace: documents are the bus between agents, not their context windows.

Agents that work together pass files, not transcripts. One writes ``incident.md``, another
reads it; the orchestrator gets a reference (``workspace://incident.md``) instead of the
content until it decides to read it. Every write names the version it was based on, and a
write based on a stale version is refused (optimistic concurrency), so two agents cannot
silently overwrite each other's work.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..core.tools import NoArgs, Tool, ToolError

SCHEME = "workspace://"


@dataclass
class Document:
    text: str
    version: int
    author: str


def ref(path: str) -> str:
    return f"{SCHEME}{path}"


def _path(value: str) -> str:
    path = value.removeprefix(SCHEME).strip().lstrip("/")
    if not path or ".." in path.split("/"):
        raise ToolError(f"Invalid workspace path '{value}'.", hint="Use a relative name such as notes/incident.md.")
    return path


class Workspace:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._lock = threading.Lock()
        self.conflicts = 0

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"ref": ref(p), "version": d.version, "author": d.author, "chars": len(d.text)}
                for p, d in sorted(self._docs.items())
            ]

    def read(self, path: str) -> dict[str, Any]:
        key = _path(path)
        with self._lock:
            doc = self._docs.get(key)
        if doc is None:
            raise ToolError(f"No document {ref(key)}.", hint="List the workspace to see what exists.")
        return {"ref": ref(key), "version": doc.version, "author": doc.author, "text": doc.text}

    def write(self, path: str, text: str, *, author: str, expected_version: int = 0) -> dict[str, Any]:
        """Create (expected_version=0) or replace a document; refuse writes based on a stale version."""
        key = _path(path)
        with self._lock:
            current = self._docs.get(key)
            version = current.version if current else 0
            if version != expected_version:
                self.conflicts += 1
                raise ToolError(
                    f"Conflict on {ref(key)}: it is at version {version}, your write was based on {expected_version}.",
                    hint="Read it again, merge your change into the current text, then write with that version.",
                )
            self._docs[key] = Document(text, version + 1, author)
            return {"ref": ref(key), "version": version + 1}

    def tools(self, author: str) -> list[Tool]:
        """Tools for one agent; its writes are signed with its name."""
        return [
            Tool(
                "workspace_list",
                "Documents in the shared workspace with version and author.",
                NoArgs,
                lambda _: self.entries(),
            ),
            Tool(
                "workspace_read",
                "Read one document; note its version before you change it.",
                ReadArgs,
                lambda a: self.read(a.path),
            ),
            Tool(
                "workspace_write",
                "Create (expected_version 0) or replace a document. A stale version is refused.",
                WriteArgs,
                lambda a: self.write(a.path, a.text, author=author, expected_version=a.expected_version),
            ),
        ]


class ReadArgs(BaseModel):
    path: str = Field(description="e.g. incident.md or workspace://incident.md")


class WriteArgs(BaseModel):
    path: str
    text: str = Field(description="The full new content of the document.")
    expected_version: int = Field(default=0, ge=0, description="0 to create; otherwise the version you read.")
