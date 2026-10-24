"""A tamper-evident audit log: every entry signed and chained to the one before.

Each entry carries the hash of the previous one and an HMAC of its own content, so changing,
removing or reordering an entry breaks verification from that point on. Entries cut from the end
leave a shorter chain that still verifies; ``head`` (the count and the last hash) kept somewhere
else, and passed to ``verify``, catches that too. Model inputs and outputs
are logged as hashes and lengths, not as text: the log proves what happened without becoming a
second copy of the personal data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable
from typing import Any


def fingerprint(text: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "chars": len(text)}


class AuditLog:
    def __init__(self, key: bytes, *, clock: Callable[[], float] = time.time) -> None:
        self._key = key
        self._clock = clock
        self.entries: list[dict[str, Any]] = []

    def append(self, kind: str, **data: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "seq": len(self.entries),
            "at": round(self._clock(), 3),
            "kind": kind,
            "data": data,
            "prev": self.entries[-1]["hash"] if self.entries else "",
        }
        entry["hash"] = self._hash(entry)
        entry["sig"] = hmac.new(self._key, entry["hash"].encode(), hashlib.sha256).hexdigest()
        self.entries.append(entry)
        return entry

    def head(self) -> tuple[int, str]:
        """The number of entries and the last hash: store it outside the log (a ticket, another system)."""
        return len(self.entries), self.entries[-1]["hash"] if self.entries else ""

    def verify(self, head: tuple[int, str] | None = None) -> tuple[bool, str]:
        if head is not None and (
            len(self.entries) < head[0] or (head[0] and self.entries[head[0] - 1]["hash"] != head[1])
        ):
            return False, f"the log no longer holds the {head[0]} entries recorded elsewhere"
        prev = ""
        for index, entry in enumerate(self.entries):
            body = {k: v for k, v in entry.items() if k not in ("hash", "sig")}
            if entry["seq"] != index or body["prev"] != prev:
                return False, f"entry {index} is out of order or does not follow the previous one"
            if self._hash(body) != entry["hash"]:
                return False, f"entry {index} was changed after it was written"
            expected = hmac.new(self._key, entry["hash"].encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, entry["sig"]):
                return False, f"entry {index} carries a signature this key did not make"
            prev = entry["hash"]
        return True, f"{len(self.entries)} entries verified"

    @staticmethod
    def _hash(entry: dict[str, Any]) -> str:
        body = {k: v for k, v in entry.items() if k not in ("hash", "sig")}
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
