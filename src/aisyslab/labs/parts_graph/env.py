"""A late part: which lines does it hold up, which orders are at risk, which can an alternate save?

The answer lies across five documents, and no single one connects the part to an order: the bill
of materials puts it in assemblies, the installation register and logbooks put those on machines,
the layout and changeover notes put machines on lines, the line plan names each line's products and
the order book lists the orders. Notes change what the tables say: a machine moved, an assembly
went to stores, an approval was withdrawn. Some only seem to: a move planned for next month, a
logbook naming an assembly a machine does not use.

The environment serves documents and a similarity search over them; the graph is built by whoever
solves the lab. The verifier compares three sets (lines, orders at risk, orders saved) and says how
many are missing or extra, never which.
"""

from __future__ import annotations

from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.retrieval import HybridIndex
from ...core.tools import ToolError
from ...core.verdict import Verdict
from .world import TODAY, build


def _compare(given: list[str], truth: list[str], noun: str) -> str:
    named = {g.strip().upper() for g in given if g.strip()}
    missing, extra = len(set(truth) - named), len(named - set(truth))
    if not missing and not extra:
        return "ok"
    parts = [f"{missing} {noun}{'s' if missing != 1 else ''} missing"] if missing else []
    parts += [f"{extra} listed that should not be"] if extra else []
    return ", ".join(parts)


class PartsGraphEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.world = build(seed)
        self.documents = {d.id: d for d in self.world.documents}
        self.index = HybridIndex([f"{d.title}\n{d.text}" for d in self.world.documents], embedder=options.embedder)

    @property
    def today(self) -> str:
        return TODAY.isoformat()

    def question(self) -> str:
        self._api("question")
        return self.world.question

    def list_documents(self) -> dict[str, Any]:
        self._api("list_documents")
        return {"today": self.today, "documents": [d.listing() for d in self.world.documents]}

    def read_document(self, doc_id: str) -> dict[str, str]:
        self._api("read_document")
        doc = self.documents.get(doc_id.strip())
        if doc is None:
            raise ToolError(f"No document '{doc_id}'.", hint="Use the ids from list_documents.")
        return doc.as_dict()

    def search(self, query: str, k: int = 5) -> dict[str, Any]:
        self._api("search")
        docs = self.world.documents
        hits = self.index.search(query, query, k=min(max(k, 1), 8))
        return {
            "results": [
                {"id": docs[h["index"]].id, "title": docs[h["index"]].title, "snippet": docs[h["index"]].text[:160]}
                for h in hits
            ]
        }

    def submit(self, lines: list[str], at_risk: list[str], saved: list[str]) -> Verdict:
        self._api("submit")
        a = self.world.answer
        checks = {
            "lines": _compare(lines, a["lines"], "line"),
            "at_risk": _compare(at_risk, a["at_risk"], "order"),
            "saved": _compare(saved, a["saved"], "order"),
        }
        both = {o.strip().upper() for o in at_risk} & {o.strip().upper() for o in saved}
        if both:
            checks["saved"] = f"{len(both)} order(s) listed as both at risk and saved"
        correct = sum(v == "ok" for v in checks.values())
        accepted = correct == len(checks)
        feedback = "All correct." if accepted else "; ".join(f"{k}: {v}" for k, v in checks.items() if v != "ok")
        details = {"score": round(correct / len(checks), 3), "checks": checks, "traps": list(self.world.traps)}
        return self._record(Verdict(accepted, "OK" if accepted else "PARTIAL", feedback, details))
