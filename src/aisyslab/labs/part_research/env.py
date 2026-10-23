"""Replacement part research: search, fetch politely, take notes with sources, report.

The environment is a small web behind a search API and a crawler. Notes can only cite pages that
were fetched. The report is written section by section from the notes, and its citations are
checked in code. The verifier grades the answer (the fastest compatible replacement, its current
lead time, its current pressure rating, the incompatible part rejected) and the report.
"""

from __future__ import annotations

import re
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict
from ...research.web import Crawler, OfflineSearch, canonical, check_citations
from .corpus import QUESTION, ROBOTS, build

PART = re.compile(r"\b((?:VT|HX)-\d{2,3})\b", re.IGNORECASE)


def parts_in(text: str) -> list[str]:
    """Part numbers named in a free-text answer ("Valvetec VT-210, 24 V" -> ["VT-210"])."""
    return list(dict.fromkeys(m.upper() for m in PART.findall(text)))


class SimClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class PartResearchEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.pages, self.answer = build(seed)
        self.search_api = OfflineSearch([p for p in self.pages if self._crawlable(p.url)])
        clock = SimClock()
        self.clock = clock
        self.crawler = Crawler(
            {canonical(p.url): p for p in self.pages},
            ROBOTS,
            clock=clock,
            sleep=clock.sleep,
            fail=lambda _url: bool(options.flaky) and self._chaos.random() < options.flaky,
        )
        self.notes: list[dict[str, str]] = []
        self.report: dict[str, Any] | None = None

    def _crawlable(self, url: str) -> bool:
        return Crawler({}, ROBOTS, clock=lambda: 0.0, sleep=lambda _s: None).allowed(canonical(url))

    def question(self) -> str:
        self._api("question")
        return QUESTION

    def search(self, query: str, k: int = 5) -> dict[str, Any]:
        self._api("search")
        return {"results": self.search_api.search(query, k=min(max(k, 1), 8))}

    def fetch(self, url: str) -> dict[str, Any]:
        self._api("fetch")
        return self.crawler.fetch(url)

    def note(self, url: str, fact: str) -> dict[str, Any]:
        self._api("note")
        if canonical(url) not in self.crawler.fetched:
            raise ToolError("Notes cite pages you fetched; fetch it first.")
        self.notes.append({"url": canonical(url), "fact": fact.strip()[:300]})
        return {"note": len(self.notes), "notes": len(self.notes)}

    def store_report(self, text: str, sources: list[str]) -> dict[str, Any]:
        problems = check_citations(text, sources, self.crawler.fetched)
        if problems:
            raise ToolError("The report's citations do not check out: " + "; ".join(problems[:5]))
        self.report = {"text": text, "sources": sources}
        return {"stored": True, "sources": len(sources), "words": len(text.split())}

    def submit(
        self, replacement: str, lead_time_weeks: int, max_pressure_bar: int, rejected: list[dict[str, str]]
    ) -> Verdict:
        self._api("submit")
        a = self.answer
        checks: dict[str, str] = {}
        named_parts = parts_in(replacement)
        part = named_parts[0] if len(named_parts) == 1 else ""
        if not named_parts:
            checks["replacement"] = "name the part number of the replacement"
        elif len(named_parts) > 1:
            checks["replacement"] = f"name one part, not {len(named_parts)}"
        else:
            checks["replacement"] = "ok" if part == a["replacement"] else "not the fastest compatible replacement"
        if part == a["replacement"]:
            if lead_time_weeks == a["lead_time_weeks"]:
                checks["lead_time"] = "ok"
            elif lead_time_weeks == a["stale_lead_time"]:
                checks["lead_time"] = "outdated: that lead time is from an archived page"
            else:
                checks["lead_time"] = "wrong lead time"
            if max_pressure_bar == a["max_pressure_bar"]:
                checks["pressure"] = "ok"
            elif max_pressure_bar == a["stale_pressure"]:
                checks["pressure"] = "outdated: that rating is from a superseded datasheet"
            else:
                checks["pressure"] = "wrong pressure rating"
        else:
            checks["lead_time"] = checks["pressure"] = "not graded: wrong replacement"
        named = {p for item in rejected for p in parts_in(f"{item.get('part', '')}")}
        checks["rejected"] = (
            "ok" if a["must_reject"] in named else "an incompatible part someone recommends was not rejected"
        )
        if self.report is None:
            checks["report"] = "no report written"
        elif len(set(self.report["sources"])) < 3:
            checks["report"] = "the report cites fewer than three sources"
        else:
            checks["report"] = "ok"
        correct = sum(v == "ok" for v in checks.values())
        accepted = correct == len(checks)
        feedback = "All correct." if accepted else "; ".join(f"{k}: {v}" for k, v in checks.items() if v != "ok")
        details = {
            "score": round(correct / len(checks), 3),
            "checks": checks,
            "pages_fetched": len(self.crawler.fetched),
        }
        return self._record(Verdict(accepted, "OK" if accepted else "PARTIAL", feedback, details))
