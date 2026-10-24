"""Search, a polite crawler, checked citations, and the part-research lab."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace as NS

import pytest

from aisyslab.core.lab import EnvOptions, RunContext
from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn, tool_turn
from aisyslab.core.resilience import RetryPolicy, TransientError
from aisyslab.core.runner import run_once
from aisyslab.core.tools import ToolError
from aisyslab.core.trace import Tracer
from aisyslab.labs import LABS
from aisyslab.labs.part_research import baseline
from aisyslab.labs.part_research.env import PartResearchEnv, SimClock
from aisyslab.research import Crawler, OfflineSearch, OpenAISearch, Page, canonical, check_citations

PAGES = [
    Page("https://a.example/one", "One", "VT-210 lead time 6 weeks.", ("https://a.example/two",)),
    Page("https://a.example/two", "Two", "VT-210 max pressure 10 bar."),
    Page("https://a.example/private/x", "Private", "Not for crawlers."),
]


def crawler(**kwargs: object) -> tuple[Crawler, SimClock]:
    clock = SimClock()
    pages = {canonical(p.url): p for p in PAGES}
    return Crawler(pages, {"a.example": ["/private/"]}, clock=clock, sleep=clock.sleep, **kwargs), clock  # type: ignore[arg-type]


def test_the_crawler_keeps_to_robots_waits_per_host_and_has_a_budget() -> None:
    c, clock = crawler(max_pages=2)
    with pytest.raises(ToolError, match=r"robots\.txt"):
        c.fetch("https://a.example/private/x")
    c.fetch("https://A.example/one/")
    c.fetch("https://a.example/two#top")
    assert c.fetched == ["https://a.example/one", "https://a.example/two"] and clock.now == pytest.approx(1.0)
    assert c.waited == pytest.approx(1.0)  # the second request to the host waited its interval
    with pytest.raises(ToolError, match="budget"):
        c.fetch("https://a.example/three")
    flaky, _ = crawler(fail=lambda _url: True)
    with pytest.raises(TransientError):
        flaky.fetch("https://a.example/one")


def test_search_and_citations() -> None:
    hits = OfflineSearch(PAGES).search("VT-210 max pressure")
    assert hits[0]["url"] == "https://a.example/two"
    fetched = ["https://a.example/one"]
    assert check_citations("Six weeks [1].", ["https://a.example/one"], fetched) == []
    problems = check_citations(
        "Six weeks [1], ten bar [3].", ["https://a.example/one", "https://a.example/two"], fetched
    )
    assert problems == ["[3] has no source", "source 2 was never fetched: https://a.example/two"]


def test_the_web_search_adapter_turns_citations_into_results() -> None:
    search = OpenAISearch.__new__(OpenAISearch)  # no SDK client: the transport is faked
    search.model = "search"
    search._openai = NS(APIError=type("APIError", (Exception,), {}))
    cite = lambda url, a, b: NS(url_citation=NS(url=url, title=url[-3:], start_index=a, end_index=b))  # noqa: E731
    message = NS(
        content="Valve A replaces valve B.",
        annotations=[cite("https://x/1", 0, 7), cite("https://x/1", 0, 7), cite("https://x/2", 8, 24)],
    )
    search._client = NS(chat=NS(completions=NS(create=lambda **_: NS(choices=[NS(message=message)]))))
    assert [r["url"] for r in search.search("valve")] == ["https://x/1", "https://x/2"]
    assert search.search("valve")[0]["snippet"] == "Valve A"


def test_the_lab_grades_the_answer_and_the_report() -> None:
    env = PartResearchEnv(0, EnvOptions())
    with pytest.raises(ToolError, match="fetch it first"):
        env.note("https://fluidparts.example/stock/vt-210", "6 weeks")
    with pytest.raises(ToolError, match=r"robots\.txt"):
        env.fetch("https://valvetec.example/internal/pricing")
    for url in (
        "https://fluidparts.example/stock/vt-210",
        "https://valvetec.example/datasheets/vt-210",
        "https://hydrax.example/products/hx-24",
    ):
        env.fetch(url)
    with pytest.raises(ToolError, match="citations"):
        env.store_report("Fine [4].", ["https://fluidparts.example/stock/vt-210"])
    env.store_report("A [1]. B [2]. C [3].", list(env.crawler.fetched))
    a = env.answer
    stale = env.submit(a["replacement"], a["stale_lead_time"] or 0, a["stale_pressure"] or 0, [])
    if a["replacement"] == "VT-210":
        assert stale.details["checks"]["lead_time"].startswith("outdated") and stale.details["checks"][
            "pressure"
        ].startswith("outdated")
    assert "rejected" in stale.feedback
    right = env.submit(
        a["replacement"], a["lead_time_weeks"], a["max_pressure_bar"], [{"part": "VT-150", "reason": "12 V coil"}]
    )
    assert right.accepted
    # A correct part named inside a sentence is still the correct part (a live run was misled here).
    wordy = f"Valvetec {a['replacement']} (24 V DC, G1/4), the fastest at {a['lead_time_weeks']} weeks"
    rejected = [{"part": "VT-150 (12 V DC coil)", "reason": "wrong voltage"}]
    assert env.submit(wordy, a["lead_time_weeks"], a["max_pressure_bar"], rejected).accepted
    both = env.submit("VT-210 or HX-24", a["lead_time_weeks"], a["max_pressure_bar"], rejected)
    assert both.details["checks"]["replacement"] == "name one part, not 2"


@pytest.mark.parametrize("seed", range(3))
def test_the_first_hit_leads_code_to_the_forum_answer(seed: int) -> None:
    env = PartResearchEnv(seed, EnvOptions(flaky=0.2))
    baseline.solve(env, RunContext(tracer=Tracer(), retry=RetryPolicy(sleep=lambda _s: None)))
    verdict = env.submissions[-1]
    assert not verdict.accepted and verdict.details["checks"]["replacement"] != "ok"


def test_the_agent_researches_notes_and_reports() -> None:
    probe = PartResearchEnv(1, EnvOptions())
    a = probe.answer
    urls = [
        "https://valvetec.example/notices/pcn-2026-03",
        "https://fluidparts.example/stock/vt-210",
        "https://fluidparts.example/stock/hx-24",
        "https://hydrax.example/products/hx-24",
        "https://valvetec.example/datasheets/vt-210",
        "https://valvetec.example/datasheets/vt-150",
    ]
    plan = [tool_turn("question"), tool_turn("web_search", query="VT-200 replacement")]
    plan += [tool_turn("fetch", url=u) for u in urls]
    plan += [tool_turn("note", url=u, fact=f"Fact from {u}.") for u in urls]
    plan += [tool_turn("write_report", title="VT-200 replacement", outline=["Answer", "Rejected parts"])]
    plan += [
        tool_turn(
            "submit_answer",
            replacement=a["replacement"],
            lead_time_weeks=a["lead_time_weeks"],
            max_pressure_bar=a["max_pressure_bar"],
            rejected=[{"part": "VT-150", "reason": "12 V DC coil"}],
        )
    ]

    def policy(system: str, messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        if system.startswith("You write one section"):
            return text_turn("The successor is confirmed [1] and its lead time is listed [2].")
        return plan.pop(0) if plan else text_turn("Done.")

    report = run_once(LABS["part-research"], "agent", 1, model=ScriptedModel(policy))
    assert report.accepted, report.error
    assert report.model_calls == 2 + len(urls) * 2 + 2 + 2  # agent turns plus one call per report section


def test_store_report_rejects_sources_never_fetched() -> None:
    env = PartResearchEnv(2, EnvOptions())
    with pytest.raises(ToolError, match="never fetched"):
        env.store_report("A [1].", ["https://hydrax.example/products/hx-24"])
    assert json.loads(json.dumps(env.search("HX-24 lead time")))["results"]


def test_a_port_user_info_or_trailing_slash_does_not_get_past_robots() -> None:
    c, _clock = crawler()
    for sneaky in ("https://a.example:443/private/x", "https://u@a.example/private/x", "https://a.example/private/"):
        with pytest.raises(ToolError, match=r"robots\.txt"):
            c.fetch(sneaky)
    assert canonical("https://U@A.example:443/one/?id=2#top") == "https://a.example/one?id=2"
    assert canonical("http://a.example:8080/one") == "http://a.example:8080/one"
    c.fetch("https://a.example:443/one")
    c.fetch("https://a.example/two")
    assert c.waited == pytest.approx(1.0)  # one host, however the URL spelled it


def test_grouped_and_ranged_citations_are_checked_too() -> None:
    sources = ["https://a.example/one", "https://a.example/two"]
    fetched = list(sources)
    assert check_citations("A [1, 2]. B [1-2].", sources, fetched) == []
    assert check_citations("A [1, 7]. B [2-3].", sources, fetched) == ["[3] has no source", "[7] has no source"]
