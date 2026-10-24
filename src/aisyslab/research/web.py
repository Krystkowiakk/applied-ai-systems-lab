"""Search and fetch: a search API behind one interface, and a crawler that behaves.

``SearchAPI`` is what a research loop calls; ``OfflineSearch`` answers from a fixed set of pages,
so a lab can be graded, and ``OpenAISearch`` asks a search model on the real web. ``Crawler``
fetches pages the way a polite client should: robots.txt first, a minimum interval per host that
it keeps by waiting, transient errors left to the caller's retries, a page budget, and every URL
counted once.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from ..core.model import ModelError
from ..core.resilience import TransientError
from ..core.retrieval import HybridIndex
from ..core.tools import ToolError


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    links: tuple[str, ...] = ()


class SearchAPI(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict[str, str]]: ...


class OfflineSearch:
    """A search engine over a fixed set of pages: hybrid ranking, snippets, no network."""

    def __init__(self, pages: Sequence[Page]) -> None:
        self.pages = list(pages)
        self.index = HybridIndex([f"{p.title}. {p.text}" for p in self.pages])

    def search(self, query: str, k: int = 5) -> list[dict[str, str]]:
        hits = self.index.search(query, query, k=k)
        return [
            {
                "url": self.pages[h["index"]].url,
                "title": self.pages[h["index"]].title,
                "snippet": self.pages[h["index"]].text[:180],
            }
            for h in hits
            if h["bm25"] > 0 or h["cosine"] > 0.15
        ]


class OpenAISearch:
    """A search model on the real web; the pages it cites become the results."""

    def __init__(self, model: str = "gpt-5-search-api") -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        self._openai = openai
        self._client = openai.OpenAI(max_retries=3)
        self.model = model

    def search(self, query: str, k: int = 5) -> list[dict[str, str]]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                web_search_options={},
                messages=[{"role": "user", "content": f"Find web pages about: {query}. Cite each page you use."}],
            )
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        message = response.choices[0].message
        text = message.content or ""
        results: list[dict[str, str]] = []
        for annotation in message.annotations or []:
            cite = annotation.url_citation
            if all(r["url"] != cite.url for r in results):
                results.append(
                    {"url": cite.url, "title": cite.title, "snippet": text[cite.start_index : cite.end_index][:180]}
                )
        return results[:k]


@dataclass
class Crawler:
    pages: dict[str, Page]
    robots: dict[str, list[str]]  # host -> disallowed path prefixes
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    min_interval: float = 1.0  # seconds between two requests to one host: the crawler waits itself
    max_pages: int = 30
    fail: Callable[[str], bool] = lambda _url: False  # injects transient failures (chaos)
    fetched: list[str] = field(default_factory=list)
    waited: float = 0.0
    _last: dict[str, float] = field(default_factory=dict)

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()  # not netloc: a port or user info must not change the host
        return not any(parsed.path.startswith(prefix) for prefix in self.robots.get(host, []))

    def fetch(self, url: str) -> dict[str, Any]:
        # robots.txt is checked on the path as asked and as stored ("/private/" and "/private").
        if not (self.allowed(url) and self.allowed(canonical(url))):
            raise ToolError("robots.txt disallows this path; the crawler does not fetch it.")
        url = canonical(url)
        if url not in self.fetched and len(self.fetched) >= self.max_pages:
            raise ToolError(f"Page budget of {self.max_pages} used up; write with what you have.")
        host = urlparse(url).hostname or ""
        wait = self.min_interval - (self.clock() - self._last.get(host, -1e9))
        if wait > 0:  # polite: slow down per host instead of being told to
            self.sleep(wait)
            self.waited += wait
        self._last[host] = self.clock()
        if self.fail(url):
            raise TransientError(503, f"{host}: service unavailable")
        page = self.pages.get(url)
        if page is None:
            raise ToolError(f"404: {url} does not exist.", hint="Use URLs from search results or page links.")
        if url not in self.fetched:
            self.fetched.append(url)
        return {"url": page.url, "title": page.title, "text": page.text, "links": list(page.links)}


def canonical(url: str) -> str:
    """One form per page: lower-case host without user info or a default port, no fragment, no
    trailing slash; the query stays, since ``?id=1`` and ``?id=2`` are different pages."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host if port in (None, {"https": 443, "http": 80}.get(scheme)) else f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


CITATION = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")  # [3], [1, 7], [2-4]


def check_citations(report: str, sources: Sequence[str], fetched: Sequence[str]) -> list[str]:
    """Problems with a report's citations: numbers without a source, sources never fetched."""
    cited: set[int] = set()
    for group in CITATION.findall(report):
        for part in re.split(r"\s*,\s*", group):
            bounds = [int(b) for b in re.split(r"\s*[\u2013-]\s*", part)]
            cited.update(range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 else bounds)
    problems = [f"[{n}] has no source" for n in sorted(cited) if not 1 <= n <= len(sources)]
    problems += [
        f"source {i} was never fetched: {url}" for i, url in enumerate(sources, 1) if canonical(url) not in fetched
    ]
    return problems
