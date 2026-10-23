"""Research on the web: a search API behind one interface, a polite crawler, checked citations."""

from .web import Crawler, OfflineSearch, OpenAISearch, Page, SearchAPI, canonical, check_citations

__all__ = ["Crawler", "OfflineSearch", "OpenAISearch", "Page", "SearchAPI", "canonical", "check_citations"]
