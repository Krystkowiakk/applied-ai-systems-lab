from __future__ import annotations

import math

from aisyslab.core.retrieval import BM25, HybridIndex, TrigramEmbedder, cosine, rrf

DOCS = [
    "Salary thresholds for the skilled worker permit.",
    "Register at the town hall within five days of arrival.",
    "Health insurance is mandatory after registration.",
    "Driving licence exchange fee for non-EU licences.",
]


def test_bm25_prefers_matching_terms() -> None:
    scores = BM25(DOCS).scores("licence exchange fee")
    assert max(range(len(DOCS)), key=lambda i: scores[i]) == 3
    assert scores[0] == 0


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = dict(rrf([[0, 1, 2], [1, 0, 2]]))
    assert fused[0] == fused[1] > fused[2]
    assert math.isclose(fused[2], 2 / 63)


def test_trigram_vectors_are_normalised_and_tolerate_typos() -> None:
    a, b, c = TrigramEmbedder().embed(["registration deadline", "registratoin deadline", "salary threshold"])
    assert math.isclose(cosine(a, a), 1.0, rel_tol=1e-9)
    assert cosine(a, b) > cosine(a, c)


def test_hybrid_search_finds_by_keywords_or_similarity() -> None:
    index = HybridIndex(DOCS)
    assert index.search("insurance mandatory")[0]["index"] == 2
    hits = index.search("zzz", question="how many days to register at the town hall")
    assert hits[0]["index"] == 1 and hits[0]["bm25"] == 0
