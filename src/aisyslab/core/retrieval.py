"""Hybrid retrieval without infrastructure: BM25 plus a vector ranking, fused with RRF.

Two rankings that fail differently beat one ranking tuned harder. BM25 matches exact terms
and numbers; a vector ranking tolerates rephrasing. Reciprocal Rank Fusion merges them by
rank, so their incomparable scores never need calibrating. The default embedder hashes
character trigrams and runs offline; any OpenAI-compatible embedding endpoint can replace it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its many may me must my of on or "
    "our the their this to was what when which who will with within you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class TrigramEmbedder:
    """Offline stand-in for an embedding model: hashed character trigrams, L2-normalised."""

    name = "trigram-hash-512"

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dims
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - 2):
                bucket = int(hashlib.md5(padded[i : i + 3].encode()).hexdigest()[:8], 16) % self.dims
                v[bucket] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class OpenAICompatibleEmbedder:
    """Embeddings from OpenAI or a local server (LM Studio, Ollama) via /v1/embeddings."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or ("local" if base_url else None))
        self.model = model
        self.name = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class HybridIndex:
    def __init__(self, texts: Sequence[str], embedder: Embedder | None = None) -> None:
        self.texts = list(texts)
        self.bm25 = BM25(self.texts)
        self.embedder: Embedder = embedder or TrigramEmbedder()
        self.vectors = self.embedder.embed(self.texts)

    def search(self, keywords: str, question: str | None = None, k: int = 5) -> list[dict[str, Any]]:
        """``keywords`` feed BM25; ``question`` (or the keywords) feeds the vector ranking."""
        lexical = self.bm25.scores(keywords)
        query_vector = self.embedder.embed([question or keywords])[0]
        semantic = [cosine(query_vector, v) for v in self.vectors]
        lexical_rank = [
            i for i in sorted(range(len(self.texts)), key=lambda i: lexical[i], reverse=True) if lexical[i] > 0
        ]
        semantic_rank = sorted(range(len(self.texts)), key=lambda i: semantic[i], reverse=True)
        fused = rrf([lexical_rank, semantic_rank])
        return [
            {"index": i, "rrf": round(score, 4), "bm25": round(lexical[i], 3), "cosine": round(semantic[i], 3)}
            for i, score in fused[:k]
        ]
