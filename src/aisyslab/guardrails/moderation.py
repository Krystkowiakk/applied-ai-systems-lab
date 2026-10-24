"""Moderation before anything reaches a provider: a moderation model, with thresholds per category.

A flagged message is not sent to a chat model at all; it goes to a person. Thresholds are set per
category because the cost of a miss differs: a threat should be caught at a lower score than
plain rudeness. ``flagged`` makes any moderator usable as the gateway's moderation hook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core.model import ModelError

DEFAULT_THRESHOLDS = {  # every category the moderation model reports; a sub-category is never laxer than its parent
    "harassment": 0.5,
    "harassment/threatening": 0.2,
    "hate": 0.3,
    "hate/threatening": 0.2,
    "illicit": 0.5,
    "illicit/violent": 0.3,
    "violence": 0.4,
    "violence/graphic": 0.4,
    "self-harm": 0.2,
    "self-harm/intent": 0.2,
    "self-harm/instructions": 0.2,
    "sexual": 0.5,
    "sexual/minors": 0.1,
}


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    categories: dict[str, float] = field(default_factory=dict)  # the categories over their threshold
    model: str = ""


def over(scores: Mapping[str, float], thresholds: Mapping[str, float], default: float) -> dict[str, float]:
    return {c: round(s, 3) for c, s in scores.items() if s >= thresholds.get(c, default)}


class OpenAIModeration:
    """OpenAI's moderation endpoint (free of charge at the time of writing)."""

    def __init__(
        self,
        model: str = "omni-moderation-latest",
        thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
        default_threshold: float = 0.5,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        self._openai = openai
        self._client = openai.OpenAI(max_retries=3)
        self.model = model
        self.thresholds = dict(thresholds)
        self.default_threshold = default_threshold

    def check(self, text: str) -> ModerationResult:
        try:
            response = self._client.moderations.create(model=self.model, input=text)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        scores = response.results[0].category_scores.model_dump(by_alias=True)
        hits = over({k: float(v or 0) for k, v in scores.items()}, self.thresholds, self.default_threshold)
        return ModerationResult(bool(hits), hits, getattr(response, "model", self.model))

    def flagged(self, text: str) -> list[str]:
        return sorted(self.check(text).categories)


class KeywordModeration:
    """Offline stand-in: a score of 1.0 for a category when one of its phrases appears."""

    def __init__(self, phrases: Mapping[str, Sequence[str]]) -> None:
        self.phrases = {c: [p.lower() for p in ps] for c, ps in phrases.items()}

    def check(self, text: str) -> ModerationResult:
        lowered = text.lower()
        hits = {c: 1.0 for c, ps in self.phrases.items() if any(p in lowered for p in ps)}
        return ModerationResult(bool(hits), hits, "keywords")

    def flagged(self, text: str) -> list[str]:
        return sorted(self.check(text).categories)
