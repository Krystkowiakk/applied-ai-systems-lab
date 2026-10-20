"""Deterministic token estimate.

Real tokenizers differ per provider. Budgets inside the simulated environments use this
single estimate so that the verifier, the tools and the solvers all agree on one number.
Provider-reported usage is still what the benchmark records for cost.
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Roughly one token per word piece or punctuation mark, never less than chars/4."""
    if not text:
        return 0
    pieces = sum(max(1, math.ceil(len(tok) / 6)) for tok in _WORD.findall(text))
    return max(pieces, math.ceil(len(text) / 4))
