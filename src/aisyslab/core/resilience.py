"""Retries for flaky upstream APIs.

Most agent failures in practice are integration failures, not model failures: a 503 or a
429 that nobody retried. Transient errors are handled here, in code, and never handed to
the model as if they were a fact about the task.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """An upstream failure worth retrying (HTTP 429/503 and friends)."""

    def __init__(self, status: int, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message or f"upstream returned {status}")
        self.status = status
        self.retry_after = retry_after


class RetriesExhausted(Exception):
    def __init__(self, attempts: int, last: TransientError) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last}")
        self.attempts = attempts
        self.last = last


@dataclass
class RetryPolicy:
    attempts: int = 5
    base_delay: float = 0.05
    max_delay: float = 1.0
    jitter: float = 0.2
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def delay(self, attempt: int, error: TransientError) -> float:
        if error.retry_after is not None:
            return min(error.retry_after, self.max_delay)
        backoff = min(self.base_delay * 2 ** (attempt - 1), self.max_delay)
        return backoff * (1 + self.jitter * self.rng.random())


def call_with_retries(
    fn: Callable[..., T],
    *args: Any,
    policy: RetryPolicy,
    on_retry: Callable[[int, float, TransientError], None] | None = None,
    **kwargs: Any,
) -> T:
    """Call ``fn``; retry only ``TransientError``. Anything else propagates immediately."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return fn(*args, **kwargs)
        except TransientError as exc:
            if attempt == policy.attempts:
                raise RetriesExhausted(attempt, exc) from exc
            wait = policy.delay(attempt, exc)
            if on_retry is not None:
                on_retry(attempt, wait, exc)
            policy.sleep(wait)
    raise AssertionError("unreachable")
