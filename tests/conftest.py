from __future__ import annotations

import pytest

from aisyslab.core.lab import RunContext
from aisyslab.core.resilience import RetryPolicy
from aisyslab.core.trace import Tracer


@pytest.fixture
def no_sleep() -> RetryPolicy:
    return RetryPolicy(sleep=lambda _seconds: None)


@pytest.fixture
def ctx(no_sleep: RetryPolicy) -> RunContext:
    return RunContext(tracer=Tracer(), retry=no_sleep)
