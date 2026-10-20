from __future__ import annotations

import pytest

from aisyslab.core.calculator import evaluate
from aisyslab.core.lab import EnvOptions, RunContext
from aisyslab.core.resilience import RetriesExhausted, RetryPolicy, TransientError, call_with_retries
from aisyslab.core.tools import ToolError
from aisyslab.labs.inbox.env import InboxEnv


def flaky(failures: int, status: int = 503):
    state = {"n": 0}

    def fn() -> str:
        state["n"] += 1
        if state["n"] <= failures:
            raise TransientError(status, retry_after=0.3 if status == 429 else None)
        return "ok"

    return fn, state


def test_retries_until_success(no_sleep: RetryPolicy) -> None:
    fn, state = flaky(3)
    assert call_with_retries(fn, policy=no_sleep) == "ok" and state["n"] == 4


def test_gives_up_after_the_last_attempt(no_sleep: RetryPolicy) -> None:
    fn, _ = flaky(10)
    with pytest.raises(RetriesExhausted):
        call_with_retries(fn, policy=no_sleep)


def test_retry_after_is_respected() -> None:
    waits: list[float] = []
    fn, _ = flaky(1, status=429)
    call_with_retries(fn, policy=RetryPolicy(sleep=waits.append))
    assert waits == [0.3]


def test_other_errors_are_not_retried(no_sleep: RetryPolicy) -> None:
    calls = []

    def fn() -> None:
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        call_with_retries(fn, policy=no_sleep)
    assert len(calls) == 1


def test_exhausted_retries_reach_the_model_as_a_tool_error(ctx: RunContext) -> None:
    env = InboxEnv(0, EnvOptions(flaky=1.0))
    with pytest.raises(ToolError):
        ctx.tool_call(env.help)
    assert ctx.tracer.count("retry") == ctx.retry.attempts - 1


@pytest.mark.parametrize(
    ("expression", "value"), [("45 - 6 * 5", 15), ("(2 + 3) ** 2", 25), ("-4 + 10 / 4", -1.5), ("7 // 2 + 7 % 2", 4)]
)
def test_calculator(expression: str, value: float) -> None:
    assert evaluate(expression) == value


@pytest.mark.parametrize("expression", ["__import__('os')", "a + 1", "2 ** 100", "1 / 0", "[1, 2]", "True + 1"])
def test_calculator_rejects(expression: str) -> None:
    with pytest.raises(ToolError):
        evaluate(expression)
