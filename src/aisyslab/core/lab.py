"""What a lab is: a seeded simulated environment, a verifier and two ways to solve it."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .agent import AgentConfig, AgentRun, AgentSpec
from .messages import Message, Usage
from .model import ChatModel, ModelError
from .resilience import RetriesExhausted, RetryPolicy, TransientError, call_with_retries
from .tools import ToolError
from .trace import Tracer
from .verdict import Verdict

T = TypeVar("T")


@dataclass(frozen=True)
class EnvOptions:
    """``flaky``: probability that any API call fails with a transient 503/429.
    ``embedder``: optional embedding model for labs with retrieval (offline default otherwise).
    ``moderator``: optional moderation model for labs that screen messages (offline default otherwise).
    ``local_model``: optional model on this machine, for data that must not go to a cloud model."""

    flaky: float = 0.0
    embedder: Any = None
    moderator: Any = None
    local_model: Any = None


class LabEnv:
    """Base class for simulated environments.

    Subclasses expose their API as methods that call ``self._api(name)`` first, which counts
    the call and injects transient failures when the environment is flaky. Data generation
    and failure injection use separate random streams, so the data for a seed is identical
    whether or not the environment is flaky.
    """

    def __init__(self, seed: int, options: EnvOptions) -> None:
        self.seed = seed
        self.options = options
        self.rng = random.Random(seed)
        self._chaos = random.Random(seed * 7919 + 17)
        self.api_calls = 0
        self.submissions: list[Verdict] = []

    @property
    def accepted(self) -> bool:
        return any(v.accepted for v in self.submissions)

    def _api(self, action: str) -> None:
        self.api_calls += 1
        if self.options.flaky and self._chaos.random() < self.options.flaky:
            if self._chaos.random() < 0.5:
                raise TransientError(503, f"{action}: service unavailable")
            raise TransientError(429, f"{action}: rate limited", retry_after=0.05)

    def _record(self, verdict: Verdict) -> Verdict:
        self.submissions.append(verdict)
        return verdict


@dataclass
class RunContext:
    tracer: Tracer
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Model calls made inside tools (vision, extraction) count toward the run's cost too.
    side_usage: Usage = field(default_factory=Usage)
    side_calls: int = 0

    def side_call(
        self, model: ChatModel, *, system: str, messages: Sequence[Message], purpose: str, max_output_tokens: int = 1024
    ) -> str:
        """A single tool-less model call made on behalf of a tool; traced and costed."""
        try:
            turn = model.complete(system=system, messages=messages, tools=[], max_output_tokens=max_output_tokens)
        except ModelError as exc:
            self.tracer.event("model_error", purpose=purpose, error=str(exc))
            raise ToolError(f"The {purpose} model call failed.", hint="Try again or continue without it.") from exc
        self.side_calls += 1
        self.side_usage = self.side_usage + turn.usage
        self.tracer.event(
            "side_generation",
            purpose=purpose,
            model=turn.model,
            input_tokens=turn.usage.input_tokens,
            output_tokens=turn.usage.output_tokens,
            text=turn.text[:240],
        )
        return turn.text

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call an environment API with retries; transient errors never reach the solver."""

        def log(attempt: int, delay: float, exc: TransientError) -> None:
            self.tracer.event("retry", attempt=attempt, delay=delay, reason=f"{exc.status} {exc}")

        return call_with_retries(fn, *args, policy=self.retry, on_retry=log, **kwargs)

    def tool_call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Same as ``call`` but exhausted retries become a ToolError the model can see."""
        try:
            return self.call(fn, *args, **kwargs)
        except RetriesExhausted as exc:
            raise ToolError(
                "Upstream service unavailable after retries.", hint="Try again later or continue with what you have."
            ) from exc

    def submit(self, fn: Callable[..., Verdict], *args: Any, as_tool: bool = False, **kwargs: Any) -> Verdict:
        """Send an answer to the verifier and trace its verdict."""
        verdict = (self.tool_call if as_tool else self.call)(fn, *args, **kwargs)
        self.tracer.event("verdict", **verdict.as_dict())
        return verdict


@dataclass(frozen=True)
class Lab:
    id: str
    title: str
    summary: str
    techniques: tuple[str, ...]
    make_env: Callable[[int, EnvOptions], Any]
    baseline: Callable[[Any, RunContext], None]
    agent: Callable[[Any, ChatModel, RunContext], AgentRun]
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    origin: str = ""
    # The agent's prompt, task and tool factory, so the lab's tools can also be served over MCP.
    spec: AgentSpec | None = None
