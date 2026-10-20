"""Run a lab once with one solver and report what it cost."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .agent import AgentConfig
from .lab import EnvOptions, Lab, RunContext
from .messages import Usage
from .model import ChatModel
from .resilience import RetriesExhausted, RetryPolicy
from .trace import Tracer

Solver = Literal["baseline", "agent"]


@dataclass
class RunReport:
    lab: str
    solver: Solver
    seed: int
    model: str | None
    accepted: bool
    score: float | None
    status: str
    submissions: int
    api_calls: int
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    retries: int
    seconds: float
    trace_path: str | None
    error: str | None


def run_once(
    lab: Lab,
    solver: Solver,
    seed: int,
    *,
    model: ChatModel | None = None,
    options: EnvOptions | None = None,
    trace_dir: Path | None = None,
    echo: Callable[[str], None] | None = None,
    agent_config: AgentConfig | None = None,
    retry: RetryPolicy | None = None,
    configure_env: Callable[[Any], None] | None = None,
) -> RunReport:
    options = options or EnvOptions()
    if solver == "agent" and model is None:
        raise ValueError("the agent solver needs a model")
    env = lab.make_env(seed, options)
    if configure_env is not None:
        configure_env(env)
    trace_path = None
    if trace_dir is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        trace_path = trace_dir / f"{lab.id}__{solver}__seed{seed}__{stamp}.jsonl"
    tracer = Tracer(trace_path, echo=echo)
    ctx = RunContext(tracer=tracer, retry=retry or RetryPolicy(), agent=agent_config or lab.agent_config)
    tracer.event(
        "run_start", lab=lab.id, solver=solver, seed=seed, model=model.name if model else None, flaky=options.flaky
    )

    started = time.perf_counter()
    status, error = "completed", None
    usage, model_calls, tool_calls = Usage(), 0, 0
    try:
        if solver == "baseline":
            lab.baseline(env, ctx)
        else:
            assert model is not None
            run = lab.agent(env, model, ctx)
            status, error = run.status, run.error
            usage, model_calls, tool_calls = run.usage, run.model_calls, run.tool_calls
    except RetriesExhausted as exc:
        status, error = "upstream_unavailable", str(exc)
    except Exception as exc:  # a crash is a result to report, not something to hide
        status, error = "crashed", f"{type(exc).__name__}: {exc}"
    usage = usage + ctx.side_usage
    model_calls += ctx.side_calls

    report = RunReport(
        lab=lab.id,
        solver=solver,
        seed=seed,
        model=model.name if model else None,
        accepted=env.accepted,
        score=_score(env.submissions),
        status=status,
        submissions=len(env.submissions),
        api_calls=env.api_calls,
        model_calls=model_calls,
        tool_calls=tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        retries=tracer.count("retry"),
        seconds=round(time.perf_counter() - started, 3),
        trace_path=str(trace_path) if trace_path else None,
        error=error,
    )
    tracer.event("run_end", **asdict(report))
    return report


def _score(submissions: list) -> float | None:
    """Fraction correct reported by the last verdict, for labs that grade partially."""
    if not submissions:
        return None
    if "score" in submissions[-1].details:
        return float(submissions[-1].details["score"])
    return 1.0 if submissions[-1].accepted else 0.0
