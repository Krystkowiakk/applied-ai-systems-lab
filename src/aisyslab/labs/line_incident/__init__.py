from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import LineIncidentEnv

LAB = Lab(
    id="line-incident",
    title="Line stoppage: a timeline the on-call engineer can use",
    summary=(
        "Compress a day of production-line logs (~2,100 lines) into a timeline under a token budget. "
        "The reviewer rejects invented events and names systems still missing from the causal chain."
    ),
    techniques=(
        "context budget with explicit token counting",
        "search tools with grouped repeats instead of raw logs",
        "seed-first, then verifier-driven expansion",
        "grounding check: every line must match a real event",
        "deterministic noise filtering before any model call",
    ),
    make_env=LineIncidentEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    agent_config=AgentConfig(max_steps=24, max_tool_calls=40),
)
