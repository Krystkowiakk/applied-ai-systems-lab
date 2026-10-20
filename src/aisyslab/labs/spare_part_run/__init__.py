from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import SparePartRunEnv

LAB = Lab(
    id="spare-part-run",
    title="Spare pump run: plan under a battery and a deadline",
    summary=(
        "Get a spare pump across the plant site to a stopped line. Forklift is fast but battery-limited and "
        "barred from walkways; walking is slow; dismounting is one-way. Rules hide in documents to discover."
    ),
    techniques=(
        "document and tool discovery before acting",
        "model proposes, simulator judges (hard local judge)",
        "several diverse candidates instead of one guess",
        "exact state-space search as the baseline",
    ),
    make_env=SparePartRunEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    agent_config=AgentConfig(max_steps=30, max_tool_calls=40),
)
