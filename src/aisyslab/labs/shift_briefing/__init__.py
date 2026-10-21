from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import ShiftBriefingEnv

LAB = Lab(
    id="shift-briefing",
    title="Shift briefing: three specialists, one report",
    summary=(
        "Answer the plant manager's three morning questions (where the line-3 stoppage started, which "
        "work orders are blocked by parts, which batches are held) from three systems at once."
    ),
    techniques=(
        "orchestrator with specialist sub-agents: separate contexts, tools, budgets and verifiers",
        "shared workspace as the bus between agents, with optimistic concurrency",
        "references to documents instead of pasted transcripts",
        "code baseline as a parallel dependency graph",
        "sub-agent cost and trace events attributed to the parent run",
    ),
    make_env=ShiftBriefingEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=16, max_tool_calls=24),
)
