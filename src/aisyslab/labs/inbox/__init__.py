from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import InboxEnv

LAB = Lab(
    id="inbox",
    title="Plant mailbox: three facts, three traps",
    summary=(
        "Extract a maintenance date, a VPN code and a change ID from ~150 live emails. "
        "Values get corrected, rotated and spoofed; new mail arrives mid-run."
    ),
    techniques=(
        "two-step retrieval (metadata, then full thread)",
        "Gmail-style query operators",
        "live source polling",
        "verifier feedback loop",
        "blackboard workers (baseline)",
        "sender trust rule",
    ),
    make_env=InboxEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=24, max_tool_calls=40),
)
