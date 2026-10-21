from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import ServiceRepairEnv

LAB = Lab(
    id="service-repair",
    title="Edge gateway won't start: repair through a restricted shell",
    summary=(
        "Bring a line-3 MES gateway back up through a non-standard shell API. Wrong config values, "
        "a rotated token in shell history, a stale lock, and forbidden paths that trigger a ban."
    ),
    techniques=(
        "policy enforced by the environment, not the prompt",
        "single-command tool with bounded output",
        "discover the contract via help before acting",
        "iterative repair against the first failing check",
        "errors and bans returned as data",
    ),
    make_env=ServiceRepairEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=30, max_tool_calls=40),
)
