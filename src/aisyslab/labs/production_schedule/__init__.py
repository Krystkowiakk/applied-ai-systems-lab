from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import ProductionScheduleEnv

LAB = Lab(
    id="production-schedule",
    title="Production schedule: code mode, in a sandbox",
    summary=(
        "Sixteen orders on three presses with changeovers, a maintenance window and two notes from the planner "
        "in plain words. The agent writes a program that plans with a scheduling library; the program runs in a "
        "sandbox without network or keys; code judges feasibility and lateness."
    ),
    techniques=(
        "code mode: the model writes a program that calls tools as a library",
        "a sandbox: separate process, time and memory limits, no network, no API keys",
        "constraints read from free text, enforced by the verifier",
        "search in code, not in the model's head",
    ),
    make_env=ProductionScheduleEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=16, max_tool_calls=16),
)
