from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import SupplyIntakeEnv

LAB = Lab(
    id="supply-intake",
    title="Supplier delays: which part stops a line tonight?",
    summary=(
        "Twelve intake items in JSON, CSV, XML, HTML, text and a rack photo, several with misleading names. "
        "Combine them into one shortage report: part, supplier, who to call, units missing."
    ),
    techniques=(
        "route by content (magic bytes), not by file name",
        "decode structured formats in code; model only where needed",
        "vision on the photo, cross-checked by a pixel validator",
        "arithmetic through a calculator tool",
        "physical count overrides the system of record",
    ),
    make_env=SupplyIntakeEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=24, max_tool_calls=40),
)
