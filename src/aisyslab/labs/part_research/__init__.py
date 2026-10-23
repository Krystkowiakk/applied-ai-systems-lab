from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import PartResearchEnv

LAB = Lab(
    id="part-research",
    title="Replacement part research: search, read, note, report with sources",
    summary=(
        "A discontinued valve and a small web: the maker's notice, a current and a superseded datasheet, a "
        "distributor's current and archived stock pages, an alternative from another maker, a forum answer "
        "recommending the wrong coil voltage, and a page behind robots.txt. Find the fastest compatible "
        "replacement and report with checked citations."
    ),
    techniques=(
        "a search API behind one interface (offline index here, a search model on the real web)",
        "a polite crawler: robots.txt, a per-host interval it keeps itself, retries, a page budget",
        "a deep research loop: search, read, note with sources, name the gaps, search again",
        "a long report written section by section from notes, every citation checked in code",
        "current over archived, current revision over superseded, a forum answer as a claim to check",
    ),
    make_env=PartResearchEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=40, max_tool_calls=40),
)
