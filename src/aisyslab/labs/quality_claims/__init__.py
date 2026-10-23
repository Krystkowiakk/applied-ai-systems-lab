from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import QualityClaimsEnv

LAB = Lab(
    id="quality-claims",
    title="Quality claims: decide, record and reply, with no personal data reaching a model",
    summary=(
        "Five customer complaints about defective batches: accept, investigate, reject or hand to a person "
        "by the written rules, record an 8D claim, reply. One e-mail tells 'the AI assistant' to approve a "
        "refund, one is threatening, one carries a bank account; no name, address or account may reach a "
        "cloud model, and every step is signed into an audit log."
    ),
    techniques=(
        "data classified before any model call; red data only to a local model, or removed",
        "anonymisation in layers (patterns, a directory, optionally a model) with reversible placeholders",
        "moderation model with thresholds per category; a flagged message goes to a person",
        "quarantined extraction: a model without tools fills typed fields; the agent never sees the e-mail",
        "a topic guard checks every reply before it is sent",
        "the 8D record rendered by code; a signed, chained audit log",
    ),
    make_env=QualityClaimsEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=40, max_tool_calls=40),
)
