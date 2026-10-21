from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import QualityGateEnv

LAB = Lab(
    id="quality-gate",
    title="Quality gate: learn the lab's rule, do not eyeball it",
    summary=(
        "Decide which of tonight's 24 batches pass the quality gate, from 240 inspected batches whose "
        "rule nobody wrote down. One column exists only after the inspection; two batches lost a "
        "sensor reading."
    ),
    techniques=(
        "classic ML (a decision tree in code) where the answer is in labelled numbers",
        "the model chooses the method and the features; code fits and predicts",
        "target leakage: a column filled in only after the inspection",
        "explicit hold rule for missing measurements",
        "set answer graded by false passes and false holds",
    ),
    make_env=QualityGateEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    agent_config=AgentConfig(max_steps=20, max_tool_calls=30),
)
