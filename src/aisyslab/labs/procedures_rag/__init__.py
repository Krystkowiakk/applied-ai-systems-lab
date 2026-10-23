from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import ProceduresRagEnv

LAB = Lab(
    id="procedures-rag",
    title="Procedures assistant: answer, cite, or say it is not covered",
    summary=(
        "Seven questions from a maintenance technician, answered from the plant's procedures: a superseded "
        "revision, an exception that overrides the rule, a calibration date to compute, a question in Polish "
        "and one the procedures do not answer. Graded on value and citation."
    ),
    techniques=(
        "hybrid retrieval: BM25 + vectors fused with Reciprocal Rank Fusion",
        "agent writes the queries (cross-language)",
        "revision metadata to avoid superseded procedures",
        "citations checked by the verifier",
        "refusal when the sources are silent",
        "date and number arithmetic in tools",
    ),
    make_env=ProceduresRagEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=30, max_tool_calls=50),
)
