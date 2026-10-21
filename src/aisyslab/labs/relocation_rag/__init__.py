from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import RelocationRagEnv

LAB = Lab(
    id="relocation-rag",
    title="Relocation assistant: answer, cite, or say it is not covered",
    summary=(
        "Seven questions from someone moving abroad for work, answered from a handbook with a superseded "
        "edition, an overriding exception, a deadline to compute, a question in Polish and one the handbook "
        "does not answer. Graded on value and citation."
    ),
    techniques=(
        "hybrid retrieval: BM25 + vectors fused with Reciprocal Rank Fusion",
        "agent writes the queries (cross-language)",
        "edition metadata to avoid superseded sources",
        "citations checked by the verifier",
        "refusal when the sources are silent",
        "date and number arithmetic in tools",
    ),
    make_env=RelocationRagEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=30, max_tool_calls=50),
)
