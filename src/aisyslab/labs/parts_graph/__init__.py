from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import PartsGraphEnv

LAB = Lab(
    id="parts-graph",
    title="Late part impact: a graph of parts, assemblies, machines, lines and orders",
    summary=(
        "A supplier delays a part. Which lines does it hold up, which open orders are at risk, which can an "
        "approved alternate save? The answer runs across five documents, and notes change what the tables say: "
        "a machine moved, an assembly went to stores, an approval was withdrawn, a move is only planned."
    ),
    techniques=(
        "graph RAG: the relations are the knowledge, so traversal tools instead of similarity",
        "tables loaded by code; notes extracted into facts by a model, one fresh context per note",
        "a parser that checks every extracted fact (grammar, relation, types, dates, IDs in the source) and "
        "sends rejected lines back once",
        "time on the edges: since and until, a functional relation whose dated fact ends the old one",
        "facts with their documents, so a doubtful hop can be read and corrected through the same parser",
    ),
    make_env=PartsGraphEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=30, max_tool_calls=30, max_input_tokens=250_000),
)
