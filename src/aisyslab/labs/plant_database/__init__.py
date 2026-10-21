from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import PlantDatabaseEnv

LAB = Lab(
    id="plant-database",
    title="Plant database: blocked work orders, read-only SQL",
    summary=(
        "Find the line-3 corrective work orders that cannot start for lack of a spare part, in a "
        "maintenance database the solver has to discover. Stock is split, reserved and superseded; "
        "the status field is stale; two assets moved lines; the database refuses writes."
    ),
    techniques=(
        "schema discovery before the query (tables, definitions, sample rows)",
        "read-only enforced by the database engine (authorizer, query_only)",
        "query cost limit and bounded result sets",
        "computation in SQL, not in the model",
        "set answer graded with categorised feedback, no IDs leaked",
    ),
    make_env=PlantDatabaseEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    agent_config=AgentConfig(max_steps=20, max_tool_calls=30),
)
