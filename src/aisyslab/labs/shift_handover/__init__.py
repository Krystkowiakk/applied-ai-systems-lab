from ...core.agent import AgentConfig
from ...core.lab import Lab
from . import agent, baseline
from .env import ShiftHandoverEnv

LAB = Lab(
    id="shift-handover",
    title="Shift handover: what the next shift needs to know",
    summary=(
        "A week of shift logs, read once, kept as observational memory with a 400-token handover record; "
        "on Monday morning the shift lead's questions are answered from memory alone. A bypass removed "
        "later, a limit that changed, temporary fixes of which one was made permanent, a suspicion nobody "
        "confirmed, the same sensor on two lines, and a question the logs never answer."
    ),
    techniques=(
        "observational memory: an observer writes scoped observations, a reflector condenses them",
        "material sealed after observation; the record goes into the prompt, observations are recalled",
        "a token budget for the record, enforced by the environment",
        "facts that supersede older ones; suspicions kept apart from confirmed faults",
        "reflection triggered by the size of new observations, in code",
    ),
    make_env=ShiftHandoverEnv,
    baseline=baseline.solve,
    agent=agent.solve,
    spec=agent.SPEC,
    agent_config=AgentConfig(max_steps=45, max_tool_calls=40),
)
