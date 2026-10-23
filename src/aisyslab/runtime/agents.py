"""The agents the service runs, the built-in tools it adds, and process files agents read on demand.

Every lab agent can run as a service job: its tools and its simulated systems come from the lab,
its standing task becomes the first message, and its lab's check says whether the result is
verified. Two agents exist only in the service: an operations assistant that hands questions to
the lab agents, and a writer that assembles the morning briefing.

Agent settings are snapshotted with every run (a revision), so a reported problem can be traced
to the prompt, budgets and models that were in force at the time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.lab import EnvOptions, Lab, RunContext
from ..core.messages import ToolSpec
from ..core.model import ChatModel
from ..core.tools import Tool, Toolbox, ToolError

ASK_HUMAN = "ask_human"
DELEGATE = "delegate_to_agent"
READ_PROCESS = "read_process"
# Calls the runtime answers itself; they never change a lab's world, so replay skips them.
BUILTINS = frozenset({ASK_HUMAN, DELEGATE, READ_PROCESS})

ToolFactory = Callable[[Any, ChatModel | None, RunContext], Toolbox]


@dataclass(frozen=True)
class RuntimeAgent:
    name: str
    description: str
    system: str
    world: Callable[[int], Any] | None = None  # the simulated systems for a seed
    tools: ToolFactory | None = None
    task: Callable[[Any], str] | None = None  # a standing task, for agents that have one
    done: Callable[[Any], bool] | None = None  # the check's verdict, for agents that have one
    delegates: tuple[str, ...] = ()
    ask_human: bool = False
    processes: bool = False
    route: str = "default"
    max_steps: int = 20
    max_tool_calls: int = 40
    max_output_tokens: int = 4096

    def first_message(self, text: str, seed: int) -> str:
        """The first message of a thread: the standing task, with the requester's note if any."""
        if self.task is None or self.world is None:
            return text.strip()
        standing = self.task(self.world(seed))
        if not text.strip():
            return standing
        # The note adds context; it does not replace the task or the check the answer goes to.
        return f"{standing}\n\nContext from the requester (the task above stands; finish it with your own tools and check):\n{text.strip()}"

    def verified(self, env: Any) -> bool | None:
        return None if self.done is None or env is None else bool(self.done(env))

    def snapshot(self, models: Sequence[str]) -> dict[str, Any]:
        return {
            "description": self.description,
            "system": self.system,
            "delegates": list(self.delegates),
            "ask_human": self.ask_human,
            "processes": self.processes,
            "route": self.route,
            "models": list(models),
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_output_tokens": self.max_output_tokens,
        }


def lab_agent(lab: Lab, options: EnvOptions | None = None) -> RuntimeAgent:
    if lab.spec is None:
        raise ValueError(f"Lab '{lab.id}' has no agent spec.")
    spec, opts = lab.spec, options or EnvOptions()
    return RuntimeAgent(
        name=lab.id,
        description=lab.summary,
        system=spec.system,
        world=lambda seed: lab.make_env(seed, opts),
        tools=spec.toolbox,
        task=spec.task,
        done=spec.done,
        max_steps=lab.agent_config.max_steps,
        max_tool_calls=lab.agent_config.max_tool_calls,
        max_output_tokens=lab.agent_config.max_output_tokens,
    )


OPERATIONS_SYSTEM = """You answer questions from the shift lead of a manufacturing plant. You cannot
see the plant's systems yourself; specialist agents can. Hand each part of a question to the right
specialist with delegate_to_agent. Each specialist has a standing question and its own check: tell
it in a sentence or two what you need and any context from this conversation (it cannot see it),
and do not prescribe an answer format, because its answer comes back as its checked submission.
Ask the shift lead with ask_human only when the request is ambiguous or the decision is theirs.
When the answers are in, reply in a few lines: what each specialist found and whether its own
check verified it. Never present an unverified finding as certain."""

BRIEFING_SYSTEM = """You write the morning briefing for the shift lead from the reports other agents
collected overnight. Read the process files you need before writing. Use only what the reports
say: a missing report stays missing, and you name it."""

OPERATIONS_ASSISTANT = RuntimeAgent(
    name="operations-assistant",
    description="Answers the shift lead's questions by handing them to the plant's specialist agents.",
    system=OPERATIONS_SYSTEM,
    delegates=("line-incident", "plant-database", "quality-gate"),
    ask_human=True,
    processes=True,
    max_steps=12,
    max_tool_calls=12,
)

BRIEFING_WRITER = RuntimeAgent(
    name="briefing-writer",
    description="Writes the morning briefing from the reports collected overnight.",
    system=BRIEFING_SYSTEM,
    processes=True,
    max_steps=8,
    max_tool_calls=8,
)


def default_agents(options: EnvOptions | None = None) -> dict[str, RuntimeAgent]:
    from ..labs import LABS

    agents = {lab.id: lab_agent(lab, options) for lab in LABS.values() if lab.spec is not None}
    for agent in (OPERATIONS_ASSISTANT, BRIEFING_WRITER):
        agents[agent.name] = agent
    return agents


def ask_human_spec() -> ToolSpec:
    return ToolSpec(
        name=ASK_HUMAN,
        description=(
            "Ask the person who made the request one question and wait for the answer. Use it when the "
            "request is ambiguous or the decision is theirs, not to confirm what you can check yourself."
        ),
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "One clear question."}},
            "required": ["question"],
            "additionalProperties": False,
        },
    )


def delegate_spec(agents: Mapping[str, RuntimeAgent], names: Sequence[str]) -> ToolSpec:
    listing = "; ".join(f"{n}: {agents[n].description}" for n in names if n in agents)
    return ToolSpec(
        name=DELEGATE,
        description=(
            "Hand a question to a specialist agent and wait for its result. The specialist works in its own "
            "context with its own tools and check, and cannot see this conversation: say what you need in a "
            "sentence or two, with the context it needs, and do not prescribe a format; the result is its "
            f"checked submission. Specialists: {listing}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": list(names)},
                "task": {"type": "string", "description": "Everything the specialist needs to know."},
            },
            "required": ["agent", "task"],
            "additionalProperties": False,
        },
    )


class _ProcessArgs(BaseModel):
    file: str = Field(description="A process file as listed, e.g. 'morning-briefing/assemble'.")


class ProcessLibrary:
    """Process descriptions as Markdown files, one folder per process.

    The front matter's description says *when* a file applies and goes into the agent's prompt;
    the body says *how* and is read only when needed, so the prompt stays short.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root

    @classmethod
    def default(cls) -> ProcessLibrary:
        return cls(Path(__file__).parent / "processes")

    def entries(self) -> dict[str, str]:
        if self.root is None or not self.root.is_dir():
            return {}
        return {
            f"{path.parent.name}/{path.stem}": _front_matter(path.read_text(encoding="utf-8"))[0].get("description", "")
            for path in sorted(self.root.glob("*/*.md"))
        }

    def read(self, name: str) -> str:
        if name not in self.entries() or self.root is None:
            raise ToolError(f"Unknown process file '{name}'.", hint=f"Available: {', '.join(self.entries())}.")
        return _front_matter((self.root / f"{name}.md").read_text(encoding="utf-8"))[1].strip()

    def listing(self) -> str:
        return "\n".join(f"- {name}: {description}" for name, description in self.entries().items())

    def tool(self) -> Tool:
        return Tool(
            name=READ_PROCESS,
            description="Read a process file: how a recurring piece of work is done here.",
            args=_ProcessArgs,
            handler=lambda args: {"file": args.file, "text": self.read(args.file)},
        )


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    head, _, body = text[4:].partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, body
