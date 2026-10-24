"""A prompt drafted from an interview with the process owner, then measured like any other.

The person who owns a process knows things no dataset holds: how operators describe a failure,
which two codes people confuse, what to ignore. A meta-prompt gets that knowledge out in two steps.
An interviewer model asks the owner one question at a time, the most useful first, within a fixed
number of questions. A drafter model turns the transcript into classification guidance. Code
assembles the prompt: the answer format and the list of codes are fixed and never rewritten, the
guidance goes in between. The drafted prompt then runs on the same dev and holdout splits as the
starting prompt; an interview that sounded good but scores worse is visible as such.

Offline, three stand-ins play the parts: a fixed list of questions, an owner who answers from the
notes below, and a drafter that turns quoted phrases into rules the rule-following model reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.messages import Message, ModelTurn, ToolSpec, Usage
from ..core.model import ChatModel
from ..core.tokens import estimate_tokens
from .triage import CODES, PROMPT_HEADER

# What the planner knows, in the planner's words. With a provider, a model plays the planner and
# answers only from these notes; the interviewer never sees them.
OWNER_NOTES = """- Noise, heat, vibration or play at a shaft or at the drive end is the bearing, MECH-BRG. Operators
  say 'grinding', 'rumbling', 'running hot', 'shaft play' or 'sounds like gravel'.
- Coolant escaping along a shaft or at a seal housing is the seal, MECH-SEAL: 'dripping at the shaft
  seal', 'weeping', 'wet patch', 'creeping out along the shaft'.
- Oil from a hose, a fitting, the tank or the hydraulic unit is HYD-LEAK, never a seal: 'oil on the
  floor', 'fitting leaking', 'oil level falling', 'puddle of oil'.
- A motor that trips, a breaker that trips for a motor, a burnt smell or failed insulation is ELEC-MTR:
  'trips on overload', 'burnt smell', 'insulation test failed', 'breaker for the motor'.
- Readings that drop, jump or flicker and signals that get lost are the sensor or its wiring,
  ELEC-SENS, not the machine: 'drops to zero', 'flickers', 'signal lost', 'jumps between zero'.
- Anything that loses air is PNEU-PRS: 'air pressure falls', 'hissing', 'low air pressure', a clamp
  that 'loses grip'.
- When a ticket says a part was replaced, checked or is fine, that part is not the problem. Classify
  by the symptom the operator reports.
- Priority 1 only when the ticket says the line stopped, the shift cannot run or there is a safety
  risk. Everything else is 2; 3 is for cosmetic things."""

INTERVIEWER_SYSTEM = """You prepare to write the instructions for a model that triages maintenance tickets into
failure codes and priorities. The codes are: {codes}. You interview the maintenance planner who owns
the process. Ask one short question at a time about what you need to classify well: how each code
shows up in operators' words, how to tell similar codes apart, what to ignore, when a ticket is
urgent. You have {n} questions in all; ask the most useful first. Reply with the question only."""

OWNER_SYSTEM = """You are the maintenance planner who owns ticket triage at the plant. Answer the
interviewer's question briefly and in plain words, from your notes below. Say only what the notes
support; when they do not cover something, say so.

Your notes:
{notes}"""

DRAFTER_SYSTEM = """You write the classification guidance for a model that triages maintenance tickets,
from an interview with the planner who owns the process. The answer format and the list of codes
are fixed and added by code: write only the guidance. Short rules in the planner's terms: how each
code shows up in operators' words, how to tell codes apart, what to ignore, when priority is 1. Use
only what the interview says. Plain text, at most 15 lines."""

QUESTIONS = (
    "How do operators describe a bearing problem?",
    "How do I tell a seal leak from a hydraulic leak?",
    "What does a motor fault look like in a ticket, compared with a sensor fault?",
    "What do air pressure problems sound like in a ticket?",
    "What should I ignore in a ticket?",
    "When is a ticket priority 1?",
)


def interview(interviewer: ChatModel, owner: ChatModel, *, questions: int = 6) -> tuple[list[tuple[str, str]], Usage]:
    """Question by question: the interviewer sees the answers so far, the owner the questions so far."""
    transcript: list[tuple[str, str]] = []
    usage = Usage()
    system = INTERVIEWER_SYSTEM.format(codes=", ".join(CODES), n=questions)
    for _ in range(questions):
        asked: list[Message] = [Message(role="user", content="Start the interview.")]
        for q, a in transcript:
            asked += [Message(role="assistant", content=q), Message(role="user", content=a)]
        turn = interviewer.complete(system=system, messages=asked, tools=[], max_output_tokens=200)
        question = turn.text.strip()
        heard: list[Message] = []
        for q, a in transcript:
            heard += [Message(role="user", content=q), Message(role="assistant", content=a)]
        reply = owner.complete(
            system=OWNER_SYSTEM.format(notes=OWNER_NOTES),
            messages=[*heard, Message(role="user", content=question)],
            tools=[],
            max_output_tokens=400,
        )
        transcript.append((question, reply.text.strip()))
        usage = usage + turn.usage + reply.usage
    return transcript, usage


def draft_prompt(drafter: ChatModel, transcript: Sequence[tuple[str, str]]) -> tuple[str, Usage]:
    """The drafter writes guidance; code puts it between the fixed header and the fixed default rule."""
    text = "\n\n".join(f"Interviewer: {q}\nPlanner: {a}" for q, a in transcript)
    turn = drafter.complete(
        system=DRAFTER_SYSTEM, messages=[Message(role="user", content=text)], tools=[], max_output_tokens=1200
    )
    guidance = turn.text.strip()
    return f"{PROMPT_HEADER}\n\nRULES:\n{guidance}\n- if nothing else applies -> MECH-BRG\n", turn.usage


class _StandIn:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _turn(self, text: str, prompt: str) -> ModelTurn:
        usage = Usage(input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(text))
        return ModelTurn(text=text, tool_calls=[], usage=usage, stop_reason="end_turn", model=self._name)


class ScriptedInterviewer(_StandIn):
    """Asks the fixed questions in order."""

    def __init__(self) -> None:
        super().__init__("scripted-interviewer")

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        asked = sum(m.role == "assistant" for m in messages)
        return self._turn(QUESTIONS[asked % len(QUESTIONS)], system)


class NotesOwner(_StandIn):
    """Answers with the note lines that share the most words with the question."""

    def __init__(self) -> None:
        super().__init__("notes-owner")

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        words = set(re.findall(r"[a-z]+", messages[-1].content.lower())) - {"a", "the", "do", "how", "what", "i", "in"}
        notes = [n.strip() for n in re.split(r"\n- ", "\n" + OWNER_NOTES) if n.strip()]
        scored = sorted(notes, key=lambda n: -len(words & set(re.findall(r"[a-z]+", n.lower()))))
        return self._turn(" ".join(scored[:2]).replace("\n  ", " "), system)


class RuleDrafter(_StandIn):
    """Turns every quoted phrase into a rule for the code the planner named last before it."""

    def __init__(self) -> None:
        super().__init__("rule-drafter")

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        rules: list[str] = []
        token = re.compile(r"\b(" + "|".join(CODES) + r")\b|'([^']+)'")
        for answer in messages[-1].content.split("Planner:")[1:]:
            current = None
            for match in token.finditer(answer.split("Interviewer:")[0]):
                current = match[1] or current
                rule = f"- if the ticket mentions '{match[2]}' -> {current}"
                if match[2] and current and rule not in rules:
                    rules.append(rule)
        return self._turn("\n".join(rules), system)
