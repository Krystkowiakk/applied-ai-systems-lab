"""The system under evaluation: a prompt that triages maintenance tickets.

Operators write tickets in their own words ("pump 2 grinding at the drive end"); the planner
needs a failure code and a priority. The classifier is one model call with a prompt, which is
exactly the kind of component that drifts when someone edits the prompt or swaps the model,
and that only an evaluation set keeps honest.

Two deterministic stand-ins make the whole evaluation pipeline runnable offline:
``RuleFollowerModel`` classifies only by rules written in the prompt (a weak model that knows
nothing the prompt does not say), and ``RuleProposerModel`` plays the stronger optimiser that
reads failures and proposes rules. With a provider, real models take both roles.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.messages import Message, ModelTurn, ToolSpec, Usage
from ..core.model import ChatModel, ModelError
from ..core.tokens import estimate_tokens

CODES = {
    "MECH-BRG": "Mechanical: bearing failure or wear",
    "MECH-SEAL": "Mechanical: seal leak",
    "ELEC-MTR": "Electrical: motor winding or insulation",
    "ELEC-SENS": "Electrical: sensor or wiring fault",
    "HYD-LEAK": "Hydraulic: leak",
    "PNEU-PRS": "Pneumatic: pressure loss",
}
ASSETS = ["coolant pump 2", "press 4", "conveyor L3", "extraction fan 1", "drive motor M7", "air compressor C2"]
SYMPTOMS: dict[str, list[str]] = {
    "MECH-BRG": [
        "grinding noise at the drive end",
        "bearing running hot",
        "rumbling and high vibration",
        "shaft play felt by hand",
    ],
    "MECH-SEAL": [
        "coolant dripping at the shaft seal",
        "seal weeping after restart",
        "wet patch under the seal housing",
    ],
    "ELEC-MTR": ["motor trips on overload", "burnt smell from the motor", "insulation test failed on the windings"],
    "ELEC-SENS": ["sensor reading drops to zero", "proximity sensor flickers", "signal lost on the encoder cable"],
    "HYD-LEAK": [
        "oil on the floor under the hydraulic hose",
        "hydraulic fitting leaking",
        "oil level falling in the tank",
    ],
    "PNEU-PRS": [
        "air pressure falls below 5 bar",
        "hissing at the pneumatic line",
        "cylinder too slow, low air pressure",
    ],
}
# Wordings that appear only in the holdout split: a prompt that memorised the training phrases
# meets symptoms described differently, which is what the holdout is for.
HOLDOUT_SYMPTOMS: dict[str, list[str]] = {
    "MECH-BRG": ["drive end sounds like gravel"],
    "MECH-SEAL": ["coolant creeping out along the shaft"],
    "ELEC-MTR": ["breaker for the motor keeps tripping"],
    "ELEC-SENS": ["reading jumps between zero and normal"],
    "HYD-LEAK": ["puddle of oil next to the hydraulic unit"],
    "PNEU-PRS": ["pneumatic clamp loses grip"],
}
DISTRACTORS = [
    "bearing was replaced last month",
    "no oil visible",
    "motor checked and fine",
    "operator says it started on the night shift",
    "no alarms on the HMI",
]
URGENT = ["line stopped", "safety risk", "cannot run the shift"]

PROMPT_HEADER = """You triage maintenance tickets for a manufacturing plant.
Answer with JSON only: {"code": "<failure code>", "priority": 1|2|3, "reason": "<one sentence>"}.
Priority 1 only when the ticket itself says the line is stopped, reports a safety risk or says the shift
cannot run; do not escalate on a risk you infer. Otherwise 2, or 3 for cosmetic issues.

Failure codes:
""" + "\n".join(f"- {code}: {meaning}" for code, meaning in CODES.items())

BASE_PROMPT = PROMPT_HEADER + "\n\nRULES:\n- if nothing else applies -> MECH-BRG\n"


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    code: str
    priority: int
    split: str  # train, dev or holdout


def generate_tickets(seed: int = 0, n: int = 90) -> list[Ticket]:
    """A seeded dataset: varied wording, distractors that name the wrong component, urgent cases.

    Every ticket's text is unique: a ticket drawn twice is drawn again. A repeat across splits
    would put a holdout ticket into training, word for word, and inflate the holdout score."""
    r = random.Random(seed)
    codes = list(CODES)
    tickets: list[Ticket] = []
    seen: set[str] = set()
    for i in range(n):
        code = codes[i % len(codes)]
        split = "holdout" if i % 5 == 4 else ("dev" if i % 5 == 3 else "train")
        pool = SYMPTOMS[code] + (HOLDOUT_SYMPTOMS[code] if split == "holdout" else [])
        while True:
            asset = r.choice(ASSETS)
            parts = [f"{asset}: {r.choice(pool)}"]
            if r.random() < 0.35:
                parts.append(r.choice(DISTRACTORS))  # names another component: a trap for keyword matching
            urgent = r.random() < 0.3
            if urgent:
                parts.append(r.choice(URGENT))
            r.shuffle(parts)
            text = ". ".join(parts) + "."
            if text not in seen:
                break
        seen.add(text)
        tickets.append(Ticket(f"T-{seed}-{i:03d}", text, code, 1 if urgent else 2, split))
    return tickets


# ----- deterministic stand-ins ------------------------------------------------------------------
RULE = re.compile(r"^- if the ticket mentions '(.+?)' -> ([A-Z]+-[A-Z]+)$", re.M)
DEFAULT = re.compile(r"^- if nothing else applies -> ([A-Z]+-[A-Z]+)$", re.M)


class RuleFollowerModel:
    """Classifies by the rules in its prompt and nothing else: a model that knows only the prompt."""

    def __init__(self, name: str = "rule-follower") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        ticket = messages[-1].content.lower()
        code = next((c for phrase, c in RULE.findall(system) if phrase.lower() in ticket), None)
        if code is None:
            default = DEFAULT.search(system)
            code = default.group(1) if default else "MECH-BRG"
        priority = 1 if any(u in ticket for u in URGENT) else 2
        reason = f"Ticket mentions {ticket.split(':')[0]}."
        text = json.dumps({"code": code, "priority": priority, "reason": reason})
        usage = Usage(input_tokens=estimate_tokens(system + ticket), output_tokens=estimate_tokens(text))
        return ModelTurn(text=text, tool_calls=[], usage=usage, stop_reason="end_turn", model=self._name)


class RuleProposerModel:
    """Plays the optimiser: reads failures and proposes the prompt with one more rule per code."""

    def __init__(self, name: str = "rule-proposer") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        request = json.loads(messages[-1].content)
        prompt: str = request["prompt"]
        known = {phrase.lower() for phrase, _ in RULE.findall(prompt)}
        added: list[str] = []
        for failure in request["failures"]:
            symptom = failure["ticket"].split(":", 1)[-1]
            phrase = next(
                (s for s in SYMPTOMS.get(failure["expected"], []) if s in symptom and s.lower() not in known), None
            )
            if phrase:
                known.add(phrase.lower())
                added.append(f"- if the ticket mentions '{phrase}' -> {failure['expected']}")
        if not added:
            raise ModelError("No change to propose.")
        new_prompt = prompt.replace("RULES:\n", "RULES:\n" + "\n".join(added) + "\n", 1)
        usage = Usage(input_tokens=estimate_tokens(messages[-1].content), output_tokens=estimate_tokens(new_prompt))
        return ModelTurn(text=new_prompt, tool_calls=[], usage=usage, stop_reason="end_turn", model=self._name)


def classify(model: ChatModel, prompt: str, ticket: Ticket) -> tuple[str, Usage]:
    """One call of the system under evaluation; returns the raw output for the graders."""
    turn = model.complete(
        system=prompt, messages=[Message(role="user", content=ticket.text)], tools=[], max_output_tokens=200
    )
    return turn.text, turn.usage
