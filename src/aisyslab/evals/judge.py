"""A model as judge, and how far to trust it.

A judge is one more model with a rubric, so it has an error rate of its own. Before its grades
are allowed to decide anything, it is calibrated against answers a human has already graded:
agreement, Cohen's kappa (agreement beyond chance), and the disagreements to read.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.messages import Message, ModelTurn, ToolSpec, Usage
from ..core.model import ChatModel
from ..core.tokens import estimate_tokens

RUBRIC = (
    "PASS if the reason names the symptom the operator reported and invents no facts: a likely cause "
    "worded as a likely cause ('points to', 'likely', 'suspected') is fine. FAIL if it is vague, names a "
    "different symptom, or states something as a fact that the ticket does not say (people, readings, "
    "alarms, events)."
)
JUDGE_SYSTEM = """You grade one answer against a rubric. Reply with PASS or FAIL on the first line and
one short sentence explaining why on the second.

Rubric: {rubric}"""


class LLMJudge:
    def __init__(self, model: ChatModel, rubric: str = RUBRIC) -> None:
        self.model = model
        self.rubric = rubric
        self.usage = Usage()
        self._lock = threading.Lock()  # graded from several threads at once

    def grade(self, ticket: str, reason: str) -> tuple[bool, str]:
        turn = self.model.complete(
            system=JUDGE_SYSTEM.format(rubric=self.rubric),
            messages=[Message(role="user", content=json.dumps({"ticket": ticket, "reason": reason}))],
            tools=[],
            max_output_tokens=120,
        )
        with self._lock:
            self.usage = self.usage + turn.usage
        first, _, rest = turn.text.strip().partition("\n")
        return first.strip().upper().startswith("PASS"), rest.strip()[:200]


class KeywordJudge:
    """Offline stand-in for a judge model: PASS when the reason shares a content word with the ticket.

    Cruder than a real judge on purpose; calibration shows exactly how crude.
    """

    def __init__(self, name: str = "keyword-judge") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        payload = json.loads(messages[-1].content)
        words = lambda text: {w for w in re.findall(r"[a-z]{5,}", text.lower())}  # noqa: E731
        shared = words(payload["ticket"]) & words(payload["reason"])
        text = "PASS\nShares: " + ", ".join(sorted(shared)) if shared else "FAIL\nNo overlap with the ticket."
        usage = Usage(input_tokens=estimate_tokens(messages[-1].content), output_tokens=estimate_tokens(text))
        return ModelTurn(text=text, tool_calls=[], usage=usage, stop_reason="end_turn", model=self._name)


@dataclass(frozen=True)
class Labelled:
    ticket: str
    reason: str
    human_pass: bool


# Reasons a human reviewer graded for the triage task (PASS = faithful to the ticket).
HUMAN_LABELS = [
    Labelled(
        "coolant pump 2: grinding noise at the drive end.",
        "Grinding noise at the drive end points to the bearing.",
        True,
    ),
    Labelled("coolant pump 2: grinding noise at the drive end.", "The pump probably needs a new impeller.", False),
    Labelled(
        "press 4: oil on the floor under the hydraulic hose. line stopped.", "Oil under the hydraulic hose: leak.", True
    ),
    Labelled(
        "press 4: oil on the floor under the hydraulic hose.",
        "Hydraulic issue reported by the night shift supervisor Anna.",
        False,
    ),
    Labelled("conveyor L3: proximity sensor flickers.", "Proximity sensor flickers, likely wiring.", True),
    Labelled("conveyor L3: proximity sensor flickers.", "Sensor problem.", False),
    Labelled("drive motor M7: motor trips on overload.", "Motor trips on overload, winding suspected.", True),
    Labelled(
        "drive motor M7: motor trips on overload. no alarms on the HMI.",
        "The motor trips; HMI alarms confirm overload.",
        False,
    ),
    Labelled("air compressor C2: air pressure falls below 5 bar.", "Air pressure falls below 5 bar.", True),
    Labelled("air compressor C2: hissing at the pneumatic line.", "Compressor overheating.", False),
    Labelled("extraction fan 1: seal weeping after restart.", "Seal weeping after restart.", True),
    Labelled("extraction fan 1: seal weeping after restart.", "Fan restarted after the weekend.", False),
]


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 3)


def calibrate(judge: LLMJudge, labelled: Sequence[Labelled] = HUMAN_LABELS) -> dict[str, object]:
    verdicts = [judge.grade(item.ticket, item.reason) for item in labelled]
    machine = [passed for passed, _ in verdicts]
    human = [item.human_pass for item in labelled]
    disagreements = [
        {"ticket": item.ticket, "reason": item.reason, "human": item.human_pass, "judge": passed, "note": note}
        for item, (passed, note) in zip(labelled, verdicts, strict=True)
        if passed != item.human_pass
    ]
    return {
        "n": len(labelled),
        "agreement": round(sum(m == h for m, h in zip(machine, human, strict=True)) / len(labelled), 3),
        "kappa": cohen_kappa(machine, human),
        "disagreements": disagreements,
    }
