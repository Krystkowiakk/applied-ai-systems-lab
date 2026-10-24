"""Observational memory: a journal the model writes and condenses, so an assistant remembers
beyond one session without carrying the whole history.

Two kinds of model calls maintain it. They are calls, not agents:

- The **observer** reads new material (a shift's log, a conversation) and writes observations,
  each labelled with what it is about (its scope) and its status. The material it has read is
  then sealed: it leaves the context for good.
- The **reflector** reads the current record and the observations made since it was written,
  and condenses them into a new record: what a later observation resolved is dropped, a changed
  value keeps only the new value, a suspicion stays a suspicion until someone checks it, every
  line keeps its scope, and the whole fits a token budget. It runs on its own once the new
  observations pass a threshold.

New observations are a working area; only the reflector writes the record, which goes at the
end of the system prompt of every new session. The observations stay searchable by scope.
Compression forgets by design, so the instructions say what must not be forgotten.

Cost is bounded in code, not by the model's cooperation: a reflection makes at most two calls,
and a record still over budget after them is cut to fit, least important lines first (resolved
items, then standing facts; open items and suspicions last), with the cut reported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TypeVar

from ..core.retrieval import HybridIndex
from ..core.tokens import estimate_tokens

STATUSES = ("open", "resolved", "unconfirmed", "info")
SCOPE = re.compile(r"^(plant|line-\d+|asset:[a-z0-9][a-z0-9 -]{0,40}|person:[a-z][a-z -]{0,30})$")

T = TypeVar("T")

# (system, user, purpose) -> the model's text. The caller decides how calls are made and costed.
ModelCall = Callable[[str, str, str], str]

OBSERVER_SYSTEM = """You keep the handover memory of a maintenance team. You get the memory so far and
one new piece of material, such as a shift's log. Write what later shifts must know, as JSON:
{"observations": [{"scope": "...", "status": "...", "text": "..."}]}

- scope is exactly one of: "plant", "line-<n>", "asset:<machine name in lower case>",
  "person:<first name in lower case>". A machine is asset:<machine> (asset:press 2); a sensor or
  anything else that belongs to a line goes under that line (line-3).
- status is one of: "open" (needs action or follow-up), "resolved" (an earlier item is now done or
  undone), "unconfirmed" (a suspicion nobody has checked), "info" (a standing request or fact).
- Write only what this material adds or changes. Do not restate an item that is already in the
  memory unless this material changes it; its first date is what later questions ask about.
- Keep faults, temporary fixes, bypasses, parts on order, changed limits or settings, requests from
  people, and anything that resolves or changes an earlier item. Skip routine work: checks and
  rounds that found nothing, cleared jams, changeovers, top-ups, fuses. Never pad: most shifts
  add zero to three observations.
- Each observation is one sentence under 20 words that is understood without the log: name the
  machine, its line, and the numbers or part numbers.
- Never state as fact what the log only suspects.
Return only the JSON."""

REFLECTOR_SYSTEM = """You condense the handover memory of a maintenance team. You get the current record
and the observations made since it was written. Write the new record, one line per item:
[<when>] (<scope>) <status>: <text>

- Drop an item that a later observation resolved, and a value that changed keeps only its newest
  value, with when it changed.
- A suspicion stays "unconfirmed" until someone confirms it or rules it out.
- Keep different lines and machines apart: the same sensor name on two lines is two items.
- Keep standing requests from people.
- Each line under 20 words. Stay under {budget} tokens; if you must cut, cut what nobody has to
  act on first.
Return only the record."""


def normalise_scope(scope: str) -> str:
    """Accept the usual variants ("line 3", "Asset: Press_2") instead of losing the observation."""
    s = " ".join(str(scope).strip().lower().replace("_", " ").split())
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"^line[ -]?(\d+)$", r"line-\1", s)
    return s


@dataclass(frozen=True)
class Observation:
    when: str
    scope: str
    status: str
    text: str

    def line(self) -> str:
        return f"[{self.when}] ({self.scope}) {self.status}: {self.text}"


def parse_observations(text: str, when: str) -> tuple[list[Observation], list[str]]:
    """The valid observations in an observer's answer, and what was wrong with the others."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        return [], ["the observer did not return JSON"]
    raw = data.get("observations") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], ["no 'observations' list"]
    kept: list[Observation] = []
    problems: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            problems.append("an observation that is not an object")
            continue
        scope = normalise_scope(item.get("scope", ""))
        status = str(item.get("status", "")).strip().lower()
        body = " ".join(str(item.get("text", "")).split())
        if not SCOPE.match(scope):
            problems.append(f"unknown scope {scope!r}")
        elif status not in STATUSES:
            problems.append(f"unknown status {status!r}")
        elif not body:
            problems.append("an empty observation")
        else:
            kept.append(Observation(when, scope, status, body[:300]))
    return kept, problems


@dataclass
class ObservationalMemory:
    call: ModelCall
    budget_tokens: int = 400  # the record, which goes into every prompt
    reflect_above: int = 600  # new observations, in tokens, that trigger a reflection
    journal: list[Observation] = field(default_factory=list)
    record: str = ""
    cut_lines: int = 0  # lines the last reflection had to cut to fit the budget
    _pending: list[Observation] = field(default_factory=list)

    def observe(self, when: str, material: str) -> tuple[list[Observation], list[str]]:
        user = f"<memory>\n{self.context()}\n</memory>\n\n<new material from {when}>\n{material}\n</new material>"
        observations, problems = parse_observations(self.call(OBSERVER_SYSTEM, user, "observer"), when)
        self.journal.extend(observations)
        self._pending.extend(observations)
        return observations, problems

    def pending_tokens(self) -> int:
        return estimate_tokens("\n".join(o.line() for o in self._pending))

    def due(self) -> bool:
        return self.pending_tokens() > self.reflect_above

    def reflect(self, attempts: int = 2) -> str:
        """Condense the record and the new observations into a record that fits the budget."""
        new = "\n".join(o.line() for o in self._pending) or "(none)"
        user = f"<record>\n{self.record or '(empty)'}\n</record>\n\n<new observations>\n{new}\n</new observations>"
        system = REFLECTOR_SYSTEM.format(budget=self.budget_tokens)
        candidate = ""
        for _ in range(attempts):
            candidate = self.call(system, user, "reflector").strip()
            size = estimate_tokens(candidate)
            if size <= self.budget_tokens:
                break
            user = (
                f"{user}\n\nYour last record was {size} tokens; the budget is {self.budget_tokens}. "
                f"Write it again, shorter:\n{candidate}"
            )
        self.record, self.cut_lines = fit(candidate, self.budget_tokens)
        self._pending = []
        return self.record

    def context(self) -> str:
        """What a new session gets: the record, then observations not yet reflected."""
        pending = "\n".join(o.line() for o in self._pending)
        return "\n".join(part for part in (self.record, pending) if part) or "(empty)"

    def recall(self, query: str, scope: str | None = None, k: int = 5) -> list[Observation]:
        """Observations about ``query``. ``scope`` narrows the search when anything matches it."""
        pool = in_scope(self.journal, scope, key=lambda o: o.scope) or list(self.journal)
        if not pool:
            return []
        hits = HybridIndex([o.line() for o in pool]).search(query, query, k=k)
        return [pool[int(h["index"])] for h in hits]

    def to_dict(self) -> dict[str, object]:
        return {
            "record": self.record,
            "journal": [asdict(o) for o in self.journal],
            "pending": [asdict(o) for o in self._pending],
        }

    def load(self, data: dict[str, object]) -> None:
        """Restore what ``to_dict`` saved, e.g. at the start of the next session."""
        self.record = str(data.get("record", ""))
        self.journal = [Observation(**o) for o in _dicts(data.get("journal"))]
        self._pending = [Observation(**o) for o in _dicts(data.get("pending"))]


def in_scope(items: Sequence[T], scope: str | None, key: Callable[[T], str]) -> list[T]:
    """Items whose scope matches: exactly, or by the whole name ("press 2" finds "asset:press 2",
    and never "asset:press 22"; "line-1" never finds "line-12")."""
    if not scope:
        return list(items)
    wanted = normalise_scope(scope)
    name = wanted.split(":", 1)[-1]
    return [i for i in items if key(i) == wanted or (name and key(i).split(":", 1)[-1] == name)]


_KEEP_ORDER = {"open": 0, "unconfirmed": 1, "info": 2, "resolved": 3}
_STATUS = re.compile(r"\)\s*(open|unconfirmed|info|resolved)\s*:")


def fit(record: str, budget: int) -> tuple[str, int]:
    """The record cut to the budget, least important lines first; returns how many lines went.

    Lines are kept in order of importance until the next one does not fit; nothing less important
    is kept after that, even if it is short, so a resolved item never outlives an open one."""
    lines = [line for line in record.splitlines() if line.strip()]
    if estimate_tokens("\n".join(lines)) <= budget:
        return "\n".join(lines), 0

    def rank(item: tuple[int, str]) -> tuple[int, int]:
        match = _STATUS.search(item[1])
        return (_KEEP_ORDER[match.group(1)] if match else 4, -item[0])  # then newest (lowest in the record) first

    kept: list[tuple[int, str]] = []
    for index, line in sorted(enumerate(lines), key=rank):
        if estimate_tokens("\n".join(text for _, text in [*kept, (index, line)])) > budget:
            break
        kept.append((index, line))
    kept.sort()
    return "\n".join(text for _, text in kept), len(lines) - len(kept)


def _dicts(value: object) -> Sequence[dict[str, str]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []
