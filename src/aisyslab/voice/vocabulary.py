"""Grounding a transcript in the plant's vocabulary: machines, part numbers, lines and their names.

Speech-to-text gets the vocabulary up front, as a prompt or as keywords. What comes back is still
spoken language: "M C fourteen", "P ten forty", "line two". Grounding turns spoken numbers into
digits, reads IDs written in any of the ways a transcriber writes them, and checks every one
against the registry:

- an ID the registry holds is found;
- one it does not hold is tried with "-teen" and "-ty" swapped (fourteen and forty sound alike on a
  shop floor); exactly one match is a correction, several are a question;
- a part number whose registry name is not in the note, while another part's name is, is a
  question: the note says "relief valve", the number heard belongs to a pressure sensor;
- a machine named by what it is ("the transfer press") is found by name; "the press on line one",
  when line one has two presses, is a question.

Nothing is guessed: what grounding cannot settle goes back as a question for the technician.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER = re.compile(
    rf"\b(?:({'|'.join(TENS)})(?:[\s-]({'|'.join(UNITS)}))?|({'|'.join(TEENS)})|({'|'.join(UNITS)}))\b",
    re.IGNORECASE,
)
OH = re.compile(r"(?<=\d )oh\b|\boh(?= \d)", re.IGNORECASE)  # "one oh four oh": oh is zero next to digits
TEEN_TY = {f"1{d}": f"{d}0" for d in range(3, 10)} | {f"{d}0": f"1{d}" for d in range(3, 10)}

MACHINE = re.compile(r"\b(?:m\.?\s?c\.?|emcee)[\s.-]*(\d)[\s.-]?(\d)(?![\s.-]?\d)", re.IGNORECASE)
PART = re.compile(r"\b(?:p|pee)[\s.-]*(\d)[\s,.-]?(\d)[\s,.-]?(\d)[\s,.-]?(\d)(?![\s,.-]?\d)", re.IGNORECASE)
LINE = re.compile(r"\bline\s*(\d)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Term:
    id: str
    kind: str  # "machine", "part" or "line"
    name: str = ""
    line: str = ""  # machines: the line they stand on


@dataclass
class Grounded:
    transcript: str
    normalised: str
    found: list[dict[str, str]] = field(default_factory=list)
    unsure: list[dict[str, Any]] = field(default_factory=list)

    def ids(self, kind: str) -> list[str]:
        return [f["id"] for f in self.found if f["kind"] == kind]

    def as_dict(self) -> dict[str, Any]:
        return {"transcript": self.transcript, "found": self.found, "unsure": self.unsure}


def spoken_numbers(text: str) -> str:
    """'P ten forty' -> 'P 10 40'; 'forty-two' -> '42'; 'one oh four oh' -> '1 0 4 0'."""

    def value(m: re.Match[str]) -> str:
        tens, unit, teen, single = m.groups()
        if tens:
            return str(TENS[tens.lower()] + (UNITS[unit.lower()] if unit else 0))
        return str(TEENS[teen.lower()] if teen else UNITS[single.lower()])

    return OH.sub("0", NUMBER.sub(value, text))


class Vocabulary:
    def __init__(self, terms: Sequence[Term]) -> None:
        self.terms = {t.id: t for t in terms}

    def keywords(self) -> list[str]:
        """What speech-to-text is told to expect: every ID and every name."""
        return [t.id for t in self.terms.values()] + sorted({t.name for t in self.terms.values() if t.name})

    def ground(self, transcript: str) -> Grounded:
        text = spoken_numbers(transcript)
        grounded = Grounded(transcript, text)
        lowered = text.lower()
        for pattern, kind, prefix in ((MACHINE, "machine", "MC-"), (PART, "part", "P-")):
            for m in pattern.finditer(text):
                self._check(grounded, m.group(0).strip(), kind, prefix + "".join(m.groups()))
        for m in LINE.finditer(text):
            if f"L{m[1]}" in self.terms:
                grounded.found.append({"heard": m.group(0), "id": f"L{m[1]}", "kind": "line", "how": "spoken"})
        machines = grounded.ids("machine")
        for term in self.terms.values():
            named = term.kind == "machine" and term.name and re.search(rf"\b{re.escape(term.name)}\b", lowered)
            if named and term.id not in machines:
                grounded.found.append({"heard": term.name, "id": term.id, "kind": "machine", "how": "name"})
        if not grounded.ids("machine"):
            for m in re.finditer(r"\b(press|cell)\b[^.]{0,25}?\bline\s*(\d)\b", lowered):
                candidates = [
                    t.id for t in self.terms.values() if t.kind == "machine" and t.line == f"L{m[2]}" and m[1] in t.name
                ]
                if len(candidates) == 1:
                    grounded.found.append(
                        {"heard": m.group(0), "id": candidates[0], "kind": "machine", "how": "kind and line"}
                    )
                elif candidates:
                    grounded.unsure.append({"heard": m.group(0), "candidates": candidates, "why": "more than one fits"})
        self._cross_check_parts(grounded, lowered)
        return grounded

    def _check(self, grounded: Grounded, heard: str, kind: str, candidate: str) -> None:
        if candidate in self.terms:
            how = "exact" if heard.upper() == candidate else "spoken"
            grounded.found.append({"heard": heard, "id": candidate, "kind": kind, "how": how})
            return
        prefix, digits = candidate.split("-")
        pairs = [digits[i : i + 2] for i in range(0, len(digits), 2)]
        options = [{p, TEEN_TY.get(p, p)} for p in pairs]
        variants = {f"{prefix}-{''.join(v)}" for v in product(*options)} - {candidate}
        known = sorted(v for v in variants if v in self.terms)
        if len(known) == 1:
            grounded.found.append({"heard": heard, "id": known[0], "kind": kind, "how": "teen/ty"})
        else:
            why = "more than one fits" if known else "not in the registry"
            grounded.unsure.append({"heard": heard, "candidates": known, "why": why})

    def _cross_check_parts(self, grounded: Grounded, lowered: str) -> None:
        named = [t for t in self.terms.values() if t.kind == "part" and t.name and t.name in lowered]
        for found in [f for f in grounded.found if f["kind"] == "part"]:
            term = self.terms[found["id"]]
            if term.name in lowered or not named:
                continue
            grounded.found.remove(found)
            names = " or ".join(t.name for t in named)
            grounded.unsure.append(
                {
                    "heard": found["heard"],
                    "candidates": [found["id"], *[t.id for t in named]],
                    "why": f"the note says {names}; {found['id']} is a {term.name}",
                }
            )
