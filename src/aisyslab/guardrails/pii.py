"""Data classification and anonymisation in layers, with placeholders that can be reversed.

Before anything goes to a model, the text is classified: red if it holds payment data (an IBAN,
a card number), yellow if it holds personal data (names, e-mail addresses, phone numbers), green
otherwise. Red values never go to a cloud model. Personal data is replaced with placeholders
such as ``[PERSON_1]`` in layers: patterns first (exact and cheap), then a directory of known
people, then, optionally, a model for what both missed, preferably a local one. The mapping
stays in code, so a reply written with placeholders gets the real values back before it is sent.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

GREEN, YELLOW, RED = "green", "yellow", "red"
LEVELS = (GREEN, YELLOW, RED)

Finder = Callable[[str], list[str]]  # the values found in a text


def _luhn(digits: str) -> bool:
    total = 0
    for index, ch in enumerate(reversed(digits)):
        n = int(ch)
        if index % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _pattern(regex: str, check: Callable[[str], bool] = lambda _v: True) -> Finder:
    compiled = re.compile(regex)
    return lambda text: [m.group(0).strip() for m in compiled.finditer(text) if check(m.group(0))]


def _card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 13 <= len(digits) <= 19 and _luhn(digits)


@dataclass(frozen=True)
class Detector:
    kind: str  # the placeholder's name: EMAIL, PHONE, IBAN, CARD, PERSON
    level: str  # yellow or red
    find: Finder


PATTERNS = (
    Detector("IBAN", RED, _pattern(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    Detector("CARD", RED, _pattern(r"\b(?:\d[ -]?){12,18}\d\b", _card)),
    Detector("EMAIL", YELLOW, _pattern(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    Detector("PHONE", YELLOW, _pattern(r"\+\d{1,3}(?:[ -]?\d{1,4}){2,6}")),
)


def directory_detector(people: Iterable[str]) -> Detector:
    """Known people (a CRM, a staff list): full names first, then the parts of each name."""
    names = sorted({n for person in people for n in (person, *person.split()) if len(n) > 2}, key=len, reverse=True)
    compiled = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b") if names else None
    return Detector("PERSON", YELLOW, lambda text: [m.group(0) for m in compiled.finditer(text)] if compiled else [])


@dataclass
class Redacted:
    text: str
    level: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> real value
    found: dict[str, int] = field(default_factory=dict)  # kind -> how many distinct values

    def restore(self, text: str) -> str:
        """Put the real values back, e.g. into a reply written with placeholders."""
        for placeholder, value in self.mapping.items():
            text = text.replace(placeholder, value)
        return text


PLACEHOLDER = re.compile(r"\[(?:EMAIL|PHONE|IBAN|CARD|PERSON)_\d+\]")


class Anonymizer:
    def __init__(self, layers: Sequence[Detector] = PATTERNS) -> None:
        self.layers = list(layers)

    def classify(self, text: str) -> str:
        levels = [d.level for d in self.layers if d.find(text)]
        return RED if RED in levels else YELLOW if YELLOW in levels else GREEN

    def anonymize(self, text: str) -> Redacted:
        result = Redacted(text, self.classify(text))
        counters: dict[str, int] = {}
        for detector in self.layers:
            # Longest values first, so "Anna Kowalska" is replaced before "Anna".
            for value in sorted(set(detector.find(result.text)), key=len, reverse=True):
                if value in result.mapping.values() or PLACEHOLDER.fullmatch(value):
                    continue
                counters[detector.kind] = counters.get(detector.kind, 0) + 1
                placeholder = f"[{detector.kind}_{counters[detector.kind]}]"
                result.mapping[placeholder] = value
                result.text = re.sub(rf"(?<![\w@.]){re.escape(value)}(?![\w@])", placeholder, result.text)
        result.found = dict(counters)
        return result


def leaked(text: str, secrets: Iterable[str]) -> list[str]:
    """Which of the given values appear in a text, as whole words: a check on what is about to leave."""
    return [s for s in secrets if len(s) > 2 and re.search(rf"(?<![\w@.]){re.escape(s)}(?![\w@])", text)]
