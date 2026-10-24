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

Finder = Callable[[str], list[tuple[int, int]]]  # where the values are in a text: (start, end)


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
    return lambda text: [m.span() for m in compiled.finditer(text) if check(m.group(0))]


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
    """Known people (a CRM, a staff list), in any case: full names first, then the parts of each name."""
    names = sorted({n for person in people for n in (person, *person.split()) if len(n) > 2}, key=len, reverse=True)
    alternatives = "|".join(re.escape(n) for n in names)
    compiled = re.compile(rf"\b({alternatives})\b", re.IGNORECASE) if names else None
    return Detector("PERSON", YELLOW, lambda text: [m.span() for m in compiled.finditer(text)] if compiled else [])


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
        """Replace exactly what each detector found, where it found it: what is detected is replaced."""
        result = Redacted(text, self.classify(text))
        counters: dict[str, int] = {}
        known: dict[tuple[str, str], str] = {}  # (kind, value) -> placeholder; a name in any case is one name
        for detector in self.layers:
            spans = [s for s in detector.find(result.text) if not PLACEHOLDER.fullmatch(result.text[s[0] : s[1]])]
            for start, end in spans:  # numbered in reading order
                value = result.text[start:end]
                key = (detector.kind, value.lower() if detector.kind == "PERSON" else value)
                if key not in known:
                    counters[detector.kind] = counters.get(detector.kind, 0) + 1
                    known[key] = f"[{detector.kind}_{counters[detector.kind]}]"
                    result.mapping[known[key]] = value
            for start, end in reversed(spans):  # replaced from the end, so earlier offsets stay valid
                value = result.text[start:end]
                key = (detector.kind, value.lower() if detector.kind == "PERSON" else value)
                result.text = result.text[:start] + known[key] + result.text[end:]
        result.found = dict(counters)
        return result


SEPARATORS = re.compile(r"[\s.()/-]")


def leaked(text: str, secrets: Iterable[str]) -> list[str]:
    """Which of the given values appear in a text: a check on what is about to leave, independent of
    how the anonymiser found them. Names and words in any case, as whole words; numbers (accounts,
    cards, phones) with or without separators."""
    squeezed = SEPARATORS.sub("", text).lower()
    found = []
    for secret in secrets:
        if len(secret) <= 2:
            continue
        if sum(c.isdigit() for c in secret) >= 6:
            if SEPARATORS.sub("", secret).lower() in squeezed:
                found.append(secret)
        elif re.search(rf"(?<!\w){re.escape(secret)}(?!\w)", text, re.IGNORECASE):
            found.append(secret)
    return found
