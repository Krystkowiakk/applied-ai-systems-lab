"""Make a card and check it before anyone uses it.

Two checks, both by a vision model answering narrow questions, both decided in code:

- the illustration: does it contain any text or numbers, does it show the component, does it show
  anything unsafe? A rejected image is generated again once, with the reason in the prompt; if the
  second one fails too, the card goes out without an illustration, which is always safe;
- the finished card: the model reads every numbered step back, and code compares the reading with
  the procedure: the number of steps, and each step's critical value (a torque, a pressure, a part
  number). A card whose read-back disagrees is not released.

The model never decides that a card is fine; it reports what it sees.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .card import Procedure, render_card
from .images import ImageGenerator

# (system, question, png) -> the model's text
VisionCall = Callable[[str, str, bytes], str]

ILLUSTRATION_PROMPT = """Technical illustration for a maintenance work instruction card: {component}.
Clean flat style, white background, navy and grey lines, one amber highlight on the part being worked
on. Absolutely no text, letters, numbers, labels, arrows with words or logos anywhere in the image."""

ILLUSTRATION_CHECK = """You check an illustration before it goes on a maintenance work instruction card.
The card is about: {component}.
Look at every surface, dial, plate and edge. Answer in JSON only:
{{"marks": [{{"characters": "the letters or digits it looks like, e.g. G5", "where": "where it is"}}],
  "shows": "what the image shows, in a few words", "matches": true or false,
  "unsafe": true or false, "problem": "what is wrong, or empty"}}
- marks: every mark that resembles letters or digits, even if it is not a real word, small, blurred
  or on a dial or plate. Shapes that resemble no letter or digit (circles, bolts, stripes) are not
  marks. An empty list means you looked and found none.
- matches: the image shows that component, recognisably. The illustrations use one amber highlight
  to mark the part being worked on; it is not damage.
- unsafe: someone working without guarding, touching a pressurised part, a hand near moving parts."""

READ_BACK = """You read a printed maintenance work instruction card. Copy the numbered steps exactly
as printed, in order, one string per step, without the numbers. Answer in JSON only:
{"steps": ["...", "..."]}"""


@dataclass
class CardResult:
    png: bytes
    released: bool
    report: dict[str, Any] = field(default_factory=dict)
    illustrations: list[bytes] = field(default_factory=list)  # every generated image, used or not


def _json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def _reads_as_text(mark: Any) -> bool:
    """Code decides what counts as text. Digits, capitals and anything beyond plain lower-case
    letters always do ("G5", "50 BAR", "WARNING!", "Ø16"), and so does a short lower-case token
    ("psi"). A lower-case phrase or a long lower-case word is the model describing a shape
    ("two small circles", "hexagon"). A bare string cannot be told apart from a description, so it
    counts."""
    if not isinstance(mark, dict):
        return True
    shape = str(mark.get("characters", "")).strip()
    if not shape:
        return False
    if any(c.isdigit() or c.isupper() or (c.isalpha() and not c.isascii()) for c in shape):
        return True
    if not re.fullmatch(r"[a-z ]+", shape):  # lower case with symbols ("bar!", "50%") reads as text
        return any(c.isalnum() or c in "°%!#" for c in shape)
    return " " not in shape and len(shape) <= 4


def _describe(mark: Any) -> str:
    return f"'{mark.get('characters', '')}' ({mark.get('where', '')})" if isinstance(mark, dict) else str(mark)


def check_illustration(vision: VisionCall, image: bytes, procedure: Procedure) -> dict[str, Any]:
    seen = _json(vision(ILLUSTRATION_CHECK.format(component=procedure.component), "Check this image.", image))
    problems = []
    marks = seen.get("marks")
    if not isinstance(marks, list):
        problems.append("the check did not list the marks it saw")
    else:  # code decides what counts as text: a mark with a letter or a digit; a dot or a bolt head does not
        readable = [_describe(m) for m in marks if _reads_as_text(m)]
        if readable:
            problems.append("it contains marks that read as letters or digits: " + "; ".join(readable))
    if seen.get("matches") is not True:
        problems.append(f"it does not clearly show {procedure.component}")
    if seen.get("unsafe") is not False:
        problems.append("it shows something unsafe")
    return {"seen": seen, "ok": not problems, "problems": problems}


def read_back(vision: VisionCall, card: bytes, procedure: Procedure) -> dict[str, Any]:
    steps = _json(vision(READ_BACK, "Read the steps on this card.", card)).get("steps") or []
    read = [str(s) for s in steps] if isinstance(steps, list) else []
    problems = []
    if len(read) != len(procedure.steps):
        problems.append(f"read {len(read)} steps, the procedure has {len(procedure.steps)}")
    for number, (step, seen) in enumerate(zip(procedure.steps, read, strict=False), 1):
        # The value must stand on its own: "195 Nm" does not contain a reading of "95 Nm".
        if step.check and not re.search(rf"(?<![\w.]){re.escape(step.check)}(?!\w)", seen):
            problems.append(f"step {number}: '{step.check}' not read back (read: {seen!r})")
    return {"read": read, "ok": not problems, "problems": problems}


def make_card(
    procedure: Procedure, generator: ImageGenerator, vision: VisionCall | None, *, attempts: int = 2
) -> CardResult:
    """Generate an illustration (at most ``attempts`` times), render the card, read it back."""
    report: dict[str, Any] = {"procedure": procedure.id, "generator": generator.name, "illustrations": []}
    illustration, note, feedback = None, "No illustration.", ""
    images: list[bytes] = []
    for _ in range(attempts):
        prompt = ILLUSTRATION_PROMPT.format(component=procedure.component) + feedback
        image = generator.generate(prompt)
        images.append(image)
        verdict = check_illustration(vision, image, procedure) if vision else None
        report["illustrations"].append({"prompt": prompt, "check": verdict})
        if verdict is None or verdict["ok"]:
            illustration = image
            break
        feedback = "\nThe previous image was rejected because " + "; ".join(verdict["problems"]) + ". Avoid that."
        note = "No illustration: the generated images failed their check."
    png = render_card(procedure, illustration, note=note)
    if vision is None:
        report["read_back"] = None
        return CardResult(png, False, report | {"released": False, "why": "not checked: no vision model"}, images)
    checked = read_back(vision, png, procedure)
    report["read_back"] = checked
    report["released"] = checked["ok"]
    return CardResult(png, checked["ok"], report, images)
