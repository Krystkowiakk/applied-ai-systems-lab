"""Visual work instructions: a card drawn in code, an illustration checked before use, a read-back."""

from __future__ import annotations

import io
import json

from PIL import Image

from aisyslab.visual import PROCEDURES, DiagramImages, Procedure, Step, make_card, render_card
from aisyslab.visual.check import ILLUSTRATION_PROMPT, _reads_as_text

HOSE = PROCEDURES["HP-07"]
CLEAN = {"marks": [], "shows": "a hose", "matches": True, "unsafe": False, "problem": ""}
LETTERED = CLEAN | {"marks": [{"characters": "G5", "where": "valve block"}]}


def read_back_of(procedure: Procedure, *, change: tuple[int, str, str] | None = None) -> dict[str, list[str]]:
    steps = [s.text for s in procedure.steps]
    if change:
        index, old, new = change
        steps[index] = steps[index].replace(old, new)
    return {"steps": steps}


class Vision:
    """Plays the vision model: one answer per illustration check, then the card's read-back."""

    def __init__(self, checks: list[dict[str, object] | str], read: dict[str, list[str]]) -> None:
        self.checks, self.read, self.calls = list(checks), read, []

    def __call__(self, system: str, question: str, png: bytes) -> str:
        self.calls.append(system[:40])
        if system.startswith("You check an illustration"):
            answer = self.checks.pop(0)
            return answer if isinstance(answer, str) else json.dumps(answer)
        return json.dumps(self.read)


class Recording(DiagramImages):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        return super().generate(prompt)


def test_the_card_is_drawn_in_code_with_or_without_an_illustration() -> None:
    long = Procedure(
        "X-01", "A long one", "MC-11 press 1", "A", "a press", (Step("word " * 80),), ("careful " * 30,), ("Gloves",)
    )
    for procedure, picture in ((HOSE, DiagramImages().generate("hose")), (long, None)):
        card = Image.open(io.BytesIO(render_card(procedure, picture)))
        assert card.size == (1240, 1754) and card.getpixel((10, 10)) == (20, 33, 61)


def test_what_counts_as_text_is_decided_in_code() -> None:
    marks = [{"characters": c, "where": "x"} for c in ("G5", "6", "00", "two small circles", "•", "")]
    assert [_reads_as_text(m) for m in marks] == [True, True, True, False, False, False]
    assert _reads_as_text("a 6 on the valve block")  # a bare description cannot be told apart: it counts


def test_a_lettered_illustration_is_drawn_again_with_the_reason() -> None:
    generator = Recording()
    vision = Vision([LETTERED, CLEAN], read_back_of(HOSE))
    result = make_card(HOSE, generator, vision)
    assert result.released and len(result.illustrations) == 2
    assert generator.prompts[0] == ILLUSTRATION_PROMPT.format(component=HOSE.component)
    assert "rejected because it contains marks that read as letters or digits: 'G5'" in generator.prompts[1]
    assert [a["check"]["ok"] for a in result.report["illustrations"]] == [False, True]


def test_two_failed_illustrations_leave_a_card_without_one() -> None:
    vision = Vision([LETTERED, "not json"], read_back_of(HOSE))
    result = make_card(HOSE, DiagramImages(), vision)
    checks = [a["check"] for a in result.report["illustrations"]]
    assert checks[1]["problems"][0] == "the check did not list the marks it saw"
    assert result.released  # a card without a picture is still a correct card


def test_a_card_that_reads_back_differently_is_not_released() -> None:
    vision = Vision([CLEAN], read_back_of(HOSE, change=(4, "95 Nm", "59 Nm")))
    result = make_card(HOSE, DiagramImages(), vision)
    assert not result.released
    assert result.report["read_back"]["problems"] == [
        "step 5: '95 Nm' not read back (read: 'Torque the M16 fittings to 59 Nm, stainless fittings to 80 Nm.')"
    ]
    short = Vision([CLEAN], {"steps": [s.text for s in HOSE.steps[:5]]})
    assert (
        "read 5 steps, the procedure has 6" in make_card(HOSE, DiagramImages(), short).report["read_back"]["problems"]
    )


def test_without_a_vision_model_nothing_is_released() -> None:
    result = make_card(PROCEDURES["FE-03"], DiagramImages(), None)
    assert not result.released and result.report["why"] == "not checked: no vision model"


def test_a_misread_that_contains_the_value_is_still_a_misread() -> None:
    for old, new in (("95 Nm", "195 Nm"), ("0 bar", "50 bar"), ("50 bar", "150 bar")):
        index = next(i for i, s in enumerate(HOSE.steps) if s.check == old)
        result = make_card(HOSE, DiagramImages(), Vision([CLEAN], read_back_of(HOSE, change=(index, old, new))))
        assert not result.released, (old, new)


def test_capitals_symbols_and_digits_read_as_text() -> None:
    marks = ("HYDRAULIC", "50 BAR", "PSI 3000", "WARNING!", "Ø16", "0-250 bar", "psi")
    assert all(_reads_as_text({"characters": m, "where": "x"}) for m in marks)
    assert not any(_reads_as_text({"characters": m, "where": "x"}) for m in ("two small circles", "hexagon", "•"))
