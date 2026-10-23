"""A work instruction card, rendered in code from the procedure's fields.

Every word and number on the card comes from the procedure and is drawn by code: image models
misspell and invent numbers, and a wrong torque value on a card at the machine is a hazard. The
only generated part is the illustration, and it is not allowed to contain text at all.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1240, 1754  # A4 at 150 dpi
MARGIN = 48
NAVY = (20, 33, 61)
ACCENT = (226, 132, 22)
INK = (34, 34, 34)
MUTED = (110, 116, 128)
RULE = (214, 219, 228)


@dataclass(frozen=True)
class Step:
    text: str
    check: str = ""  # a value the read-back must find in this step, exactly as written


@dataclass(frozen=True)
class Procedure:
    id: str
    title: str
    machine: str
    revision: str
    component: str  # what the illustration shows, in words, for the image model and the check
    steps: tuple[Step, ...]
    warnings: tuple[str, ...]
    ppe: tuple[str, ...]


PROCEDURES = {
    "HP-07": Procedure(
        "HP-07",
        "Replace a hydraulic hose",
        "MC-14 transfer press",
        "C",
        "a hydraulic hose with two crimped steel fittings, running from a valve block to a cylinder on an "
        "industrial press",
        (
            Step("Lock out MC-14 and test for zero energy."),
            Step("Release the accumulator: the gauge must read 0 bar.", "0 bar"),
            Step("Remove the hose and cap both ports."),
            Step("Fit hose P-2210 without twisting it.", "P-2210"),
            Step("Torque the M16 fittings to 95 Nm, stainless fittings to 80 Nm.", "95 Nm"),
            Step("Run at 50 bar for 5 minutes and check every fitting for leaks.", "50 bar"),
        ),
        ("Oil under pressure can pierce skin. Never feel for a leak with your hand.",),
        ("Safety glasses", "Oil-resistant gloves"),
    ),
    "FE-03": Procedure(
        "FE-03",
        "Change the coolant filter element",
        "MC-17 welding cell",
        "A",
        "a cylindrical coolant filter housing on a pipe, with one hand valve before it and one after it",
        (
            Step("Stop the coolant pump and lock it out."),
            Step("Close valve V1, then valve V2.", "V1"),
            Step("Open the drain and wait until the housing is empty."),
            Step("Replace filter element P-1314 and its O-ring.", "P-1314"),
            Step("Open V2 first, then V1, and vent the housing.", "V2 first"),
            Step("Start the pump: the pressure drop must stay below 0.5 bar.", "0.5 bar"),
        ),
        ("Coolant can be hot. Let it cool below 40 °C before you open the housing.",),
        ("Safety glasses", "Nitrile gloves"),
    ),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def wrap(text: str, typeface: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if typeface.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def render_card(procedure: Procedure, illustration: bytes | None, *, note: str = "") -> bytes:
    """The card as PNG: header, safety box, illustration (or a plain frame), numbered steps, footer."""
    card = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, WIDTH, 160), fill=NAVY)
    draw.text((MARGIN, 34), procedure.title, font=font(52), fill="white")
    meta = f"{procedure.id}  |  revision {procedure.revision}  |  {procedure.machine}"
    draw.text((MARGIN, 104), meta, font=font(28), fill=(196, 206, 224))

    box = (WIDTH - MARGIN - 560, 200, WIDTH - MARGIN, 760)
    if illustration is not None:
        picture = Image.open(io.BytesIO(illustration)).convert("RGB")
        picture.thumbnail((box[2] - box[0], box[3] - box[1]))
        card.paste(picture, (box[0] + (560 - picture.width) // 2, box[1] + (560 - picture.height) // 2))
    else:
        draw.rectangle(box, fill=(246, 247, 250))
        for i, line in enumerate(wrap(note or "No illustration.", font(24), 480)):
            draw.text((box[0] + 40, box[1] + 250 + 32 * i), line, font=font(24), fill=MUTED)
    draw.rectangle(box, outline=RULE, width=2)

    left = (MARGIN, 200, box[0] - 40)
    y = left[1]
    draw.text((left[0], y), "Before you start", font=font(30), fill=NAVY)
    y += 52
    for item in procedure.ppe:
        draw.text((left[0], y), f"- {item}", font=font(26), fill=INK)
        y += 38
    y += 24
    for warning in procedure.warnings:
        lines = wrap(warning, font(26), left[2] - left[0] - 40)
        height = 24 + 36 * len(lines) + 16
        draw.rectangle((left[0], y, left[2], y + height), fill=(253, 244, 232))
        draw.rectangle((left[0], y, left[0] + 8, y + height), fill=ACCENT)
        for i, line in enumerate(lines):
            draw.text((left[0] + 28, y + 20 + 36 * i), line, font=font(26), fill=INK)
        y += height + 20

    y = max(y, box[3]) + 50
    draw.line((MARGIN, y - 20, WIDTH - MARGIN, y - 20), fill=RULE, width=2)
    for number, step in enumerate(procedure.steps, 1):
        draw.ellipse((MARGIN, y, MARGIN + 56, y + 56), fill=NAVY)
        label = str(number)
        draw.text((MARGIN + 28 - font(30).getlength(label) / 2, y + 10), label, font=font(30), fill="white")
        lines = wrap(step.text, font(32), WIDTH - 2 * MARGIN - 90)
        for i, line in enumerate(lines):
            draw.text((MARGIN + 90, y + 8 + 44 * i), line, font=font(32), fill=INK)
        y += max(80, 44 * len(lines) + 36)
    footer = f"Printed from {procedure.id} revision {procedure.revision}. Check the revision before use."
    draw.text((MARGIN, HEIGHT - 70), footer, font=font(22), fill=MUTED)
    out = io.BytesIO()
    card.save(out, "PNG")
    return out.getvalue()
