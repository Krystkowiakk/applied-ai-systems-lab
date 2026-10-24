"""``aisyslab card``: make a work instruction card, check it, and say whether it may be used."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..core.messages import ImagePart, Message, Usage
from ..core.model import ChatModel
from ..providers import PROVIDERS, make_model
from .card import PROCEDURES
from .check import VisionCall, make_card
from .images import DiagramImages, ImageGenerator, OpenAIImages


def add_parser(sub: Any) -> None:
    card = sub.add_parser("card", help="Make a work instruction card, check it with a vision model, release it or not.")
    card.add_argument("procedure", choices=sorted(PROCEDURES))
    card.add_argument(
        "--images", choices=["diagram", "openai"], default="diagram", help="Where the illustration comes from."
    )
    card.add_argument("--image-model", default="gpt-image-1-mini")
    card.add_argument("--provider", choices=PROVIDERS, default=None, help="A vision model for the checks.")
    card.add_argument("--model", default=None)
    card.add_argument("--effort", default=None)
    card.add_argument("--out", type=Path, default=Path("runs/cards"))


class _Vision:
    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.usage = Usage()

    def __call__(self, system: str, question: str, png: bytes) -> str:
        message = Message(role="user", content=question, images=[ImagePart("image/png", png)])
        turn = self.model.complete(system=system, messages=[message], tools=[])
        self.usage = self.usage + turn.usage
        return turn.text


def run(args: argparse.Namespace) -> int:
    if (args.model or args.effort) and not args.provider:
        print("error: --model and --effort need --provider", file=sys.stderr)
        return 2
    procedure = PROCEDURES[args.procedure]
    generator: ImageGenerator = OpenAIImages(args.image_model) if args.images == "openai" else DiagramImages()
    vision = _Vision(make_model(args.provider, args.model, effort=args.effort)) if args.provider else None
    checker: VisionCall | None = vision
    result = make_card(procedure, generator, checker)
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"{procedure.id}.png"
    png.write_bytes(result.png)
    (args.out / f"{procedure.id}.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    for i, image in enumerate(result.illustrations, 1):
        (args.out / f"{procedure.id}-illustration-{i}.png").write_bytes(image)
    for i, attempt in enumerate(result.report["illustrations"], 1):
        check = attempt["check"]
        verdict = "not checked" if check is None else ("accepted" if check["ok"] else "; ".join(check["problems"]))
        print(f"Illustration {i}: {verdict}")
    read = result.report.get("read_back")
    if read is not None:
        print("Read-back: " + ("matches the procedure" if read["ok"] else "; ".join(read["problems"])))
    print(f"Released: {result.released}  ({png})")
    if vision:
        print(f"Vision tokens: {vision.usage.input_tokens} in, {vision.usage.output_tokens} out")
    return 0 if result.released or vision is None else 1
