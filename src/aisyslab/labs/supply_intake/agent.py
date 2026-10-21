"""Agent solver: code routes and decodes, the model reasons, vision reads the photo.

This is the architecture-first gateway: structured formats are decoded in code and handed
to the model as data, and a model is called on the image only because nothing else can read
it. The vision answer is checked against a pixel count without revealing that count.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import AgentSpec
from ...core.calculator import calculator_tool
from ...core.lab import RunContext
from ...core.messages import ImagePart, Message
from ...core.model import ChatModel
from ...core.tools import NoArgs, Tool, Toolbox, ToolError
from .env import TODAY, SupplyIntakeEnv, count_bins, sniff

SYSTEM = f"""You support an operations manager in a manufacturing plant. Today is {TODAY:%A %d %B %Y}.

Find the one part that will stop a line during the next shift because its delivery is late:
a part counts only if its revised delivery date is after today, the delivery is still
delayed, and what is available does not cover the next shift's planned consumption (summed
over all lines). A physical count of the rack overrides the ERP stock figure for that part.

Report the part number, its supplier, the phone number to call at that supplier (use the
latest instruction from the plant if the contact changed) and the shortfall in units
(planned consumption minus what is available).

Items are routed by content: read_item returns decoded data. Images need inspect_image.
Use calculate for arithmetic. Submit with submit_report and act on its feedback."""

TASK = "Work through the intake and submit the shortage report."

VISION_SYSTEM = """You read photos for an operations team. Answer the question about the image.
Then end with one line of JSON: {"full_bins": <int or null>, "empty_bins": <int or null>}.
Count carefully, row by row; use null if the image shows no bins."""


class ItemArgs(BaseModel):
    item_id: str = Field(description="Item id from list_items, e.g. I07.")


class ImageArgs(BaseModel):
    item_id: str
    question: str = Field(description="What you want to know about the image.")


class ReportArgs(BaseModel):
    part_number: str
    supplier: str
    contact_phone: str
    shortfall_units: int


def toolbox(env: SupplyIntakeEnv, model: ChatModel | None, ctx: RunContext) -> Toolbox:
    def list_items(_: NoArgs) -> list[dict[str, Any]]:
        rows = ctx.tool_call(env.list_items)
        for row in rows:
            row["type"] = sniff(ctx.tool_call(env.fetch, row["id"]))
        return rows

    def read_item(args: ItemArgs) -> Any:
        data = ctx.tool_call(env.fetch, args.item_id)
        return decode(data)

    def inspect_image(args: ImageArgs) -> dict[str, Any]:
        data = ctx.tool_call(env.fetch, args.item_id)
        if sniff(data) != "image/png":
            raise ToolError(f"{args.item_id} is not an image.", hint="Use read_item for documents.")
        if model is None:
            raise ToolError("No vision model is configured here.", hint="Answer from the other intake items.")
        text = ctx.side_call(
            model,
            system=VISION_SYSTEM,
            messages=[Message(role="user", content=args.question, images=[ImagePart("image/png", data)])],
            purpose="vision",
        )
        counts = _last_json(text)
        result: dict[str, Any] = {"answer": text.strip(), "counts": counts}
        full, empty = count_bins(data)
        if counts and counts.get("full_bins") is not None:
            consistent = counts.get("full_bins") == full and counts.get("empty_bins") in (empty, None)
            result["validator"] = (
                "consistent with a pixel check of the bin colours"
                if consistent
                else "INCONSISTENT with a pixel check of the bin colours; inspect again, row by row"
            )
        return result

    return Toolbox(
        [
            Tool(
                "list_items", "List intake items with the content type detected from their bytes.", NoArgs, list_items
            ),
            Tool(
                "read_item",
                "Decode one document item (JSON, CSV, XML, HTML, text) into readable data.",
                ItemArgs,
                read_item,
            ),
            Tool(
                "inspect_image",
                "Ask a vision model about an image item; its counts are cross-checked.",
                ImageArgs,
                inspect_image,
            ),
            calculator_tool(),
            Tool(
                "submit_report",
                "Send the shortage report to the verifier. Returns per-field feedback.",
                ReportArgs,
                lambda a: ctx.submit(
                    env.submit, a.part_number, a.supplier, a.contact_phone, a.shortfall_units, as_tool=True
                ).as_dict(),
            ),
        ]
    )


def decode(data: bytes) -> Any:
    kind = sniff(data)
    if kind == "application/json":
        return {"type": kind, "data": json.loads(data)}
    if kind == "text/csv":
        return {"type": kind, "rows": list(csv.DictReader(io.StringIO(data.decode())))}
    if kind == "application/xml":
        root = ET.fromstring(data)
        return {
            "type": kind,
            "root": root.tag,
            "attributes": dict(root.attrib),
            "elements": [{"tag": e.tag, **e.attrib} for e in root],
        }
    if kind == "text/html":
        return {"type": kind, "text": _html_text(data.decode(errors="replace"))}
    if kind == "image/png":
        return {"type": kind, "hint": "Binary image. Use inspect_image with a question."}
    return {"type": kind, "text": data.decode(errors="replace")}


def _last_json(text: str) -> dict[str, Any] | None:
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _html_text(markup: str) -> str:
    chunks: list[str] = []

    class Collector(HTMLParser):
        def handle_data(self, data: str) -> None:
            if data.strip():
                chunks.append(data.strip())

    Collector().feed(markup)
    return " ".join(chunks)


SPEC = AgentSpec(system=SYSTEM, task=lambda _env: TASK, toolbox=toolbox)
solve = SPEC.solve
