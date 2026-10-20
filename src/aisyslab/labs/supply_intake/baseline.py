"""Deterministic solver: route by content, decode every format in code, count bins by pixels.

No model is called. That only works because every source here has a stable shape; the
free-text notes are the weak spot, handled with patterns that would break on a rephrasing.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from ...core.lab import RunContext
from .env import TODAY, SupplyIntakeEnv, count_bins, sniff

PHONE = r"\+\d{2} \d{2} \d{4} \d{4}"


def solve(env: SupplyIntakeEnv, ctx: RunContext) -> None:
    decoded: dict[str, tuple[str, str, bytes]] = {}
    for item in ctx.call(env.list_items):
        item_id = str(item["id"])
        data = ctx.call(env.fetch, item_id)
        decoded[item_id] = (str(item["name"]), sniff(data), data)

    stock: dict[str, int] = {}
    need: dict[str, int] = defaultdict(int)
    delayed: dict[str, str] = {}
    contacts: dict[str, str] = {}
    notes: list[str] = []
    images: dict[str, bytes] = {}
    for name, kind, data in decoded.values():
        if kind == "application/json":
            payload = json.loads(data)
            if isinstance(payload, dict) and "stock" in payload:
                stock.update({row["part"]: int(row["on_hand"]) for row in payload["stock"]})
        elif kind == "application/xml":
            for notice in ET.fromstring(data).iter("notice"):
                if notice.get("status") == "delayed" and (notice.get("revised") or "") > TODAY.isoformat():
                    delayed[notice.get("part", "")] = notice.get("supplier", "")
        elif kind == "text/csv":
            rows = list(csv.DictReader(io.StringIO(data.decode())))
            if rows and "qty_next_shift" in rows[0]:
                for row in rows:
                    need[row["part"]] += int(row["qty_next_shift"])
            elif rows and "phone" in rows[0]:
                contacts.update({row["supplier"]: row["phone"] for row in rows})
        elif kind == "image/png":
            images[name] = data
        elif kind == "text/plain":
            notes.append(data.decode())

    # A physical count beats the ERP figure.
    available = dict(stock)
    for note in notes:
        rack = re.search(r"rack for (P-\d{4}).*?attached as (\S+?)\.\s.*?hold (\d+)", note, re.S)
        if rack and rack.group(2) in images:
            full, _empty = count_bins(images[rack.group(2)])
            available[rack.group(1)] = full * int(rack.group(3))

    at_risk = [(available.get(part, 0) - need[part], part) for part in delayed if available.get(part, 0) < need[part]]
    _gap, part = min(at_risk)
    supplier = delayed[part]
    phone = contacts.get(supplier, "")
    for note in notes:
        if supplier in note:
            match = re.search(rf"[Cc]all .*? on ({PHONE})", note)
            if match:
                phone = match.group(1)
    ctx.submit(env.submit, part, supplier, phone, need[part] - available[part])
