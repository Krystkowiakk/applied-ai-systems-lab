"""Monday-morning intake for an operations manager: which supplier delay will stop a line?

Twelve items in mixed formats: an ERP stock export (JSON), the shift plan (CSV), supplier
delivery notices (XML), a supplier contact list (CSV), shop-floor notes (text) and a photo of
a kanban rack (PNG). Several files carry misleading names, so routing has to look at content.

The answer needs all of them: the delayed parts (XML), what the next shift consumes (CSV),
stock on hand (JSON, unless a physical count says otherwise), the physical count itself (the
photo plus the bin size from a note), and the supplier contact (corrected in a note).
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from PIL import Image, ImageDraw

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict

TODAY = date(2026, 3, 16)
RACK_ROWS, RACK_COLS = 4, 8
CELL, GAP, ORIGIN = 64, 8, (24, 56)
GREEN, RED = (46, 139, 87), (196, 64, 52)

PARTS = [
    ("P-4417", "Hydraulic seal kit"),
    ("P-2093", "Bearing 6205-2RS"),
    ("P-7781", "Proximity sensor M12"),
    ("P-3356", "Conveyor belt splice kit"),
    ("P-1188", "Shrink film 500 mm"),
    ("P-5120", "Welding wire 1.2 mm"),
    ("P-6604", "Pallet label roll"),
    ("P-8842", "Servo drive fuse 16 A"),
]
SUPPLIERS = [
    "Orvik Hydraulics",
    "Kestrel Bearings",
    "Quillon Sensors",
    "Vellmar Belting",
    "Brakka Packaging",
    "Weldora Wire",
]
PEOPLE = ["Marta Kowal", "Jonas Berg", "Elif Kaya", "Hugo Moreau", "Ines Costa", "Ravi Iyer", "Lena Dahl"]
LINES = ["L1", "L2", "L3", "L4"]


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    data: bytes


class SupplyIntakeEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        r = self.rng
        parts = r.sample(PARTS, 6)
        suppliers = r.sample(SUPPLIERS, 6)
        self.part_supplier = {p[0]: s for p, s in zip(parts, suppliers, strict=True)}
        critical, enough_delayed, short_on_time, resolved = (p[0] for p in parts[:4])
        self.critical = critical

        # Consumption of the next shift, split across lines.
        need = {p[0]: r.randint(20, 60) for p in parts}
        need[critical] = r.randint(45, 70)
        on_hand = {p[0]: need[p[0]] + r.randint(5, 40) for p in parts}
        on_hand[short_on_time] = need[short_on_time] - r.randint(5, 15)  # short, but delivery lands today
        on_hand[critical] = need[critical] + r.randint(5, 20)  # ERP says fine; the rack says otherwise

        self.bin_size = r.choice([4, 5, 6])
        full_bins = r.randint(3, 6)
        self.full_bins = full_bins
        physical = full_bins * self.bin_size
        self.shortfall = need[critical] - physical
        assert self.shortfall > 0

        people = r.sample(PEOPLE, 7)
        self.contacts = {s: (people[i], f"+44 20 7946 0{r.randint(100, 999)}") for i, s in enumerate(suppliers)}
        new_person = people[6]
        self.old_phone = self.contacts[self.part_supplier[critical]][1]
        self.new_phone = self.old_phone
        while self.new_phone == self.old_phone:
            self.new_phone = f"+44 20 7946 0{r.randint(100, 999)}"

        late = lambda days: (TODAY + timedelta(days=days)).isoformat()  # noqa: E731
        notices = [
            (critical, "delayed", TODAY.isoformat(), late(r.randint(2, 4))),
            (enough_delayed, "delayed", TODAY.isoformat(), late(r.randint(1, 3))),
            (short_on_time, "confirmed", TODAY.isoformat(), TODAY.isoformat()),
            (resolved, "resolved", (TODAY - timedelta(days=2)).isoformat(), (TODAY - timedelta(days=1)).isoformat()),
        ]
        r.shuffle(notices)
        xml = ['<?xml version="1.0" encoding="UTF-8"?>', f'<deliveryNotices generated="{TODAY.isoformat()}T05:10">']
        for part, status, original, revised in notices:
            xml.append(
                f'  <notice supplier="{self.part_supplier[part]}" part="{part}" original="{original}" revised="{revised}" status="{status}"/>'
            )
        xml.append("</deliveryNotices>")

        stock = {
            "exported_at": f"{TODAY.isoformat()}T05:30",
            "source": "ERP nightly export",
            "stock": [{"part": p, "description": d, "on_hand": on_hand[p], "unit": "pcs"} for p, d in parts],
        }

        plan_rows = ["line,part,qty_next_shift"]
        for p, _ in parts:
            lines = r.sample(LINES, 2) if need[p] > 30 else [r.choice(LINES)]
            first = need[p] // len(lines)
            split = [first] * (len(lines) - 1) + [need[p] - first * (len(lines) - 1)]
            plan_rows += [f"{line},{p},{qty}" for line, qty in zip(lines, split, strict=True)]

        contact_rows = ["supplier,contact,phone"] + [f"{s},{n},{ph}" for s, (n, ph) in self.contacts.items()]

        image_name = f"IMG_{r.randint(2000, 2999)}"
        notes = [
            (
                "shift_note_night.txt",
                "Night shift: L2 changeover took 40 minutes longer than planned. Two short stops on L4, no quality issues.",
            ),
            (
                "rack_count_note.txt",
                f"Counted the kanban rack for {critical} myself at 05:50, photo attached as {image_name}. Bins hold {self.bin_size} kits each. Green bins are full, red bins are empty.",
            ),
            (
                "supplier_contact_update.txt",
                f"Heads-up from purchasing: {self.part_supplier[critical]} changed their account manager. Call {new_person} on {self.new_phone} from now on; "
                f"the old number {self.old_phone} is disconnected.",
            ),
            (
                "it_notice.txt",
                "The ERP nightly export moved to 05:30. Old exports in the shared drive will be deleted on Friday.",
            ),
        ]

        items = [
            ("stock_export.txt", json.dumps(stock, indent=2).encode()),  # JSON behind a .txt name
            ("shift_plan.csv", "\n".join(plan_rows).encode()),
            ("delivery_notices.dat", "\n".join(xml).encode()),  # XML behind a .dat name
            ("supplier_contacts.csv", "\n".join(contact_rows).encode()),
            (image_name, _rack_png(critical, full_bins, random.Random(seed + 101))),  # PNG without an extension
            *((name, text.encode()) for name, text in notes),
            ("newsletter.html", b"<html><body><h1>Plant newsletter</h1><p>Summer party on 12 June.</p></body></html>"),
            ("old_stock_export.csv", b"part,on_hand\nP-0000,0\n"),
            ("canteen_menu.txt", b"Monday: soup of the day, chicken curry, vegetable lasagne."),
        ]
        r.shuffle(items)
        self.items = {f"I{i + 1:02d}": Item(f"I{i + 1:02d}", name, data) for i, (name, data) in enumerate(items)}
        self.need = need
        self.on_hand = on_hand

    # ----- API ----------------------------------------------------------------------------
    def list_items(self) -> list[dict[str, Any]]:
        self._api("list_items")
        return [{"id": i.id, "name": i.name, "bytes": len(i.data)} for i in self.items.values()]

    def fetch(self, item_id: str) -> bytes:
        self._api("fetch")
        item = self.items.get(item_id)
        if item is None:
            raise ToolError(f"No item {item_id}.", hint="Use ids from list_items.")
        return item.data

    def submit(self, part_number: str, supplier: str, contact_phone: str, shortfall_units: int) -> Verdict:
        self._api("submit")
        status = {
            "part_number": "ok" if part_number.strip().upper() == self.critical else "wrong",
            "supplier": "ok" if supplier.strip().lower() == self.part_supplier[self.critical].lower() else "wrong",
            "contact_phone": "ok"
            if _digits(contact_phone) == _digits(self.new_phone)
            else ("outdated" if _digits(contact_phone) == _digits(self.old_phone) else "wrong"),
            "shortfall_units": "ok" if shortfall_units == self.shortfall else "wrong",
        }
        accepted = all(v == "ok" for v in status.values())
        feedback = "Report accepted." if accepted else "; ".join(f"{k}: {v}" for k, v in status.items())
        return self._record(Verdict(accepted, "OK" if accepted else "FIELDS_REJECTED", feedback, {"fields": status}))


def sniff(data: bytes) -> str:
    """Route by content, not by file name."""
    head = data[:16].lstrip()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"<?xml"):
        return "application/xml"
    if head[:1] in (b"{", b"["):
        try:
            json.loads(data)
            return "application/json"
        except ValueError:
            pass
    if head.lower().startswith((b"<html", b"<!doctype html")):
        return "text/html"
    text = data.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    if (
        "," in first
        and len(text.splitlines()) > 1
        and all(line.count(",") == first.count(",") for line in text.splitlines()[1:] if line)
    ):
        return "text/csv"
    return "text/plain"


def count_bins(png: bytes) -> tuple[int, int]:
    """Deterministic validator for the rack photo: sample the centre of each bin."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    full = empty = 0
    for row in range(RACK_ROWS):
        for col in range(RACK_COLS):
            x = ORIGIN[0] + col * (CELL + GAP) + CELL // 2
            y = ORIGIN[1] + row * (CELL + GAP) + CELL // 2
            pixel = image.getpixel((x, y))
            assert isinstance(pixel, tuple)
            red, green = pixel[0], pixel[1]
            if green > red:
                full += 1
            else:
                empty += 1
    return full, empty


def _rack_png(part: str, full_bins: int, rng: random.Random) -> bytes:
    width = ORIGIN[0] * 2 + RACK_COLS * (CELL + GAP)
    height = ORIGIN[1] + RACK_ROWS * (CELL + GAP) + 24
    image = Image.new("RGB", (width, height), (232, 230, 224))
    draw = ImageDraw.Draw(image)
    draw.text((ORIGIN[0], 16), f"KANBAN RACK L3 - {part}", fill=(25, 25, 25))
    slots = set(rng.sample(range(RACK_ROWS * RACK_COLS), full_bins))
    for row in range(RACK_ROWS):
        for col in range(RACK_COLS):
            x = ORIGIN[0] + col * (CELL + GAP)
            y = ORIGIN[1] + row * (CELL + GAP)
            fill = GREEN if row * RACK_COLS + col in slots else RED
            draw.rectangle([x, y, x + CELL, y + CELL], fill=fill, outline=(30, 30, 30), width=2)
            draw.text((x + 4, y + 4), f"{row + 1}{chr(65 + col)}", fill=(250, 250, 250))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _digits(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())
