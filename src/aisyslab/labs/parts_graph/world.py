"""One seed's plant documents: six tables, a handful of notes, and the answer they imply.

The tables (bill of materials, installation register, layout, line plan, order book, part master)
are machine-readable. The notes are not: a supplier's delay notice, engineering change notes,
changeover notes, planning minutes, logbook entries, and notes that state nothing the graph needs.
Each seed draws one to three of five traps into the notes:

- moved: a machine carrying the late part moved to another line after the layout was drawn;
- planned: a machine on the path will move to another line, but not before next month;
- swapped: an assembly with the late part was taken off its machine and sent to stores;
- negation: a logbook says a machine does *not* use an assembly with the late part;
- withdrawn: an alternate part was approved for both assemblies, then withdrawn for one.

The answer comes from the world model, not from the documents. Each note also carries the facts a
careful reader extracts from it, in the parser's format, so tests can check the whole pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

TODAY = date(2026, 10, 26)
AUDIT = date(2026, 6, 30)
LINES = ("L1", "L2", "L3", "L4")
TRAPS = ("moved", "planned", "swapped", "negation", "withdrawn")
SUPPLIERS = ("Kestrel Seals", "Brennmoor Hydraulics", "Ostara Drives", "Lumen Sensors")
PART_KINDS = (
    "seal kit",
    "proportional valve",
    "pressure sensor",
    "gear pump",
    "solenoid coil",
    "filter element",
    "servo drive",
    "encoder",
    "accumulator",
    "hose set",
    "limit switch",
    "check valve",
    "bearing set",
    "coupling",
    "relief valve",
    "flow meter",
)
ASSEMBLY_KINDS = (
    "hydraulic pack",
    "clamp unit",
    "ejector unit",
    "feed unit",
    "cooling unit",
    "lube unit",
    "drive unit",
    "transfer unit",
    "press head",
    "tool changer",
)
MACHINE_KINDS = (
    "press",
    "press",
    "transfer press",
    "trim press",
    "robot cell",
    "welding cell",
    "assembly cell",
    "press",
)
PRODUCT_KINDS = ("bracket", "housing", "cover", "flange", "mounting plate", "hinge", "clamp", "frame")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    kind: str  # "table" or "note"
    date: str
    text: str
    facts: tuple[str, ...] = ()  # notes: what a careful reader extracts, in the parser's format

    def listing(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "kind": self.kind, "date": self.date}

    def as_dict(self) -> dict[str, str]:
        return self.listing() | {"text": self.text}


@dataclass(frozen=True)
class World:
    documents: tuple[Document, ...]
    question: str
    part: str
    arrival: str
    answer: dict[str, list[str]]
    traps: tuple[str, ...]


def long_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _days(d: date, lo: int, hi: int, r: random.Random) -> date:
    return d + timedelta(days=r.randint(lo, hi))


def build(seed: int) -> World:
    r = random.Random(seed * 131 + 9)
    traps = tuple(sorted(r.sample(TRAPS, r.choice((1, 2, 2, 3)))))

    machines = [f"MC-{n}" for n in range(11, 19)]
    placed = machines[:]
    r.shuffle(placed)
    layout = {m: LINES[i // 2] for i, m in enumerate(placed)}
    machine_kind = dict(zip(machines, r.sample(MACHINE_KINDS, len(machines)), strict=True))
    products = [f"FG-{n}" for n in range(201, 209)]
    product_kind = dict(zip(products, r.sample(PRODUCT_KINDS, len(products)), strict=True))
    shuffled = products[:]
    r.shuffle(shuffled)
    makes = {line: sorted(shuffled[2 * i : 2 * i + 2]) for i, line in enumerate(LINES)}
    line_of = {p: line for line, ps in makes.items() for p in ps}
    assemblies = [f"A-{n}" for n in range(21, 31)]
    assembly_kind = dict(zip(assemblies, r.sample(ASSEMBLY_KINDS, len(assemblies)), strict=True))
    pool = [f"P-{n}" for n in sorted(r.sample(range(1010, 1990), 18))]
    r.shuffle(pool)
    bom_parts, (alt, alt2) = pool[:16], pool[16:]
    part_kind = dict(zip(bom_parts, r.sample(PART_KINDS, len(PART_KINDS)), strict=True))
    supplier = {p: r.choice(SUPPLIERS) for p in pool}
    part = r.choice(bom_parts)
    part_kind[alt] = part_kind[part]
    supplier[alt] = r.choice([s for s in SUPPLIERS if s != supplier[part]])

    # The two real paths: the late part in assemblies x and y, on machines mx and my, on two lines.
    x, y = r.sample(assemblies, 2)
    others = [a for a in assemblies if a not in (x, y)]
    w = others.pop(r.randrange(len(others))) if "swapped" in traps else None
    mx, my = r.sample(machines, 2)
    while layout[mx] == layout[my]:
        mx, my = r.sample(machines, 2)
    final = dict(layout)
    moved_on = None
    if "moved" in traps:
        final[mx] = r.choice([line for line in LINES if line not in (layout[mx], layout[my])])
        moved_on = _days(AUDIT, 60, 110, r)
    affected = {final[mx], final[my]}
    spare = [m for m in machines if m not in (mx, my) and final[m] not in affected]
    r.shuffle(spare)
    mw = spare.pop() if w else None
    mn = spare.pop() if "negation" in traps else None
    mp = lp = planned_on = None
    if "planned" in traps:
        mp = my if moved_on else r.choice((mx, my))
        lp = r.choice([line for line in LINES if line != final[mp]])
        planned_on = _days(TODAY, 24, 40, r)

    register: dict[str, list[str]] = {m: [] for m in machines}
    register[mx].append(x)
    register[my].append(y)
    if mw and w:
        register[mw].append(w)
    r.shuffle(others)
    empty = [m for m in machines if not register[m]]
    for m, a in zip(empty, others, strict=False):
        register[m].append(a)
    for a in others[len(empty) :]:
        register[r.choice(machines)].append(a)
    carriers = {x, y} | ({w} if w else set())
    rest = [p for p in bom_parts if p != part]
    bom = {a: sorted([*r.sample(rest, 2), part]) if a in carriers else sorted(r.sample(rest, 3)) for a in assemblies}
    a2 = r.choice([a for a in assemblies if a not in carriers])  # an unrelated approval, off the path
    p2 = r.choice(bom[a2])
    part_kind[alt2] = part_kind[p2]

    arrival = _days(TODAY, 18, 27, r)
    numbers = r.sample(range(5501, 5560), 3 * len(products))
    orders: list[dict[str, Any]] = []
    for i, product in enumerate(p for p in products for _ in range(3)):
        due = _days(TODAY, 2, 45, r)
        if due == arrival:
            due += timedelta(days=1)
        shipped = due < TODAY + timedelta(days=10) and r.random() < 0.3
        quantity = r.randrange(100, 900, 50)
        orders.append(
            {
                "order": f"SO-{numbers[i]}",
                "product": product,
                "quantity": quantity,
                "due": due,
                "status": "shipped" if shipped else "open",
            }
        )
    orders.sort(key=lambda o: o["order"])

    def due_before(line: str) -> list[str]:
        held = [o for o in orders if line_of[o["product"]] == line]
        if not any(o["status"] == "open" and o["due"] < arrival for o in held):
            held[0].update(status="open", due=_days(TODAY, 3, 12, r))
        return sorted(o["order"] for o in held if o["status"] == "open" and o["due"] < arrival)

    answer = {"lines": sorted(affected), "at_risk": due_before(final[my]), "saved": due_before(final[mx])}

    tables = [
        _table(
            "bom",
            "Bill of materials (engineering release, 12 May 2026)",
            "2026-05-12",
            "assembly,assembly_description,part,quantity",
            [f"{a},{assembly_kind[a]},{p},{r.choice((1, 1, 2))}" for a in assemblies for p in bom[a]],
        ),
        _table(
            "register",
            "Installation register (audit of 30 June 2026)",
            AUDIT.isoformat(),
            "machine,assembly",
            [f"{m},{a}" for m in machines for a in sorted(register[m])],
        ),
        _table(
            "layout",
            "Plant layout (30 June 2026)",
            AUDIT.isoformat(),
            "machine,machine_description,line",
            [f"{m},{machine_kind[m]},{layout[m]}" for m in machines],
        ),
        _table(
            "line-plan",
            "Line plan, fourth quarter 2026",
            "2026-09-28",
            "line,product,product_description",
            [f"{line},{p},{product_kind[p]}" for line in LINES for p in makes[line]],
        ),
        _table(
            "orders",
            "Order book (26 October 2026)",
            TODAY.isoformat(),
            "order,product,quantity,due,status",
            [f"{o['order']},{o['product']},{o['quantity']},{o['due'].isoformat()},{o['status']}" for o in orders],
        ),
        _table(
            "parts",
            "Part master",
            "2026-10-01",
            "part,description,supplier",
            [f"{p},{part_kind.get(p, 'spare')},{supplier[p]}" for p in sorted(pool)],
        ),
    ]

    ecn = sorted(r.sample(range(101, 160), 3))
    notes: list[Document] = []
    noticed = TODAY - timedelta(days=2)
    notes.append(
        _note(
            f"notice-{part.lower()}",
            f"{supplier[part]}: {part} delayed",
            noticed,
            f"From: order desk, {supplier[part]}\nTo: plant planning\n\n"
            f"A tooling failure at our plant has stopped production of {part} ({part_kind[part]}). We will not "
            f"ship {part} before {long_date(arrival)}. Open call-offs move to that date; we will confirm the exact "
            "day next week.",
        )
    )
    approved = _days(AUDIT, 40, 70, r)
    both = "withdrawn" in traps
    where = f"the {x} {assembly_kind[x]}" + (f" and in the {y} {assembly_kind[y]}" if both else "")
    notes.append(
        _note(
            f"ecn-2026-{ecn[0]}",
            f"ECN-2026-{ecn[0]}: alternate for {part}",
            approved,
            f"{alt} ({part_kind[part]}, second source, {supplier[alt]}) is approved as an alternate for {part} "
            f"in {where}. Stores may issue either part for {x}" + (f" and {y}" if both else "") + ". Fit and "
            "function were checked on a pilot build.",
            [f"{alt} ALTERNATE_FOR {part} in {a} since {approved}" for a in ([x, y] if both else [x])],
        )
    )
    if both:
        withdrawn = _days(approved, 15, 35, r)
        notes.append(
            _note(
                f"ecn-2026-{ecn[1]}",
                f"ECN-2026-{ecn[1]}: {y} {assembly_kind[y]}",
                withdrawn,
                f"After two failures on the {y} {assembly_kind[y]} of {my}, quality asks that {alt} is no longer "
                f"fitted in place of {part} in {y} until the root cause is closed. {x} is not affected.",
                [f"{alt} ALTERNATE_FOR {part} in {y} until {withdrawn}", f"{y} INSTALLED_ON {my}"],
            )
        )
    if moved_on:
        notes.append(
            _move(
                mx, layout[mx], final[mx], moved_on, machine_kind[mx], f"to support the {makes[final[mx]][0]} ramp-up"
            )
        )
    if mp and lp and planned_on:
        minutes = TODAY - timedelta(days=r.randint(2, 6))
        notes.append(
            _note(
                f"minutes-{minutes}",
                "Planning meeting, minutes",
                minutes,
                "Present: planning, production, maintenance.\n\n"
                f"1. {mp} ({machine_kind[mp]}) will run on line {lp} from {long_date(planned_on)}, once the "
                f"{final[mp]} volumes drop. Until then it stays on {final[mp]}.\n"
                f"2. Saturday overtime for {makes[final[mp]][1]} stays as planned.",
                [f"{mp} RUNS_ON {lp} since {planned_on}", f"{mp} RUNS_ON {final[mp]} until {planned_on}"],
            )
        )
    if mw and w:
        swapped = _days(AUDIT, 30, 100, r)
        v = r.choice([a for a in others if a not in register[mw]])
        notes.append(
            _note(
                f"logbook-{mw.lower()}",
                f"Maintenance logbook, {mw}",
                swapped,
                f"Replaced the worn {w} {assembly_kind[w]} with a rebuilt {v} {assembly_kind[v]}. {w} went to "
                "stores for overhaul; parts for it are on order.",
                [f"{w} INSTALLED_ON {mw} until {swapped}", f"{v} INSTALLED_ON {mw} since {swapped}"],
            )
        )
    if mn:
        q = register[mn][0]
        checked = TODAY - timedelta(days=r.randint(5, 30))
        notes.append(
            _note(
                f"logbook-{mn.lower()}",
                f"Maintenance logbook, {mn}",
                checked,
                f"Alarm on the night shift: low clamp pressure. Checked it: unlike {mx}, this "
                f"{machine_kind[mn]} does not use the {x} {assembly_kind[x]}; it runs on the {q} "
                f"{assembly_kind[q]}, which was within limits. No fault found.",
                [f"{q} INSTALLED_ON {mn}"],
            )
        )
    # Notes off the path: a move that changes nothing for this part, an unrelated approval, and
    # two notes that state nothing the graph holds (one names the late part, as noise for search).
    md = r.choice([m for m in machines if m not in (mx, my, mw, mn)])
    notes.append(
        _move(
            md,
            layout[md],
            r.choice([line for line in LINES if line != layout[md]]),
            _days(AUDIT, 30, 100, r),
            machine_kind[md],
            f"while {layout[md]} is rebuilt",
        )
    )
    other_ecn = _days(AUDIT, 20, 100, r)
    notes.append(
        _note(
            f"ecn-2026-{ecn[2]}",
            f"ECN-2026-{ecn[2]}: alternate for {p2}",
            other_ecn,
            f"{alt2} ({part_kind[p2]}, second source) is approved as an alternate for {p2} in the {a2} "
            f"{assembly_kind[a2]}.",
            [f"{alt2} ALTERNATE_FOR {p2} in {a2} since {other_ecn}"],
        )
    )
    ml = r.choice(machines)
    notes.append(
        _note(
            f"lube-{ml.lower()}",
            "Lubrication plan change",
            TODAY - timedelta(days=r.randint(3, 20)),
            f"The lubrication interval for {ml} ({machine_kind[ml]}) changes from 250 to 400 operating hours "
            "after the last oil analysis came back clean.",
        )
    )
    neighbour = r.choice([p for p in bom_parts if p != part])
    notes.append(
        _note(
            f"prices-{supplier[part].split()[0].lower()}",
            f"{supplier[part]}: prices from January",
            TODAY - timedelta(days=r.randint(5, 40)),
            f"From: sales, {supplier[part]}\n\nPrices for {part} and {neighbour} rise by 4% from 1 January 2027. "
            "Lead times are unchanged except where we have written to you separately.",
        )
    )
    notes.sort(key=lambda d: (d.date, d.id))
    question = (
        f"{supplier[part]} writes that {part} ({part_kind[part]}) will not ship before {long_date(arrival)}. "
        "Which lines does that affect today? Of the open orders due before then, which are at risk, and which "
        "can an approved alternate part still save?"
    )
    return World(tuple(tables + notes), question, part, arrival.isoformat(), answer, traps)


def _table(doc_id: str, title: str, day: str, header: str, rows: list[str]) -> Document:
    return Document(doc_id, title, "table", day, "\n".join([title, header, *rows]))


def _note(doc_id: str, title: str, day: date, body: str, facts: list[str] | None = None) -> Document:
    text = f"{title}\nDate: {day.isoformat()}\n\n{body}"
    return Document(doc_id, title, "note", day.isoformat(), text, tuple(facts or ()))


def _move(machine: str, old: str, new: str, day: date, kind: str, why: str) -> Document:
    return _note(
        f"changeover-{machine.lower()}",
        f"Changeover: {machine} to line {new}",
        day,
        f"From {long_date(day)}, {machine} ({kind}) runs on line {new} {why}. It no longer feeds {old}.",
        [f"{machine} RUNS_ON {new} since {day}", f"{machine} RUNS_ON {old} until {day}"],
    )
