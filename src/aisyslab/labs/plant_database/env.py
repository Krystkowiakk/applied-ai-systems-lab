"""The plant's maintenance database: answer a planner's question with read-only SQL.

Which open corrective work orders on line 3 cannot start because a spare part is short?
Everything needed is in the database, but not where a quick look suggests:

- the line is a column on the asset record, and two assets moved lines without being re-tagged;
- stock is split between the main store and line-side stores, and some of it is reserved;
- superseded part numbers are replaced by their successor, whose stock counts;
- the "waiting parts" status is set by hand and is stale in both directions.

The connection is read-only at the engine level: an authorizer allows reading and nothing
else, ``query_only`` is on, and a progress handler stops runaway queries. Write attempts are
refused and counted, and the verifier fails a run that made one: the task only asks to read.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, NoReturn

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict

LINE = "L3"
ROW_LIMIT = 50
STEP_BUDGET = 200  # progress-handler ticks of 1,000 virtual-machine instructions per query

SCHEMA = """
CREATE TABLE assets (
    asset_id INTEGER PRIMARY KEY,
    tag TEXT NOT NULL,
    description TEXT NOT NULL,
    line TEXT NOT NULL
);
CREATE TABLE parts (
    part_no TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    unit TEXT NOT NULL
);
CREATE TABLE part_supersessions (
    old_part_no TEXT NOT NULL REFERENCES parts(part_no),
    new_part_no TEXT NOT NULL REFERENCES parts(part_no),
    effective_on TEXT NOT NULL
);
CREATE TABLE storerooms (
    storeroom TEXT PRIMARY KEY,
    kind TEXT NOT NULL
);
CREATE TABLE stock (
    part_no TEXT NOT NULL REFERENCES parts(part_no),
    storeroom TEXT NOT NULL REFERENCES storerooms(storeroom),
    on_hand INTEGER NOT NULL
);
CREATE TABLE work_orders (
    wo_id TEXT PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(asset_id),
    wo_type TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    opened_on TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE TABLE wo_parts (
    wo_id TEXT NOT NULL REFERENCES work_orders(wo_id),
    part_no TEXT NOT NULL REFERENCES parts(part_no),
    qty_required INTEGER NOT NULL
);
CREATE TABLE reservations (
    reservation_id INTEGER PRIMARY KEY,
    part_no TEXT NOT NULL REFERENCES parts(part_no),
    wo_id TEXT NOT NULL REFERENCES work_orders(wo_id),
    qty INTEGER NOT NULL
);
"""
TABLES = ("assets", "parts", "part_supersessions", "storerooms", "stock", "work_orders", "wo_parts", "reservations")

ASSET_KINDS = [
    ("PMP", "Coolant pump"),
    ("CNV", "Belt conveyor"),
    ("PRS", "Hydraulic press"),
    ("FAN", "Extraction fan"),
    ("MTR", "Drive motor"),
    ("VLV", "Steam valve"),
    ("CMP", "Air compressor"),
]
PART_NAMES = [
    "Bearing 6205-2RS",
    "Mechanical seal 35 mm",
    "V-belt A42",
    "Proximity sensor M12",
    "Contactor 24 V DC",
    "Hydraulic hose 1/2 in",
    "Coupling insert",
    "Solenoid valve 24 V",
    "Gear motor 0.75 kW",
    "Pressure switch",
    "Filter cartridge",
    "Timing belt",
    "Thermocouple type K",
    "Limit switch",
    "Gasket set",
    "O-ring kit",
    "Roller chain 08B",
    "Encoder 1024 ppr",
    "Brake pad set",
    "Cooling fan 120 mm",
    "Relay 24 V",
    "Check valve 1 in",
    "Impeller 160 mm",
    "Bushing 30 mm",
    "Cylinder seal kit",
    "Level sensor",
    "Flow meter",
    "Belt tensioner",
    "Servo cable 5 m",
    "Fuse 10 A",
]
PROBLEMS = [
    "noise and vibration",
    "leak at the seal",
    "trips on overload",
    "sensor drops out",
    "belt slipping",
    "low pressure",
    "running hot",
]
STATUSES_CLOSED = ("closed", "cancelled")

# The authorizer allows reading and nothing else.
_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_READ_PRAGMAS = {"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo", "foreign_key_list"}
_NOT_WRITES = {sqlite3.SQLITE_TRANSACTION, getattr(sqlite3, "SQLITE_SAVEPOINT", 32)}
_WRITE_WORDS = re.compile(r"\b(insert|update|delete|replace|drop|alter|create|attach|detach|vacuum|reindex)\b", re.I)

# Why an order the solver listed should not be there, and why a blocked order is missing.
EXTRA_REASONS = {
    "split_store": "stock counted in only some storerooms",
    "superseded": "a superseded part whose successor is in stock",
    "stale_status": "status 'waiting parts' is stale, the parts are in stock now",
    "relocated_out": "asset tag says line 3, but the asset is on another line now",
    "available": "all parts available",
    "closed": "closed or cancelled",
    "not_corrective": "not a corrective work order",
    "other_line": "another line",
}
MISSING_REASONS = {
    "short": "short of one of the parts it needs",
    "short_status_open": "short of a part although the status does not say so",
    "short_reserved": "short once stock reserved for other work orders is taken out",
    "short_qty": "some stock, but less than the quantity needed",
    "relocated_in": "asset moved to line 3, its tag still says otherwise",
}
BLOCKED_KINDS = set(MISSING_REASONS)


@dataclass
class Asset:
    asset_id: int
    tag: str
    description: str
    line: str


class PlantDatabaseEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.write_attempts = 0
        self._denied: list[tuple[int, str | None, str | None]] = []
        self._ticks = 0
        self.categories: dict[str, str] = {}
        self._generate()
        self.expected = self._blocked_orders()
        designed = {wo for wo, kind in self.categories.items() if kind in BLOCKED_KINDS}
        assert self.expected == designed, (sorted(self.expected), sorted(designed))
        self.db = self._build_database()
        self.instruction = (
            "Planner's question for tonight's shift: which open corrective work orders on line 3 cannot "
            "start because a required spare part is short? Open means any status except closed and "
            "cancelled. A part is short when the stock on hand across all storerooms, minus the quantity "
            "reserved for other work orders, is less than the quantity the work order needs. A superseded "
            "part number is replaced by its successor, whose stock counts. Answer from the plant database "
            "with the work order IDs."
        )

    # ----- data ----------------------------------------------------------------------------
    def _generate(self) -> None:
        r = self.rng
        asset_ids = iter(range(101, 200))
        self.assets: list[Asset] = []
        for line in ("L1", "L2", "L3", "L4"):
            for n, (code, desc) in enumerate(r.sample(ASSET_KINDS, 5), start=1):
                self.assets.append(Asset(next(asset_ids), f"{line}-{code}-{n:02d}", desc, line))
        moved_in = r.choice([a for a in self.assets if a.line == "L2"])
        moved_out = r.choice([a for a in self.assets if a.line == LINE])
        moved_in.line, moved_out.line = LINE, "L4"  # relocated; the tags were never changed
        line3 = [a for a in self.assets if a.line == LINE and a is not moved_in]
        elsewhere = [a for a in self.assets if a.line != LINE and a is not moved_out]

        numbers = r.sample(range(10000, 99999), len(PART_NAMES))
        self.parts = [(f"P-{n}", name, "pcs") for n, name in zip(numbers, PART_NAMES, strict=True)]
        free = [p for p, _n, _u in self.parts]
        r.shuffle(free)
        self.storerooms = [("MAIN", "central")] + [(f"{line}-SIDE", "line-side") for line in ("L1", "L2", "L3", "L4")]
        self.stock: list[tuple[str, str, int]] = []
        self.supersessions: list[tuple[str, str, str]] = []
        self.reservations: list[tuple[int, str, str, int]] = []
        self.work_orders: list[dict[str, Any]] = []
        self.wo_parts: list[tuple[str, str, int]] = []
        wo_numbers = iter(r.sample(range(2001, 2999), 40))

        def part(*stores: tuple[str, int]) -> str:
            p = free.pop()
            for store, qty in stores:
                self.stock.append((p, store, qty))
            return p

        def order(
            asset: Asset, kind: str, status: str, needs: list[tuple[str, int]], wo_type: str = "corrective"
        ) -> str:
            wo = f"WO-{next(wo_numbers)}"
            self.work_orders.append(
                {
                    "wo_id": wo,
                    "asset_id": asset.asset_id,
                    "wo_type": wo_type,
                    "status": status,
                    "priority": r.randint(1, 4),
                    "opened_on": f"2026-10-{r.randint(1, 20):02d}",
                    "summary": f"{asset.description}: {r.choice(PROBLEMS)}",
                }
            )
            self.wo_parts.extend((wo, p, q) for p, q in needs)
            self.categories[wo] = kind
            return wo

        pick = lambda pool: r.choice(pool)  # noqa: E731
        plenty = lambda: ("MAIN", r.randint(3, 8))  # noqa: E731

        # Blocked: what the planner needs to see.
        order(pick(line3), "short", "waiting parts", [(part(plenty()), 1), (part(("MAIN", 0)), 1)])
        order(pick(line3), "short_status_open", "open", [(part(), 1)])
        reserved = part(("MAIN", 2))
        order(pick(line3), "short_reserved", "in progress", [(reserved, 1)])
        order(pick(line3), "short_qty", "open", [(part((f"{LINE}-SIDE", 1)), 3)])
        order(moved_in, "relocated_in", "waiting parts", [(part(), 1)])

        # Not blocked, though a quick query says otherwise.
        order(pick(line3), "split_store", "open", [(part(("MAIN", 0), ("L1-SIDE", 1), (f"{LINE}-SIDE", 1)), 2)])
        old, new = part(("MAIN", 0)), part(plenty())
        self.supersessions.append((old, new, "2026-09-01"))
        order(pick(line3), "superseded", "in progress", [(old, 1)])
        order(pick(line3), "stale_status", "waiting parts", [(part(plenty()), 1)])
        order(moved_out, "relocated_out", "waiting parts", [(part(), 1)])
        for _ in range(r.randint(1, 3)):
            order(pick(line3), "available", r.choice(["open", "in progress"]), [(part(plenty()), 1)])

        # Out of scope: closed, other types, other lines.
        order(pick(line3), "closed", r.choice(STATUSES_CLOSED), [(part(), 1)])
        order(pick(line3), "not_corrective", "open", [(part(), 1)], wo_type="preventive")
        for _ in range(r.randint(2, 4)):
            order(pick(elsewhere), "other_line", r.choice(["open", "waiting parts"]), [(part(), 1)])
        holder = order(pick(elsewhere), "other_line", "open", [(part(plenty()), 1)], wo_type="preventive")
        self.reservations.append((1, reserved, holder, 2))

        # The rest of the store: parts nobody on line 3 needs, so the tables look like a real store.
        others = [w["wo_id"] for w in self.work_orders if self.categories[w["wo_id"]] == "other_line"]
        while len(free) > 2:
            p = free.pop()
            main = r.randint(0, 12)
            self.stock.append((p, "MAIN", main))
            if r.random() < 0.4:
                self.stock.append((p, f"{r.choice(['L1', 'L2', 'L3', 'L4'])}-SIDE", r.randint(1, 4)))
            if main > 2 and len(self.reservations) < 4:
                self.reservations.append((len(self.reservations) + 1, p, r.choice(others), r.randint(1, 2)))
        self.supersessions.append((free.pop(), free.pop(), "2026-06-15"))
        self.work_orders.sort(key=lambda w: w["wo_id"])

    def _blocked_orders(self) -> set[str]:
        """The answer, computed from the records in Python (the baseline does it in SQL)."""
        on_hand: defaultdict[str, int] = defaultdict(int)
        held: defaultdict[str, int] = defaultdict(int)
        for p, _store, qty in self.stock:
            on_hand[p] += qty
        for _rid, p, _wo, qty in self.reservations:
            held[p] += qty
        successor = {old: new for old, new, _d in self.supersessions}
        line_of = {a.asset_id: a.line for a in self.assets}
        needs: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for wo, p, qty in self.wo_parts:
            needs[wo].append((successor.get(p, p), qty))
        return {
            w["wo_id"]
            for w in self.work_orders
            if line_of[w["asset_id"]] == LINE
            and w["wo_type"] == "corrective"
            and w["status"] not in STATUSES_CLOSED
            and any(on_hand[p] - held[p] < qty for p, qty in needs[w["wo_id"]])
        }

    def _build_database(self) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO assets VALUES (?, ?, ?, ?)", [(a.asset_id, a.tag, a.description, a.line) for a in self.assets]
        )
        db.executemany("INSERT INTO parts VALUES (?, ?, ?)", self.parts)
        db.executemany("INSERT INTO part_supersessions VALUES (?, ?, ?)", self.supersessions)
        db.executemany("INSERT INTO storerooms VALUES (?, ?)", self.storerooms)
        db.executemany("INSERT INTO stock VALUES (?, ?, ?)", self.stock)
        db.executemany(
            "INSERT INTO work_orders VALUES (:wo_id, :asset_id, :wo_type, :status, :priority, :opened_on, :summary)",
            self.work_orders,
        )
        db.executemany("INSERT INTO wo_parts VALUES (?, ?, ?)", self.wo_parts)
        db.executemany("INSERT INTO reservations VALUES (?, ?, ?, ?)", self.reservations)
        db.commit()
        db.execute("PRAGMA query_only = ON")
        db.set_authorizer(self._authorize)
        db.set_progress_handler(self._tick, 1000)
        return db

    # ----- the guard ------------------------------------------------------------------------
    def _authorize(self, action: int, arg1: str | None, arg2: str | None, _db: str | None, _src: str | None) -> int:
        if action in _ALLOWED:
            return sqlite3.SQLITE_OK
        self._denied.append((action, arg1, arg2))
        return sqlite3.SQLITE_DENY

    def _tick(self) -> int:
        self._ticks += 1
        return 1 if self._ticks > STEP_BUDGET else 0

    def _is_write(self, action: int, arg1: str | None, arg2: str | None) -> bool:
        if action == sqlite3.SQLITE_PRAGMA:
            return arg2 is not None and (arg1 or "").lower() not in _READ_PRAGMAS
        return action not in _NOT_WRITES

    def _run(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        self._denied, self._ticks = [], 0
        try:
            return self.db.execute(sql, params)
        except sqlite3.ProgrammingError as exc:
            if "one statement" in str(exc):
                if _WRITE_WORDS.search(sql):
                    self.write_attempts += 1
                raise ToolError("One statement per call.", hint="Send a single SELECT.") from exc
            raise ToolError(f"SQL error: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            self._refuse(exc)

    def _refuse(self, exc: sqlite3.DatabaseError) -> NoReturn:
        if "not authorized" in str(exc) and self._denied:
            if any(self._is_write(*d) for d in self._denied):
                self.write_attempts += 1
                raise ToolError(
                    "Read-only database: this statement would change data and was refused.",
                    hint="Only SELECT is allowed. The question needs no changes to the data.",
                ) from exc
            if any(a == sqlite3.SQLITE_PRAGMA for a, _x, _y in self._denied):
                raise ToolError("PRAGMA is not available.", hint="Use describe_table for columns and keys.") from exc
            raise ToolError("Only single SELECT statements are allowed.") from exc
        if "interrupted" in str(exc):
            raise ToolError(
                "Query stopped: it exceeded the cost limit.",
                hint="Join on keys (asset_id, wo_id, part_no) and filter early; avoid cross joins.",
            ) from exc
        raise ToolError(f"SQL error: {exc}", hint="Check table and column names with describe_table.") from exc

    # ----- the API --------------------------------------------------------------------------
    def tables(self) -> list[dict[str, Any]]:
        self._api("tables")
        return [{"table": t, "rows": self._run(f"SELECT COUNT(*) FROM {t}").fetchone()[0]} for t in TABLES]

    def describe(self, table: str) -> dict[str, Any]:
        self._api("describe")
        if table not in TABLES:
            raise ToolError(f"No table '{table}'.", hint=f"Tables: {', '.join(TABLES)}.")
        ddl = self._run("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()[0]
        cursor = self._run(f"SELECT * FROM {table} LIMIT 3")
        return {
            "table": table,
            "ddl": " ".join(ddl.split()),
            "columns": [c[0] for c in cursor.description],
            "sample_rows": [list(row) for row in cursor.fetchall()],
        }

    def query(self, sql: str, limit: int = ROW_LIMIT) -> dict[str, Any]:
        self._api("query")
        limit = max(1, min(limit, ROW_LIMIT))
        cursor = self._run(sql)
        try:
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            self._refuse(exc)
        result: dict[str, Any] = {
            "columns": [c[0] for c in cursor.description or ()],
            "rows": [list(row) for row in rows[:limit]],
            "row_count": len(rows),
            "truncated": len(rows) > limit,
        }
        if result["truncated"]:
            result["hint"] = f"Showing {limit} of {len(rows)} rows. Aggregate or filter instead of paging."
        return result

    def submit(self, work_order_ids: list[str]) -> Verdict:
        self._api("submit")
        given = {w.strip().upper() for w in work_order_ids if w.strip()}
        unknown = sorted(w for w in given if w not in self.categories)
        extra = [w for w in given - self.expected if w in self.categories]
        missing = sorted(self.expected - given)
        problems: list[str] = []
        if extra:
            problems.append(
                f"{_n(len(extra), 'listed order')} should not be there ({_reasons(extra, self.categories, EXTRA_REASONS)})"
            )
        if missing:
            problems.append(
                f"{_n(len(missing), 'blocked order')} missing ({_reasons(missing, self.categories, MISSING_REASONS)})"
            )
        if unknown:
            problems.append(f"{_n(len(unknown), 'ID')} not in work_orders")
        if self.write_attempts:
            problems.append(
                f"{_n(self.write_attempts, 'write attempt')} against the database; the question only needs reading"
            )
        hits = len(given & self.expected)
        score = 2 * hits / (len(given) + len(self.expected)) if given or self.expected else 1.0
        accepted = not problems
        return self._record(
            Verdict(
                accepted,
                "OK" if accepted else "REJECTED",
                "All blocked work orders found." if accepted else "; ".join(problems) + ".",
                {
                    "score": round(score, 3),
                    "missing": len(missing),
                    "extra": len(extra),
                    "unknown": len(unknown),
                    "write_attempts": self.write_attempts,
                },
            )
        )


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _reasons(ids: list[str], categories: dict[str, str], texts: dict[str, str]) -> str:
    counts = Counter(categories[i] for i in ids)
    return "; ".join(f"{n}x {texts[kind]}" for kind, n in sorted(counts.items()))
