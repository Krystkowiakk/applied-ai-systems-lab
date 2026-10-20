"""Line 3 is down until a spare pump reaches its maintenance door. Plan the run across the site.

A forklift is fast but has a small battery and cannot use walkways or stairs; walking is
slow; dismounting is one-way. The map, the vehicle rules and the budgets live in documents
the solver must discover first. The verifier simulates the plan and names the first step
that breaks a rule, which makes it a hard, deterministic judge.
"""

from __future__ import annotations

import heapq
import json
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict

ROWS, COLS = 10, 12
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
TERRAIN = {
    "=": "road",
    ".": "yard",
    ":": "walkway",
    "^": "stairs",
    "#": "building wall",
    "~": "cooling pond",
    "X": "furnace exclusion zone",
    "S": "stores (start)",
    "G": "line 3 maintenance door (goal)",
}
RULES: dict[str, Any] = {
    "start_mode": "forklift",
    "commands": ["up", "down", "left", "right", "dismount"],
    "forklift": {"minutes_per_step": {"=": 1, ".": 1, "S": 1}, "battery_per_step": {"=": 1, ".": 2, "S": 1}},
    "walk": {"minutes_per_step": {"=": 3, ".": 3, ":": 3, "^": 5, "S": 3, "G": 3}, "battery_per_step": 0},
    "dismount": {"minutes": 2, "note": "Leave the forklift and carry the part. You cannot drive again."},
    "impassable": ["#", "~", "X"],
    "notes": "A mode can enter a cell only if the cell's symbol has a cost for that mode.",
}
MAX_COMMANDS = 60

DOCS = {
    "site_map": ("Map of the plant site with a legend.", ["map", "site", "grid", "terrain", "layout", "route"]),
    "vehicle_rules": (
        "Movement costs per mode, dismounting, budgets for this run.",
        ["vehicle", "forklift", "walk", "rules", "cost", "battery", "budget", "time"],
    ),
    "safety_rules": ("Site safety rules for moving loads.", ["safety", "rules", "furnace", "zone"]),
    "canteen_hours": ("Opening hours of the canteen.", ["canteen", "food", "hours"]),
}


class SparePartRunEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        for _ in range(200):
            grid, start, goal = self._generate()
            plan = solve_grid(grid, start, goal, battery_limit=10**6, minutes_limit=10**6)
            walk_only = solve_grid(grid, start, goal, battery_limit=0, minutes_limit=10**6)
            if plan is None or walk_only is None:
                continue
            minutes, battery, _ = plan
            budget_minutes, budget_battery = minutes + 2, battery + 1
            if walk_only[0] <= budget_minutes:  # walking all the way must not be enough
                continue
            self.grid, self.start, self.goal = grid, start, goal
            self.budget_minutes, self.budget_battery = budget_minutes, budget_battery
            self.optimal_minutes = minutes
            break
        else:  # pragma: no cover - generation always converges in practice
            raise RuntimeError(f"could not generate a site for seed {seed}")

    def _generate(self) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
        r = self.rng
        grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
        top, left = r.randint(1, 3), r.randint(7, 8)
        bottom, right = top + r.randint(4, 5), COLS - 1
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                edge = row in (top, bottom) or col in (left, right)
                grid[row][col] = "#" if edge else ":"
        door_row = r.randint(top + 1, bottom - 1)
        grid[door_row][left] = "^"
        goal = (r.randint(top + 1, bottom - 1), r.randint(left + 2, right - 1))
        grid[goal[0]][goal[1]] = "G"
        road_row = r.choice([row for row in range(ROWS) if row < top - 1 or row > bottom + 1] or [ROWS - 1])
        road_col = r.randint(1, left - 3)
        for col in range(0, left):
            grid[road_row][col] = "="
        for row in range(ROWS):
            if grid[row][road_col] == ".":
                grid[row][road_col] = "="
        pond_row, pond_col = r.randint(0, ROWS - 3), r.randint(road_col + 1, max(road_col + 1, left - 3))
        for row in range(pond_row, pond_row + 2):
            for col in range(pond_col, min(pond_col + 2, left - 1)):
                if grid[row][col] == ".":
                    grid[row][col] = "~"
        hot_row = r.choice([row for row in (door_row - 2, door_row + 2) if 0 <= row < ROWS])
        for col in (left - 2, left - 1):
            if grid[hot_row][col] == ".":
                grid[hot_row][col] = "X"
        start_row = r.choice([0, ROWS - 1])
        start = (start_row, road_col)
        grid[start_row][road_col] = "S"
        return grid, start, goal

    # ----- discovery API ------------------------------------------------------------------
    def search_docs(self, query: str) -> list[dict[str, str]]:
        self._api("search_docs")
        words = {w.lower() for w in query.split()}
        hits = [
            {"name": name, "description": desc}
            for name, (desc, keys) in DOCS.items()
            if words & set(keys) or query.lower() in name
        ]
        return hits or [{"hint": "No document matched. Try words like map, vehicle, rules, safety."}]

    def read_doc(self, name: str) -> str:
        self._api("read_doc")
        if name == "site_map":
            header = "     " + " ".join(f"{c:>2}" for c in range(COLS))
            rows = [f"{r:>3}  " + " ".join(f"{cell:>2}" for cell in row) for r, row in enumerate(self.grid)]
            legend = "\n".join(f"  {k} = {v}" for k, v in TERRAIN.items())
            return (
                f"Rows go top to bottom, columns left to right, both from 0.\n{header}\n"
                + "\n".join(rows)
                + f"\nLegend:\n{legend}"
            )
        if name == "vehicle_rules":
            return json.dumps(
                {**RULES, "budget": {"minutes": self.budget_minutes, "battery": self.budget_battery}}, indent=2
            )
        if name == "safety_rules":
            return (
                "Keep out of the furnace exclusion zone and the cooling pond. Forklifts stay off walkways and stairs."
            )
        if name == "canteen_hours":
            return "Canteen: 06:00-20:00. Night shift: vending machines in Hall B."
        raise ToolError(f"No document '{name}'.", hint="Use search_docs to find document names.")

    # ----- judge --------------------------------------------------------------------------
    def simulate(self, commands: list[str]) -> dict[str, Any]:
        self._api("simulate")
        return simulate(self.grid, self.start, self.goal, commands, self.budget_minutes, self.budget_battery)

    def submit(self, commands: list[str]) -> Verdict:
        self._api("submit")
        result = simulate(self.grid, self.start, self.goal, commands, self.budget_minutes, self.budget_battery)
        if result["arrived"]:
            return self._record(
                Verdict(
                    True,
                    "OK",
                    f"The pump reached line 3 in {result['minutes']} minutes using {result['battery']} battery.",
                    result,
                )
            )
        return self._record(Verdict(False, "REJECTED", result["problem"], result))


def simulate(
    grid: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    commands: list[str],
    minutes_budget: int,
    battery_budget: int,
) -> dict[str, Any]:
    if len(commands) > MAX_COMMANDS:
        return {
            "arrived": False,
            "problem": f"plan has {len(commands)} commands (max {MAX_COMMANDS})",
            "minutes": 0,
            "battery": 0,
        }
    row, col = start
    mode, minutes, battery = "forklift", 0, 0
    trail: list[list[Any]] = []
    for i, raw in enumerate(commands, 1):
        command = raw.strip().lower()
        if command == "dismount":
            if mode == "walk":
                return _fail(f"step {i}: already walking", minutes, battery, (row, col), trail)
            mode, minutes = "walk", minutes + RULES["dismount"]["minutes"]
        elif command in MOVES:
            dr, dc = MOVES[command]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                return _fail(
                    f"step {i} ({command}): leaves the site at ({nr},{nc})", minutes, battery, (row, col), trail
                )
            cell = grid[nr][nc]
            costs = RULES[mode]["minutes_per_step"]
            if cell not in costs:
                return _fail(
                    f"step {i} ({command}): {mode} cannot enter {TERRAIN[cell]} at ({nr},{nc})",
                    minutes,
                    battery,
                    (row, col),
                    trail,
                )
            minutes += costs[cell]
            battery += RULES["forklift"]["battery_per_step"][cell] if mode == "forklift" else 0
            row, col = nr, nc
        else:
            return _fail(f"step {i}: unknown command '{raw}'", minutes, battery, (row, col), trail)
        trail.append([row, col, mode])
        if battery > battery_budget:
            return _fail(
                f"step {i}: forklift battery exhausted ({battery} > {battery_budget})",
                minutes,
                battery,
                (row, col),
                trail,
            )
        if minutes > minutes_budget:
            return _fail(
                f"step {i}: out of time ({minutes} > {minutes_budget} minutes)", minutes, battery, (row, col), trail
            )
    if (row, col) != goal:
        return _fail(f"plan ends at ({row},{col}); the goal is at {goal}", minutes, battery, (row, col), trail)
    return {"arrived": True, "minutes": minutes, "battery": battery, "position": [row, col]}


def _fail(
    problem: str, minutes: int, battery: int, position: tuple[int, int], trail: list[list[Any]]
) -> dict[str, Any]:
    return {
        "arrived": False,
        "problem": problem,
        "minutes": minutes,
        "battery": battery,
        "position": list(position),
        "trail_tail": trail[-5:],
    }


def solve_grid(
    grid: list[list[str]], start: tuple[int, int], goal: tuple[int, int], *, battery_limit: int, minutes_limit: int
) -> tuple[int, int, list[str]] | None:
    """Exact search over (cell, mode, battery used), minimising minutes. Returns (minutes, battery, commands)."""
    frontier: list[tuple[int, int, int, int, str, list[str]]] = [(0, 0, start[0], start[1], "forklift", [])]
    seen: dict[tuple[int, int, str, int], int] = {}
    while frontier:
        minutes, battery, row, col, mode, path = heapq.heappop(frontier)
        if (row, col) == goal:
            return minutes, battery, path
        key = (row, col, mode, battery)
        if seen.get(key, 10**9) <= minutes:
            continue
        seen[key] = minutes
        if mode == "forklift":
            heapq.heappush(
                frontier, (minutes + RULES["dismount"]["minutes"], battery, row, col, "walk", [*path, "dismount"])
            )
        for command, (dr, dc) in MOVES.items():
            nr, nc = row + dr, col + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            cell = grid[nr][nc]
            costs = RULES[mode]["minutes_per_step"]
            if cell not in costs:
                continue
            nb = battery + (RULES["forklift"]["battery_per_step"][cell] if mode == "forklift" else 0)
            nm = minutes + costs[cell]
            if nb <= battery_limit and nm <= minutes_limit:
                heapq.heappush(frontier, (nm, nb, nr, nc, mode, [*path, command]))
    return None
