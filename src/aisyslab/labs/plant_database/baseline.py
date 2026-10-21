"""Deterministic solver: the question is known, so the answer is one SQL statement.

Code that knows the business rule has nothing to discover: it sums stock across storerooms,
subtracts reservations, follows supersessions and takes the line from the asset record. The
agent has to find all of that in a schema it has never seen.
"""

from __future__ import annotations

from ...core.lab import RunContext
from .env import PlantDatabaseEnv

BLOCKED_ORDERS = """
WITH available AS (
    SELECT p.part_no,
           COALESCE((SELECT SUM(s.on_hand) FROM stock s WHERE s.part_no = p.part_no), 0)
         - COALESCE((SELECT SUM(r.qty) FROM reservations r WHERE r.part_no = p.part_no), 0) AS qty
    FROM parts p
),
needed AS (
    SELECT wp.wo_id, COALESCE(ps.new_part_no, wp.part_no) AS part_no, wp.qty_required
    FROM wo_parts wp
    LEFT JOIN part_supersessions ps ON ps.old_part_no = wp.part_no
)
SELECT DISTINCT w.wo_id
FROM work_orders w
JOIN assets a ON a.asset_id = w.asset_id
JOIN needed n ON n.wo_id = w.wo_id
JOIN available v ON v.part_no = n.part_no
WHERE a.line = 'L3'
  AND w.wo_type = 'corrective'
  AND w.status NOT IN ('closed', 'cancelled')
  AND v.qty < n.qty_required
ORDER BY w.wo_id
"""


def solve(env: PlantDatabaseEnv, ctx: RunContext) -> None:
    rows = ctx.call(env.query, BLOCKED_ORDERS)["rows"]
    ctx.submit(env.submit, [wo for (wo,) in rows])
