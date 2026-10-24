"""What a verifier says must never hand over the answer, and a generator must never contradict itself."""

from __future__ import annotations

import re

import pytest

from aisyslab.core.lab import EnvOptions
from aisyslab.labs.line_incident.env import LineIncidentEnv
from aisyslab.labs.maintenance_portal.env import MaintenancePortalEnv
from aisyslab.labs.parts_graph.env import PartsGraphEnv
from aisyslab.labs.production_schedule.plan import build
from aisyslab.labs.production_schedule.plantlib import list_schedule
from aisyslab.labs.quality_claims.claims import build as build_claims


@pytest.mark.parametrize("seed", range(6))
def test_line_incident_feedback_does_not_name_the_cause_first(seed: int) -> None:
    env = LineIncidentEnv(seed, EnvOptions())
    feedback = env.submit("2026-03-14 00:00 [INFO] nothing: a line that covers no system").feedback
    named = feedback.split("no evidence for ", 1)[1].split(", ")
    assert named == sorted(env.required)  # alphabetical, not the causal chain's order


def test_parts_graph_verdict_does_not_list_the_traps() -> None:
    env = PartsGraphEnv(0, EnvOptions())
    verdict = env.submit([], [], [])
    assert "traps" not in verdict.details and "planned" not in verdict.feedback


def test_maintenance_portal_feedback_describes_the_orders_without_naming_them() -> None:
    env = MaintenancePortalEnv(5, EnvOptions())
    verdict = env.finish()
    assert not verdict.accepted and not re.search(r"WO-\d+", verdict.feedback)


@pytest.mark.parametrize("seed", [0, 2900, 4369])
def test_quality_claims_never_put_two_claims_on_one_batch(seed: int) -> None:
    claims, batches = build_claims(seed)
    assert len(batches) == len(claims) == 5


def test_the_scheduling_library_refuses_a_successor_placed_before_its_predecessor() -> None:
    data, _notes, _rules = build(4)
    ids = [o["id"] for o in data["orders"]]
    with pytest.raises(ValueError, match="put O-02 earlier"):
        list_schedule(data, ids, after={"O-01": "O-02"})
    tight = data | {"horizon": 60}  # nothing fits, so the predecessor is left out, and so is its successor
    assert list_schedule(tight, ["O-02", "O-01"], after={"O-01": "O-02"}) == []
