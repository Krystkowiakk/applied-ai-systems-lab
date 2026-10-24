"""Registry of labs; `aisyslab list` shows them in this order."""

from __future__ import annotations

from ..core.lab import Lab
from .inbox import LAB as INBOX
from .line_incident import LAB as LINE_INCIDENT
from .maintenance_portal import LAB as MAINTENANCE_PORTAL
from .part_research import LAB as PART_RESEARCH
from .parts_graph import LAB as PARTS_GRAPH
from .plant_database import LAB as PLANT_DATABASE
from .procedures_rag import LAB as PROCEDURES_RAG
from .production_schedule import LAB as PRODUCTION_SCHEDULE
from .quality_claims import LAB as QUALITY_CLAIMS
from .quality_gate import LAB as QUALITY_GATE
from .service_repair import LAB as SERVICE_REPAIR
from .shift_briefing import LAB as SHIFT_BRIEFING
from .shift_handover import LAB as SHIFT_HANDOVER
from .spare_part_run import LAB as SPARE_PART_RUN
from .supply_intake import LAB as SUPPLY_INTAKE

LABS: dict[str, Lab] = {
    lab.id: lab
    for lab in (
        INBOX,
        LINE_INCIDENT,
        SERVICE_REPAIR,
        SUPPLY_INTAKE,
        SPARE_PART_RUN,
        MAINTENANCE_PORTAL,
        PLANT_DATABASE,
        PRODUCTION_SCHEDULE,
        PART_RESEARCH,
        PARTS_GRAPH,
        QUALITY_GATE,
        QUALITY_CLAIMS,
        SHIFT_BRIEFING,
        SHIFT_HANDOVER,
        PROCEDURES_RAG,
    )
}


def get_lab(lab_id: str) -> Lab:
    try:
        return LABS[lab_id]
    except KeyError:
        raise KeyError(f"Unknown lab '{lab_id}'. Available: {', '.join(LABS)}") from None
