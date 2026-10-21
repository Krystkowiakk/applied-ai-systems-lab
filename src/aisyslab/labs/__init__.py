"""Registry of labs, in the order the README presents them."""

from __future__ import annotations

from ..core.lab import Lab
from .inbox import LAB as INBOX
from .line_incident import LAB as LINE_INCIDENT
from .maintenance_portal import LAB as MAINTENANCE_PORTAL
from .plant_database import LAB as PLANT_DATABASE
from .relocation_rag import LAB as RELOCATION_RAG
from .service_repair import LAB as SERVICE_REPAIR
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
        RELOCATION_RAG,
    )
}


def get_lab(lab_id: str) -> Lab:
    try:
        return LABS[lab_id]
    except KeyError:
        raise KeyError(f"Unknown lab '{lab_id}'. Available: {', '.join(LABS)}") from None
