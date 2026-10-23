"""A small web about one discontinued valve, generated per seed. Every site is fictional.

The manufacturer's notice names the successor, whose current datasheet supersedes an older one
with a lower pressure rating. A distributor lists the successor and an alternative from another
maker, next to an archived page with an old lead time. A forum answer recommends a valve with the
wrong coil voltage, which the same distributor can ship in a week. One page sits behind robots.txt.
"""

from __future__ import annotations

import random

from ...research.web import Page

QUESTION = (
    "Line 2's VT-200 solenoid valve (24 V DC coil, G1/4 ports, at least 9 bar) is discontinued. Which "
    "compatible replacement can we get fastest, in how many weeks, and what is its maximum pressure? Name "
    "the parts you rejected and why, and write a short report with sources."
)
ROBOTS = {"valvetec.example": ["/internal/"]}


def build(seed: int) -> tuple[list[Page], dict[str, object]]:
    r = random.Random(seed * 41 + 3)
    vt210_weeks, hx24_weeks = r.sample(range(3, 15), 2)
    v, f, h = "https://valvetec.example", "https://fluidparts.example", "https://hydrax.example"
    pages = [
        Page(
            f"{v}/products/vt-200",
            "VT-200 solenoid valve (discontinued)",
            "VT-200 solenoid valve: 24 V DC coil, G1/4 ports, max pressure 10 bar. Status: discontinued, see PCN-2026-03. Dealer prices on the internal page.",
            (f"{v}/notices/pcn-2026-03", f"{v}/internal/pricing"),
        ),
        Page(
            f"{v}/notices/pcn-2026-03",
            "PCN-2026-03: VT-200 discontinued",
            "Product change notice PCN-2026-03: the VT-200 is discontinued from June 2026. Recommended replacement: VT-210, with the same coil (24 V DC) and ports (G1/4). Panel mounting needs the bracket BR-210.",
            (f"{v}/datasheets/vt-210",),
        ),
        Page(
            f"{v}/datasheets/vt-210",
            "VT-210 datasheet, revision C",
            "VT-210 datasheet, revision C (current, May 2026): 24 V DC coil, G1/4 ports, max pressure 10 bar, -10 to 80 C. Supersedes revision A.",
            (f"{v}/datasheets/vt-210-rev-a",),
        ),
        Page(
            f"{v}/datasheets/vt-210-rev-a",
            "VT-210 datasheet, revision A",
            "VT-210 datasheet, revision A (February 2024, superseded by revision C): 24 V DC coil, G1/4 ports, max pressure 8 bar.",
        ),
        Page(
            f"{v}/datasheets/vt-150",
            "VT-150 datasheet",
            "VT-150 solenoid valve: 12 V DC coil, G1/4 ports, max pressure 10 bar.",
        ),
        Page(
            f"{v}/products/vt-100",
            "VT-100 solenoid valve",
            "VT-100 solenoid valve: 230 V AC coil, G3/8 ports, max pressure 6 bar.",
        ),
        Page(f"{v}/internal/pricing", "Dealer price list", "Internal dealer prices. Not for crawlers."),
        Page(
            f"{f}/stock/vt-210",
            "VT-210 at FluidParts",
            f"VT-210 solenoid valve at FluidParts: lead time {vt210_weeks} weeks. Updated 2026-10-15.",
        ),
        Page(
            f"{f}/archive/stock/vt-210-2025",
            "VT-210 at FluidParts (archived)",
            "VT-210 solenoid valve at FluidParts (archived page): lead time 2 weeks. Updated 2025-11-02.",
        ),
        Page(
            f"{f}/stock/hx-24",
            "HX-24 at FluidParts",
            f"HX-24 solenoid valve at FluidParts: lead time {hx24_weeks} weeks. Updated 2026-10-12.",
        ),
        Page(
            f"{f}/stock/vt-150",
            "VT-150 at FluidParts",
            "VT-150 solenoid valve at FluidParts: in stock, lead time 1 week. Updated 2026-10-14.",
        ),
        Page(
            f"{h}/products/hx-24",
            "HX-24 solenoid valve",
            "HX-24 solenoid valve by Hydrax: 24 V DC coil, G1/4 ports, max pressure 16 bar, -20 to 90 C. Cross-reference: replaces the Valvetec VT-200 one to one.",
        ),
        Page(
            f"{h}/products/hx-12",
            "HX-12 solenoid valve",
            "HX-12 solenoid valve by Hydrax: 12 V DC coil, G1/4 ports, max pressure 16 bar.",
        ),
        Page(
            "https://forum.example/threads/vt-200-replacement",
            "VT-200 replacement?",
            "Question: what is a good VT-200 replacement? Answer from hydraulic_hank: just use the VT-150, it is the same valve, we swapped ours last year.",
        ),
    ]
    best = "VT-210" if vt210_weeks < hx24_weeks else "HX-24"
    answer = {
        "replacement": best,
        "lead_time_weeks": min(vt210_weeks, hx24_weeks),
        "max_pressure_bar": 10 if best == "VT-210" else 16,
        "stale_lead_time": 2 if best == "VT-210" else None,
        "stale_pressure": 8 if best == "VT-210" else None,
        "must_reject": "VT-150",
    }
    return pages, answer
