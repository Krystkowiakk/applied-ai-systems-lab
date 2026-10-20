"""A relocation handbook for a fictional country, generated per seed.

Fictional on purpose: answers can be verified exactly, and nothing here can be mistaken for
legal advice about a real jurisdiction. The traps mirror real relocation questions: a
superseded edition with older numbers, an exception that overrides the main rule, a
deadline that needs date arithmetic, a fact split across documents, and a question the
handbook does not answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Section:
    id: str
    doc: str
    title: str
    text: str
    edition: str
    status: str  # "current" or "superseded"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    value: str  # normalised expected value: digits, ISO date, yes/no, or "not covered"
    cites: tuple[str, ...]  # any one of these section ids supports the answer
    stale: str | None = None  # the value a superseded source would give


def build(seed: int) -> tuple[list[Section], list[Question]]:
    r = random.Random(seed * 31 + 7)
    t30 = r.randrange(4900, 5600, 25)
    t29 = r.randrange(3600, 4100, 25)
    tg = r.randrange(2700, 3000, 25)
    old = {k: v - r.randrange(100, 250, 25) for k, v in (("t30", t30), ("t29", t29), ("tg", tg))}
    weeks, old_weeks = r.choice([3, 4, 6]), r.choice([8, 10])
    permit_fee = r.randrange(320, 420, 5)
    reg_days = r.choice([5, 10, 14])
    arn_days = r.choice([3, 7])
    allowance_pct, allowance_months, allowance_min = (
        r.choice([25, 30]),
        r.choice([36, 48, 60]),
        r.randrange(4200, 4800, 50),
    )
    licence_days, licence_fee = r.choice([90, 180]), r.randrange(40, 95, 5)
    insurance_days = r.choice([30, 60, 120])
    arrival = date(2026, r.randint(4, 9), r.randint(1, 28))

    def permits(edition: str, status: str, v: dict[str, int], w: int) -> list[Section]:
        doc = f"permits-{edition}"
        intro = (
            "This edition applies to applications filed from 1 January 2026."
            if status == "current"
            else (
                "This edition applies to applications filed before 1 January 2026. It is superseded by the 2026 edition."
            )
        )
        return [
            Section(
                f"{doc}#1",
                doc,
                "Who can apply",
                f"{intro} A skilled-worker residence permit requires a job offer from an employer registered as a recognised sponsor with the Arvenland Migration Office (AMO). The employer files the application; the employee does not apply directly.",
                edition,
                status,
            ),
            Section(
                f"{doc}#2",
                doc,
                "Salary thresholds",
                f"The gross monthly salary must be at least {v['t30']:,} ARV for applicants aged 30 or older and at least {v['t29']:,} ARV for applicants under 30. Recent graduates, meaning applicants who obtained a degree from an Arvenland university or a top-200 university within the last three years, qualify at the graduate threshold of {v['tg']:,} ARV regardless of age. Holiday allowance does not count toward the threshold.",
                edition,
                status,
            ),
            Section(
                f"{doc}#3",
                doc,
                "Processing",
                f"The AMO decides within {w} weeks of a complete application. The application fee is {permit_fee} ARV and is not refunded if the application is rejected.",
                edition,
                status,
            ),
        ]

    sections = [
        *permits("2026", "current", {"t30": t30, "t29": t29, "tg": tg}, weeks),
        *permits("2025", "superseded", old, old_weeks),
        Section(
            "registration#1",
            "registration",
            "Registering with the municipality",
            f"Everyone who will stay longer than four months must register in person at the town hall within {reg_days} days of arrival. Bring a passport, a rental contract or a landlord's declaration, and a birth certificate.",
            "2026",
            "current",
        ),
        Section(
            "registration#2",
            "registration",
            "Your personal number",
            f"After registration you receive an Arvenland Resident Number (ARN) by post within {arn_days} working days. You need the ARN for your salary, bank account and health insurance.",
            "2026",
            "current",
        ),
        Section(
            "family#1",
            "family",
            "Partners",
            "A partner who moves with a skilled-worker permit holder receives a dependent residence permit. The dependent permit carries the same right to work as the main permit, so the partner may take any job without a separate work permit.",
            "2026",
            "current",
        ),
        Section(
            "family#2",
            "family",
            "Children",
            "Children aged 5 to 16 must attend school. The municipality assigns a school after registration; international schools charge their own fees.",
            "2026",
            "current",
        ),
        Section(
            "tax#1",
            "tax",
            "Expat allowance",
            f"Employees recruited from abroad may receive {allowance_pct}% of their salary tax-free as an expat allowance for at most {allowance_months} months, provided their taxable salary exceeds {allowance_min:,} ARV per month. The employer applies within four months of the start date.",
            "2026",
            "current",
        ),
        Section(
            "driving#1",
            "driving",
            "Driving licences from the EU",
            "Licences issued in an EU or EEA country remain valid in Arvenland until they expire. No exchange is needed.",
            "2026",
            "current",
        ),
        Section(
            "driving#2",
            "driving",
            "Driving licences from other countries",
            f"Holders of a licence issued outside the EU or EEA may drive for {licence_days} days after registration. To keep driving they must exchange it at the road authority; the exchange fee is {licence_fee} ARV.",
            "2026",
            "current",
        ),
        Section(
            "health#1",
            "health",
            "Health insurance",
            f"Basic health insurance is mandatory. Take out a policy within {insurance_days} days of registering with the municipality; cover is backdated to the registration date. Late insurance carries a fine.",
            "2026",
            "current",
        ),
        Section(
            "housing#1",
            "housing",
            "Renting a home",
            "Deposits are capped at two months of base rent. Agencies may not charge tenants a commission for rentals.",
            "2026",
            "current",
        ),
    ]

    pool = [
        Question(
            "salary-30",
            f"What gross monthly salary does a {r.randint(31, 45)}-year-old need for a skilled-worker permit?",
            str(t30),
            ("permits-2026#2",),
            str(old["t30"]),
        ),
        Question(
            "salary-graduate",
            f"I am {r.randint(25, 28)} and finished my master's at an Arvenland university last year. What salary threshold applies to me?",
            str(tg),
            ("permits-2026#2",),
            str(old["tg"]),
        ),
        Question(
            "processing",
            "How many weeks does the migration office take to decide on a skilled-worker permit?",
            str(weeks),
            ("permits-2026#3",),
            str(old_weeks),
        ),
        Question(
            "register-by",
            f"I arrive on {arrival.isoformat()}. What is the last day I can register with the municipality?",
            (arrival + timedelta(days=reg_days)).isoformat(),
            ("registration#1",),
        ),
        Question(
            "partner-work",
            "Can my partner work in Arvenland without getting a separate work permit?",
            "yes",
            ("family#1",),
        ),
        Question(
            "licence-fee-pl", "Ile kosztuje wymiana prawa jazdy wydanego poza UE?", str(licence_fee), ("driving#2",)
        ),
        Question(
            "allowance-months",
            "For how many months can I receive the expat allowance?",
            str(allowance_months),
            ("tax#1",),
        ),
        Question(
            "insurance-days",
            "How many days after registering do I have to take out health insurance?",
            str(insurance_days),
            ("health#1",),
        ),
        Question("corporate-tax", "What is the corporate income tax rate in Arvenland?", "not covered", ()),
    ]
    questions = [
        q for q in pool if q.id in ("salary-30", "salary-graduate", "register-by", "licence-fee-pl", "corporate-tax")
    ]
    questions += r.sample([q for q in pool if q not in questions], 2)
    return sections, questions


def normalise(value: str) -> str:
    """Digits for amounts and counts, ISO for dates, yes/no, or 'not covered'."""
    v = value.strip().lower()
    if v in ("yes", "no", "not covered"):
        return v
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return v
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or v
