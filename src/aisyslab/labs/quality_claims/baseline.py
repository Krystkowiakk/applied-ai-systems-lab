"""Deterministic solver: the same pipeline in code, with keyword rules where a model would read.

Patterns find the part, batch and quantity; a keyword list stands in for moderation; the policy
is applied as written; the reply is a template. The defect category is where rules run out:
"our pins wobble in the holes" or "it stops turning under load" contain none of the keywords, so
those claims are recorded under the wrong category.
"""

from __future__ import annotations

import re

from ...core.lab import RunContext
from ...guardrails.moderation import KeywordModeration
from .env import QualityClaimsEnv

KEYWORDS = {
    "dimension": ("oversized", "bore", "mm", "fixture"),
    "surface": ("scratch", "pitted", "burr"),
    "coating": ("coating", "flake", "rust", "finish"),
    "seizing": ("seize", "locks up", "jam"),
    "leak": ("leak", "weep", "pressure"),
}
MODERATION = KeywordModeration(
    {"harassment": ("idiot", "useless", "disgrace"), "threat": ("make you pay", "make sure nobody")}
)


def solve(env: QualityClaimsEnv, ctx: RunContext) -> None:
    for item in ctx.call(env.list_claims):
        claim_id = item["claim_id"]
        text = env.raw_email(claim_id)
        low = text.lower()
        part = re.search(r"P-\d{4}", text)
        batch = re.search(r"B-\d{4}-\d{4}", text)
        quantity = re.search(r"(\d+) pcs", text)
        defect = next((k for k, words in KEYWORDS.items() if any(w in low for w in words)), "other")
        fields = {
            "part_number": part.group(0) if part else None,
            "batch": batch.group(0) if batch else None,
            "quantity": int(quantity.group(1)) if quantity else None,
            "defect_category": defect,
        }
        record = ctx.call(env.lookup_batch, fields["batch"])
        if MODERATION.check(text).flagged or re.search(r"lawyer|court|legal action", low):
            decision = "escalate"
        elif record["days_before_claim"] > 365:
            decision = "reject"
        elif defect in record["nonconformities"]:
            decision = "accept"
        else:
            decision = "investigate"
        number = ctx.call(env.open_claim, claim_id, decision, fields)["claim_number"]
        if decision == "escalate":
            ctx.call(env.escalate, claim_id, "abusive or threatening message")
            continue
        sender = re.search(r"^From: (\S+)", text, re.M)
        first = sender.group(1) if sender else "customer"
        outcome = {
            "accept": "we confirm the defect; replacement parts are being prepared",
            "investigate": "we will analyse the parts and come back to you",
            "reject": "the batch is outside the 12-month warranty period",
        }[decision]
        ctx.call(
            env.send_reply, claim_id, f"Dear {first},\n\nthank you for your claim {number}: {outcome}.\n\nQuality team"
        )
    ctx.submit(env.finish)
