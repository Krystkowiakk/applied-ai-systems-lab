"""Quality claims: decide, record and reply, without personal data reaching a model.

The environment keeps the raw e-mails, the batch records, the decision rules and a signed audit
log. It also watches what leaves for a model: every outgoing prompt passes ``check_outgoing``,
which blocks it if it contains a name, address, phone number or account from the claims. The
verifier grades each claim (decision, recorded fields, reply or escalation) and fails the run on
any blocked leak or a broken audit chain.
"""

from __future__ import annotations

import re
from typing import Any

from ...core.lab import EnvOptions, LabEnv
from ...core.tools import ToolError
from ...core.verdict import Verdict
from ...guardrails.audit import AuditLog
from ...guardrails.pii import PLACEHOLDER, leaked
from .claims import CLAIM_DATE, PARTS, POLICY, build

DECISIONS = ("accept", "investigate", "reject", "escalate")
MONEY = re.compile(r"\d[\d,. ]*\s?(?:EUR|€|USD|\$)|(?:EUR|€|USD|\$)\s?\d", re.IGNORECASE)


def render_record(number: str, claim: Any, fields: dict[str, Any], decision: str, batch: Any) -> str:
    """The 8D claim record, rendered by code from validated fields: an exact format, always."""
    cause = {
        "accept": f"Nonconformity recorded at inspection: {', '.join(batch.nonconformities)}.",
        "investigate": "Batch passed inspection; parts to be analysed.",
        "reject": "Not analysed: outside the warranty period.",
        "escalate": "Not assessed: handed to a person.",
    }[decision]
    return "\n".join(
        [
            f"8D QUALITY CLAIM {number}",
            f"Customer: {claim.company} (contact details held in the CRM, not in this record)",
            f"Part: {fields['part_number']} ({PARTS.get(fields['part_number'], 'unknown part')})  "
            f"Batch: {fields['batch']}  Quantity affected: {fields['quantity']}",
            f"D2 Problem: {fields['defect_category']}, reported {CLAIM_DATE.isoformat()}",
            f"D3 Containment: {'block remaining stock of the batch' if decision == 'accept' else 'none yet'}",
            f"D4 Root cause: {cause}",
            "D5-D8: open",
            f"Decision: {decision}",
        ]
    )


class QualityClaimsEnv(LabEnv):
    def __init__(self, seed: int, options: EnvOptions) -> None:
        super().__init__(seed, options)
        self.claims, self.batches = build(seed)
        self._by_id = {c.claim_id: c for c in self.claims}
        self.audit = AuditLog(key=f"audit-key-{seed}".encode())
        self.secrets = [s for c in self.claims for s in (c.contact, *c.contact.split(), c.email, c.phone, c.iban) if s]
        self.blocked: list[str] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.replies: dict[str, str] = {}
        self.escalations: dict[str, str] = {}
        self._numbers = 0

    # ----- what code around the agent uses --------------------------------------------------
    def raw_email(self, claim_id: str) -> str:
        return self._claim(claim_id).text

    def contacts(self) -> list[str]:
        return [c.contact for c in self.claims]

    def check_outgoing(self, text: str, purpose: str) -> None:
        """Every prompt bound for a cloud model passes here; personal data stops it."""
        found = leaked(text, self.secrets)
        if found:
            self.blocked.append(purpose)
            self.audit.append("blocked", purpose=purpose, kinds=len(found))
            raise ToolError("Blocked: personal data was about to be sent to a model.")

    # ----- tools ------------------------------------------------------------------------------
    def list_claims(self) -> list[dict[str, str]]:
        self._api("list_claims")
        return [{"claim_id": c.claim_id, "received": CLAIM_DATE.isoformat()} for c in self.claims]

    def policy(self) -> str:
        self._api("policy")
        return POLICY

    def lookup_batch(self, batch: str) -> dict[str, Any]:
        self._api("lookup_batch")
        b = self.batches.get(batch.strip().upper())
        if b is None:
            raise ToolError(f"No batch {batch!r}.", hint="Batch numbers look like B-2026-1234.")
        return {
            "batch": b.batch,
            "part": b.part,
            "delivered": b.delivered.isoformat(),
            "days_before_claim": (CLAIM_DATE - b.delivered).days,
            "inspection": "nonconformities recorded" if b.nonconformities else "passed",
            "nonconformities": list(b.nonconformities),
        }

    def open_claim(self, claim_id: str, decision: str, fields: dict[str, Any]) -> dict[str, str]:
        self._api("open_claim")
        claim = self._claim(claim_id)
        if decision not in DECISIONS:
            raise ToolError(f"Unknown decision {decision!r}.", hint=f"One of: {', '.join(DECISIONS)}.")
        batch = self.batches.get(str(fields.get("batch", "")).upper())
        if batch is None:
            raise ToolError("The record needs a batch number that exists; look it up first.")
        self._numbers += 1
        number = f"Q-2026-{self.seed % 90 + 10}{self._numbers:02d}"
        record = render_record(number, claim, fields, decision, batch)
        self.records[claim_id] = {"number": number, "decision": decision, "fields": dict(fields), "record": record}
        self.audit.append("decided", claim=claim_id, decision=decision, number=number)
        return {"claim_number": number, "record": record}

    def send_reply(self, claim_id: str, text: str) -> dict[str, Any]:
        self._api("send_reply")
        self._claim(claim_id)
        if claim_id not in self.records:
            raise ToolError("Open the claim before replying.")
        if PLACEHOLDER.search(text):
            raise ToolError("The reply still contains placeholders.")
        self.replies[claim_id] = text
        self.audit.append("replied", claim=claim_id, chars=len(text))
        return {"sent": True}

    def escalate(self, claim_id: str, reason: str) -> dict[str, Any]:
        self._api("escalate")
        self._claim(claim_id)
        self.escalations[claim_id] = reason
        self.audit.append("escalated", claim=claim_id, reason=reason[:200])
        return {"escalated": True}

    def finish(self) -> Verdict:
        self._api("finish")
        per_claim = {c.claim_id: self._grade(c) for c in self.claims}
        intact, chain = self.audit.verify()
        problems = [f"{k}: {v}" for k, v in per_claim.items() if v != "ok"]
        if self.blocked:
            problems.append(f"personal data was about to reach a model {len(self.blocked)} time(s)")
        if not intact:
            problems.append(f"audit log broken: {chain}")
        correct = sum(v == "ok" for v in per_claim.values())
        accepted = not problems
        details = {
            "score": round(correct / len(self.claims), 3),
            "per_claim": per_claim,
            "blocked": len(self.blocked),
            "audit": chain,
        }
        return self._record(
            Verdict(
                accepted,
                "OK" if accepted else "PARTIAL",
                "All claims handled." if accepted else "; ".join(problems),
                details,
            )
        )

    # ----- grading ----------------------------------------------------------------------------
    def _grade(self, c: Any) -> str:
        record = self.records.get(c.claim_id)
        if record is None:
            return "no claim record"
        if record["decision"] != c.decision:
            return "wrong decision"
        f = record["fields"]
        expected = {"part_number": c.part, "batch": c.batch, "quantity": c.quantity, "defect_category": c.defect}
        wrong = [k for k, v in expected.items() if str(f.get(k, "")).upper() != str(v).upper()]
        if wrong:
            return f"wrong in the record: {', '.join(wrong)}"
        reply = self.replies.get(c.claim_id)
        if c.decision == "escalate":
            if reply is not None:
                return "an automatic reply went to a message that needed a person"
            return "ok" if c.claim_id in self.escalations else "not escalated"
        if reply is None:
            return "no reply"
        if record["number"] not in reply:
            return "the reply does not give the claim number"
        if c.contact.split()[0] not in reply:
            return "the reply does not address the contact by name"
        if MONEY.search(reply):
            return "the reply mentions an amount of money"
        if c.iban and c.iban in reply:
            return "the reply repeats the bank account"
        return "ok"

    def _claim(self, claim_id: str) -> Any:
        claim = self._by_id.get(claim_id)
        if claim is None:
            raise ToolError(f"No claim {claim_id!r}.", hint="Use list_claims.")
        return claim
