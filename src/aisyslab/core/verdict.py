from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """What a verifier says about a submission.

    ``feedback`` is written for the solver: specific enough to act on ("missing evidence for
    db-replica-1"), never so specific that it leaks the answer.
    """

    accepted: bool
    code: str
    feedback: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"accepted": self.accepted, "code": self.code, "feedback": self.feedback}
        if self.details:
            out["details"] = self.details
        return out
