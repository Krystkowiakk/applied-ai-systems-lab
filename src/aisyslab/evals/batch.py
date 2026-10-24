"""Many tickets in one request, under a JSON schema the provider enforces.

One request per ticket repeats the system prompt for every ticket; a batch sends it once. The
provider's structured-output mode guarantees the shape of the answer: every item has an id from the
batch, a known code, a priority from 1 to 3 and a reason, so nothing is lost to a stray sentence or a
missing bracket. What a schema cannot guarantee, code checks: that every ticket came back exactly
once. Missing tickets are asked for once more, on their own; repeats are dropped and counted.

The price of a batch is attention: every ticket is classified next to others, and a model can let
one ticket colour the next. The comparison is item by item against one request per ticket.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.messages import Message, Usage
from ..core.model import ModelError
from ..core.tokens import estimate_tokens
from .graders import DETERMINISTIC, Grader, parse
from .runner import EvalReport, ItemResult, fingerprint, prompt_id
from .triage import CODES, RuleFollowerModel, Ticket

BATCH_INSTRUCTIONS = """

You get several tickets at once, each with an id. Classify every ticket on its own, as if it were the
only one, and answer for each id exactly once."""


def batch_schema(ids: Sequence[str]) -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": list(ids)},
            "code": {"type": "string", "enum": list(CODES)},
            "priority": {"type": "integer", "enum": [1, 2, 3]},
            "reason": {"type": "string"},
        },
        "required": ["id", "code", "priority", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
        "additionalProperties": False,
    }


class JsonModel(Protocol):
    name: str

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> tuple[str, Usage]: ...


class OpenAIJsonModel:
    """Chat completions with a strict JSON schema: the provider enforces the shape of every answer."""

    def __init__(self, model: str = "gpt-5-mini", *, effort: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        from ..providers.openai_chat import MIN_COMPLETION_TOKENS

        self._openai: Any = openai
        self._client = openai.OpenAI(max_retries=3)
        self.model, self.effort, self.limit = model, effort, MIN_COMPLETION_TOKENS
        self.name = model

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], schema_name: str) -> tuple[str, Usage]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            "max_completion_tokens": self.limit,
        }
        if self.effort:
            payload["reasoning_effort"] = self.effort
        try:
            response = self._client.chat.completions.create(**payload)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        details = getattr(response.usage, "prompt_tokens_details", None)
        usage = Usage(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            (getattr(details, "cached_tokens", 0) or 0) if details else 0,
        )
        return response.choices[0].message.content or "", usage


class RuleFollowerBatch:
    """Offline: the rule follower answers each ticket in the schema's shape. ``drop`` tickets are
    left out of the first answer, the way a model sometimes skips one in a long batch."""

    def __init__(self, drop: int = 0) -> None:
        self.inner = RuleFollowerModel()
        self.drop = drop
        self.name = "rule-follower"

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], schema_name: str) -> tuple[str, Usage]:
        tickets = json.loads(user)["tickets"]
        answered = tickets[self.drop :] if len(tickets) > 1 else tickets
        items = []
        for ticket in answered:
            turn = self.inner.complete(system=system, messages=[Message(role="user", content=ticket["text"])], tools=[])
            items.append({"id": ticket["id"]} | json.loads(turn.text))
        text = json.dumps({"items": items})
        return text, Usage(estimate_tokens(system + user), estimate_tokens(text))


@dataclass
class BatchRun:
    report: EvalReport
    calls: int
    missing_first: int  # tickets a batch answer left out, before they were asked again
    repeated: int  # items answered twice or with an id from another batch, dropped


def classify_batches(
    model: JsonModel,
    prompt: str,
    tickets: Sequence[Ticket],
    *,
    size: int = 12,
    graders: Sequence[Grader] = DETERMINISTIC,
) -> BatchRun:
    outputs: dict[str, str] = {}
    usage, calls, missing_first, repeated = Usage(), 0, 0, 0
    queue = [list(tickets[i : i + size]) for i in range(0, len(tickets), size)]
    asked_again: set[str] = set()
    while queue:
        batch = queue.pop(0)
        ids = [t.id for t in batch]
        user = json.dumps({"tickets": [{"id": t.id, "text": t.text} for t in batch]}, ensure_ascii=False)
        text, used = model.complete_json(
            system=prompt + BATCH_INSTRUCTIONS, user=user, schema=batch_schema(ids), schema_name="triage_batch"
        )
        calls, usage = calls + 1, usage + used
        seen: set[str] = set()
        for item in (parse(text) or {}).get("items") or []:
            if isinstance(item, dict) and item.get("id") in ids and item["id"] not in seen:
                seen.add(item["id"])
                outputs[item["id"]] = json.dumps({k: item.get(k) for k in ("code", "priority", "reason")})
            else:
                repeated += 1
        missing = [t for t in batch if t.id not in seen]
        if missing and not asked_again & {t.id for t in missing}:
            missing_first += len(missing)
            asked_again |= {t.id for t in missing}
            queue.append(missing)  # once more, on their own; after that a missing ticket stays failed
    items = [
        ItemResult(t.id, t.code, outputs.get(t.id, ""), [g(t, outputs.get(t.id, "")) for g in graders]) for t in tickets
    ]
    report = EvalReport(
        f"{model.name} (batch of {size})", prompt_id(prompt + BATCH_INSTRUCTIONS), fingerprint(tickets), items, usage
    )
    return BatchRun(report, calls, missing_first, repeated)
