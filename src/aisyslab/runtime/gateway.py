"""One way to call models: routes with fallbacks, a circuit breaker, tenant budgets, moderation,
a response cache and a log of every call.

Every model call the service makes goes through here. Switching a provider, adding a fallback
or blocking a request is one change in one place, and every call is accounted to a tenant.
The provider adapters below this layer only translate formats; the gateway decides which model
answers.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.messages import Message, ModelTurn, ToolCall, ToolSpec, Usage
from ..core.model import ChatModel, ModelError, StreamingChatModel, TextSink
from ..core.resilience import RetriesExhausted, TransientError
from .store import Store, digest

EventSink = Callable[[str, dict[str, Any]], object]


class ModerationBlocked(ModelError):
    """The input was flagged before it reached any provider."""


class BudgetExhausted(ModelError):
    """The tenant has used its token budget."""


class ModelUnavailable(ModelError):
    """Every model on the route failed or is cooling down. Worth retrying later."""


class Moderator(Protocol):
    def flagged(self, text: str) -> list[str]: ...


class PhraseModerator:
    """Offline stand-in for a moderation API: flags text that contains a listed phrase."""

    def __init__(self, phrases: Mapping[str, Sequence[str]]) -> None:
        self.phrases = {category: [p.lower() for p in found] for category, found in phrases.items()}

    def flagged(self, text: str) -> list[str]:
        lowered = text.lower()
        return [category for category, found in self.phrases.items() if any(p in lowered for p in found)]


@dataclass(frozen=True)
class Route:
    """Models to try in order. ``cache`` serves repeated identical requests from the cache,
    which suits classification, not conversations."""

    models: tuple[ChatModel, ...]
    cache: bool = False


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0


@dataclass
class ModelGateway:
    routes: Mapping[str, Route | Sequence[ChatModel]]
    store: Store
    moderator: Moderator | None = None
    failure_threshold: int = 3
    cooldown: float = 60.0
    _breakers: dict[str, _Breaker] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.routes = {name: r if isinstance(r, Route) else Route(tuple(r)) for name, r in self.routes.items()}
        if "default" not in self.routes:
            raise ValueError("The gateway needs a 'default' route.")

    def route(self, name: str) -> Route:
        route = self.routes.get(name) or self.routes["default"]
        assert isinstance(route, Route)
        return route

    def model_names(self, route: str) -> list[str]:
        return [m.name for m in self.route(route).models]

    def bind(
        self, *, tenant_id: str, run_id: str | None = None, route: str = "default", on_event: EventSink | None = None
    ) -> GatewayModel:
        """A ``ChatModel`` that calls through the gateway on behalf of one tenant and run."""
        return GatewayModel(self, tenant_id, run_id, route, on_event)

    # A circuit breaker per model: after ``failure_threshold`` failures in a row the model is
    # skipped for ``cooldown`` seconds; the first call after that is a trial, and one more
    # failure opens the breaker again.
    def available(self, model: str) -> bool:
        with self._lock:
            breaker = self._breakers.get(model)
            return breaker is None or breaker.open_until <= self.store.clock()

    def _failed(self, model: str) -> bool:
        with self._lock:
            breaker = self._breakers.setdefault(model, _Breaker())
            breaker.failures += 1
            if breaker.failures < self.failure_threshold:
                return False
            breaker.open_until = self.store.clock() + self.cooldown
            breaker.failures = self.failure_threshold - 1
            return True

    def _succeeded(self, model: str) -> None:
        with self._lock:
            self._breakers.pop(model, None)


class GatewayModel:
    def __init__(
        self, gateway: ModelGateway, tenant_id: str, run_id: str | None, route: str, on_event: EventSink | None
    ) -> None:
        self._gateway = gateway
        self._tenant = tenant_id
        self._run = run_id
        self._route = route
        self._on_event = on_event

    @property
    def name(self) -> str:
        return f"gateway:{self._route}"

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec], max_output_tokens: int = 4096
    ) -> ModelTurn:
        return self._call(system, messages, tools, max_output_tokens, None)

    def complete_stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        on_text: TextSink,
        max_output_tokens: int = 4096,
    ) -> ModelTurn:
        return self._call(system, messages, tools, max_output_tokens, on_text)

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(kind, data)

    def _log(self, model: str, ok: bool, **fields: Any) -> None:
        self._gateway.store.log_model_call(
            tenant_id=self._tenant, run_id=self._run, route=self._route, model=model, ok=ok, **fields
        )

    def _call(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
        on_text: TextSink | None,
    ) -> ModelTurn:
        gateway = self._gateway
        if gateway.store.budget_left(self._tenant) <= 0:
            self._event("budget_exhausted", {})
            raise BudgetExhausted("The tenant's token budget is used up.")
        last = messages[-1] if messages else None
        if gateway.moderator is not None and last is not None and last.role == "user":
            categories = gateway.moderator.flagged(last.content)
            if categories:
                self._event("moderation_blocked", {"categories": categories})
                raise ModerationBlocked(
                    f"Blocked by moderation ({', '.join(categories)}); nothing was sent to a model."
                )

        route = gateway.route(self._route)
        key = None
        if route.cache and on_text is None:
            key = _cache_key(self._tenant, self._route, system, messages, tools)
            hit = gateway.store.cached_response(key)
            if hit is not None:
                self._log(hit["model"], True, cached=True)
                self._event("cache_hit", {"model": hit["model"]})
                return ModelTurn(
                    text=hit["text"],
                    tool_calls=[ToolCall(**c) for c in hit["tool_calls"]],
                    usage=Usage(),
                    stop_reason=hit["stop_reason"],
                    model=hit["model"],
                )

        errors: list[str] = []
        for model in route.models:
            if not gateway.available(model.name):
                errors.append(f"{model.name}: cooling down")
                continue
            history = [_for_model(m, model.name) for m in messages]
            started = time.perf_counter()
            streamed = False
            try:
                if on_text is not None and isinstance(model, StreamingChatModel):

                    def relay(text: str) -> None:
                        nonlocal streamed
                        streamed = True
                        on_text(text)

                    turn = model.complete_stream(
                        system=system, messages=history, tools=tools, on_text=relay, max_output_tokens=max_output_tokens
                    )
                else:
                    turn = model.complete(
                        system=system, messages=history, tools=tools, max_output_tokens=max_output_tokens
                    )
                    if on_text is not None and turn.text:
                        on_text(turn.text)
            except (ModelError, TransientError, RetriesExhausted) as exc:
                opened = gateway._failed(model.name)
                self._log(model.name, False, latency_ms=_ms(started), error=str(exc)[:500])
                if streamed:  # text from a failed attempt must not run into the next model's text
                    reset = getattr(on_text, "reset", None)
                    if reset is not None:
                        reset()
                self._event(
                    "model_failed",
                    {
                        "model": model.name,
                        "error": str(exc)[:300],
                        "breaker_open": opened,
                        "partial_text_dropped": streamed,
                    },
                )
                errors.append(f"{model.name}: {exc}")
                continue
            gateway._succeeded(model.name)
            self._log(
                model.name,
                True,
                input_tokens=turn.usage.input_tokens,
                output_tokens=turn.usage.output_tokens,
                latency_ms=_ms(started),
            )
            if key is not None:
                gateway.store.cache_response(
                    key,
                    {
                        "text": turn.text,
                        "tool_calls": [dataclasses.asdict(c) for c in turn.tool_calls],
                        "stop_reason": turn.stop_reason,
                        "model": model.name,
                    },
                )
            if turn.provider_state is not None:
                # Reasoning blocks are bound to the model that wrote them; remember which one.
                turn.provider_state = {"by": model.name, "state": turn.provider_state}
            return turn
        raise ModelUnavailable(f"No model on route '{self._route}' answered: " + "; ".join(errors))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _for_model(message: Message, model: str) -> Message:
    """Replay provider state (reasoning blocks) only to the model that produced it."""
    state = message.provider_state
    if message.role != "assistant" or not isinstance(state, dict) or "by" not in state:
        return message
    return dataclasses.replace(message, provider_state=state["state"] if state["by"] == model else None)


def _cache_key(tenant: str, route: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> str:
    payload = {
        "tenant": tenant,  # never shared between tenants
        "route": route,
        "system": system,
        "tools": [t.name for t in tools],
        "messages": [
            [
                m.role,
                m.content,
                m.tool_call_id,
                [[c.name, c.arguments] for c in m.tool_calls],
                [digest(i.data.hex()) for i in m.images],
            ]
            for m in messages
        ],
    }
    return digest(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
