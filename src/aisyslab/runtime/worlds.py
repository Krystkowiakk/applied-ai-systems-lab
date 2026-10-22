"""The simulated systems each conversation acts on, kept per thread and rebuilt from the journal.

A lab environment stands in for the plant's systems. Real systems keep their own state; here the
state lives in this process. When it is missing (another worker process, a restart, a new
branch, an evicted entry), it is rebuilt: a fresh environment from the session's seed, and every
tool call in the thread's history run again, in order. Each result must equal the one in the
journal. A difference means the tools are not deterministic, and the run fails instead of
continuing in a world that is not the one its history describes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from ..core.lab import RunContext
from ..core.messages import ToolCall
from ..core.model import ChatModel
from ..core.resilience import RetryPolicy
from ..core.trace import Tracer
from .agents import BUILTINS, RuntimeAgent
from .store import Store


class ReplayDiverged(RuntimeError):
    """Running the journaled calls again gave a different result."""


class Worlds:
    def __init__(self, store: Store, capacity: int = 64) -> None:
        self.store = store
        self.capacity = capacity
        self._envs: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        thread_id: str,
        agent: RuntimeAgent,
        seed: int,
        model: ChatModel,
        on_rebuilt: Callable[[int], object] | None = None,
    ) -> Any:
        if agent.world is None:
            return None
        with self._lock:
            if thread_id in self._envs:
                self._envs.move_to_end(thread_id)
                return self._envs[thread_id]
        env, replayed = self._rebuild(thread_id, agent, seed, model)
        if replayed and on_rebuilt is not None:
            on_rebuilt(replayed)
        with self._lock:
            self._envs[thread_id] = env
            while len(self._envs) > self.capacity:  # safe to evict: the journal can rebuild it
                self._envs.popitem(last=False)
        return env

    def forget(self, thread_id: str) -> None:
        with self._lock:
            self._envs.pop(thread_id, None)

    def _rebuild(self, thread_id: str, agent: RuntimeAgent, seed: int, model: ChatModel) -> tuple[Any, int]:
        assert agent.world is not None
        env = agent.world(seed)
        if agent.tools is None:
            return env, 0
        ctx = RunContext(tracer=Tracer(), retry=RetryPolicy(sleep=lambda _seconds: None))
        toolbox = agent.tools(env, model, ctx)
        items = self.store.history(thread_id)
        results = {i["data"]["call_id"]: i["data"] for i in items if i["kind"] == "tool"}
        replayed = 0
        for item in items:
            if item["kind"] != "assistant":
                continue
            for call in item["data"]["tool_calls"]:
                recorded = results.get(call["id"])
                if recorded is None or call["name"] in BUILTINS:
                    continue
                result = toolbox.execute(ToolCall(call["id"], call["name"], call["arguments"], call.get("parse_error")))
                if result.content != recorded["content"]:
                    raise ReplayDiverged(
                        f"Replaying {call['name']} gave a different result than the journal, so the run cannot "
                        "continue in the same world. Tools that call a model or read the clock are not replayable."
                    )
                replayed += 1
        return env, replayed
