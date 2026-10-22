"""Model providers behind one ``ChatModel`` interface."""

from __future__ import annotations

import os

from ..core.model import ChatModel, ModelError

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_LOCAL_URL = "http://localhost:1234/v1"  # LM Studio; Ollama serves http://localhost:11434/v1

PROVIDERS = ("openai", "anthropic", "local")


def make_model(
    provider: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    effort: str | None = None,
) -> ChatModel:
    """Build a model client. Keys come from the environment, never from arguments or files."""
    if provider == "openai":
        from .openai_chat import OpenAIChatModel

        return OpenAIChatModel(
            model or os.environ.get("AISYSLAB_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            base_url=base_url,
            reasoning_effort=effort,
        )
    if provider == "anthropic":
        from .anthropic_messages import DEFAULT_MODEL, AnthropicModel

        return AnthropicModel(
            model or os.environ.get("AISYSLAB_ANTHROPIC_MODEL", DEFAULT_MODEL), effort=effort or "medium"
        )
    if provider == "local":
        from .openai_chat import OpenAIChatModel

        url = base_url or os.environ.get("AISYSLAB_LOCAL_URL", DEFAULT_LOCAL_URL)
        name = model or os.environ.get("AISYSLAB_LOCAL_MODEL") or _first_local_model(url)
        # Local servers expect the classic parameter name.
        return OpenAIChatModel(
            name, base_url=url, api_key="local", token_param="max_tokens", label=f"local:{name}", stream_usage=False
        )
    raise ModelError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")


def _first_local_model(url: str) -> str:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise ModelError("Install the OpenAI SDK to talk to a local OpenAI-compatible server.") from exc
    try:
        models = [m.id for m in openai.OpenAI(base_url=url, api_key="local").models.list()]
    except openai.APIError as exc:
        raise ModelError(
            f"No local model server at {url} ({exc}). Start LM Studio or Ollama, or pass --model."
        ) from exc
    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        raise ModelError(f"The server at {url} lists no chat model. Load one or pass --model.")
    return chat_models[0]


__all__ = ["DEFAULT_LOCAL_URL", "DEFAULT_OPENAI_MODEL", "PROVIDERS", "make_model"]
