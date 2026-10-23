"""Illustrations behind one interface: an image model, or a plain line drawing offline."""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any, Protocol

from PIL import Image, ImageDraw

from ..core.model import ModelError


class ImageGenerator(Protocol):
    name: str

    def generate(self, prompt: str) -> bytes: ...  # PNG


class OpenAIImages:
    def __init__(self, model: str = "gpt-image-1-mini", *, quality: str = "low", size: str = "1024x1024") -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
        self._openai: Any = openai
        self._client = openai.OpenAI(max_retries=3)
        self.model, self.quality, self.size = model, quality, size
        self.name = f"openai:{model}"

    def generate(self, prompt: str) -> bytes:
        options: dict[str, Any] = {"quality": self.quality, "size": self.size, "output_format": "png"}
        try:
            result = self._client.images.generate(model=self.model, prompt=prompt, **options)
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return base64.b64decode(result.data[0].b64_json)


class DiagramImages:
    """Offline: a plain line drawing, the same for the same prompt."""

    name = "diagram"

    def generate(self, prompt: str) -> bytes:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        image = Image.new("RGB", (1024, 1024), (250, 250, 252))
        draw = ImageDraw.Draw(image)
        y = 380 + seed % 200
        draw.rectangle((140, y - 120, 360, y + 120), outline=(20, 33, 61), width=10)  # a valve block
        draw.rectangle((680, y - 80, 900, y + 80), outline=(20, 33, 61), width=10)  # a cylinder
        draw.arc((300, y - 260, 740, y + 140), start=200, end=340, fill=(20, 33, 61), width=18)  # the hose
        out = io.BytesIO()
        image.save(out, "PNG")
        return out.getvalue()
