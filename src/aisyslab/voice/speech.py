"""Speech in and out behind two small interfaces, with OpenAI adapters and offline stand-ins.

``Transcriber`` turns audio into text and takes the plant's vocabulary: as a prompt for models
that read one (``gpt-4o-transcribe``), as keywords for models that take a list (``gpt-transcribe``).
``Synthesizer`` turns text into audio. Offline, ``RecordedTranscriber`` replays transcripts recorded
from a real model, so everything after speech-to-text runs and is tested without a key, and
``SilentSynthesizer`` writes short silent WAV files for the cache, the journal and timing tests.
"""

from __future__ import annotations

import io
import json
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.model import ModelError


@dataclass(frozen=True)
class Audio:
    data: bytes
    format: str = "mp3"
    id: str = ""  # which recording this is, for replay and traces


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Audio, *, vocabulary: Sequence[str] = (), language: str = "en") -> str: ...


class Synthesizer(Protocol):
    name: str

    def synthesize(self, text: str, *, voice: str, instructions: str = "") -> Audio: ...


def vocabulary_prompt(terms: Sequence[str]) -> str:
    return "A maintenance technician's voice note from the shop floor. Plant terms: " + ", ".join(terms) + "."


def _openai() -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ModelError("Install the OpenAI SDK: pip install 'applied-ai-systems-lab[openai]'") from exc
    return openai


class OpenAITranscriber:
    def __init__(self, model: str = "gpt-4o-transcribe") -> None:
        self._openai = _openai()
        self._client = self._openai.OpenAI(max_retries=3)
        self.model = model
        self.name = f"openai:{model}"

    def transcribe(self, audio: Audio, *, vocabulary: Sequence[str] = (), language: str = "en") -> str:
        options: dict[str, Any] = {}
        if vocabulary:
            if self.model == "gpt-transcribe":
                options["keywords"] = list(vocabulary)
            else:
                options["prompt"] = vocabulary_prompt(vocabulary)
        try:
            result = self._client.audio.transcriptions.create(
                model=self.model,
                file=(f"{audio.id or 'note'}.{audio.format}", audio.data),
                language=language,
                **options,
            )
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return str(result.text).strip()


class OpenAISynthesizer:
    def __init__(self, model: str = "gpt-4o-mini-tts") -> None:
        self._openai = _openai()
        self._client = self._openai.OpenAI(max_retries=3)
        self.model = model
        self.name = f"openai:{model}"

    def synthesize(self, text: str, *, voice: str = "alloy", instructions: str = "") -> Audio:
        options = {"instructions": instructions} if instructions else {}
        try:
            response = self._client.audio.speech.create(
                model=self.model, voice=voice, input=text, response_format="mp3", **options
            )
        except self._openai.APIError as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return Audio(response.read(), "mp3")


class RecordedTranscriber:
    """Replays transcripts recorded from a real model: one with the vocabulary, one without."""

    def __init__(self, recorded: Mapping[str, Any] | Path) -> None:
        data = json.loads(recorded.read_text(encoding="utf-8")) if isinstance(recorded, Path) else recorded
        self.name = f"recorded:{data.get('model', 'unknown')}"
        self.transcripts: Mapping[str, Mapping[str, str]] = data["transcripts"]

    def transcribe(self, audio: Audio, *, vocabulary: Sequence[str] = (), language: str = "en") -> str:
        entry = self.transcripts.get(audio.id)
        if entry is None:
            raise ModelError(f"No recorded transcript for '{audio.id}'.")
        return entry["vocabulary" if vocabulary else "plain"]


class SilentSynthesizer:
    """Offline: a silent WAV whose length follows the number of words (a third of a second each)."""

    name = "silent"

    def synthesize(self, text: str, *, voice: str = "alloy", instructions: str = "") -> Audio:
        rate = 8000
        frames = int(rate * 0.33 * max(1, len(text.split())))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(b"\x00\x00" * frames)
        return Audio(buffer.getvalue(), "wav")
