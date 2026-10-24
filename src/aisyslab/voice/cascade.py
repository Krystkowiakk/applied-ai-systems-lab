"""A voice agent as a cascade: speech-to-text, a model, text-to-speech, each stage on a clock.

The question is transcribed with the plant's vocabulary. The model's answer streams in, and every
sentence goes to speech as soon as it is complete, so the listener hears the first sentence while
the model is still writing the third. The first sentence is cut once more, at its first clause of
four words or more: speech of a short clause starts sooner than speech of a long sentence. Each stage has a latency budget. When the model has not
produced a sentence within its budget, the speaker says a short filler instead of leaving silence;
stages over budget are reported, so a slow stage shows up in the numbers rather than as a feeling
that "voice is slow". Every sentence goes through the speaker, so it is cached and spoken at most
once. The finished answer is journaled: a turn retried after a crash speaks that answer again, from
the first sentence that cannot have been heard, instead of asking the model for a new one.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..core.messages import Message
from ..core.model import ChatModel, StreamingChatModel
from .delivery import Speaker
from .speech import Audio, Transcriber

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CLAUSE_END = re.compile(r"(?<=[,:;])\s+")


class Chunker:
    """Cuts streamed text into utterances: sentences, the first one cut at its first long clause.

    The cut depends only on the text, not on how it arrived, so a retried turn cuts its journaled
    answer into exactly the utterances it spoke the first time."""

    def __init__(self) -> None:
        self.buffer = ""
        self.started = False

    def feed(self, piece: str) -> list[str]:
        self.buffer += piece
        *complete, self.buffer = SENTENCE_END.split(self.buffer)
        ready = [s.strip() for s in complete if s.strip()]
        if not self.started:
            first = ready[0] if ready else self.buffer
            for boundary in CLAUSE_END.finditer(first):
                if len(first[: boundary.start()].split()) >= 4:
                    head, tail = first[: boundary.start()].strip(), first[boundary.end() :]
                    if ready:
                        ready[0] = tail.strip()
                    else:
                        self.buffer = tail
                    ready.insert(0, head)
                    break
        self.started = self.started or bool(ready)
        return [s for s in ready if s]

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []


def utterances(text: str) -> list[str]:
    chunker = Chunker()
    return chunker.feed(text) + chunker.flush()


@dataclass(frozen=True)
class LatencyBudget:
    transcribe_s: float = 1.5
    first_sentence_s: float = 2.0  # from the end of transcription to the model's first full sentence
    first_audio_s: float = 4.0  # from the end of the question to the first sound


@dataclass
class TurnReport:
    question: str
    answer: str
    seconds: dict[str, float] = field(default_factory=dict)
    over_budget: list[str] = field(default_factory=list)
    spoken: list[tuple[str, str]] = field(default_factory=list)  # (sentence, what happened)
    filler: bool = False


class VoiceCascade:
    def __init__(
        self,
        transcriber: Transcriber,
        model: ChatModel,
        speaker: Speaker,
        *,
        system: str,
        vocabulary: Sequence[str] = (),
        budget: LatencyBudget | None = None,
        filler: str = "One moment, I'm checking.",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.transcriber = transcriber
        self.model = model
        self.speaker = speaker
        self.system = system
        self.vocabulary = list(vocabulary)
        self.budget = budget or LatencyBudget()
        self.filler = filler
        self.clock = clock

    def warm(self) -> None:
        """Synthesize the filler before the first turn: it is said when the model is slow, not after."""
        self.speaker.prepare(self.filler)

    def turn(self, audio: Audio, *, turn_id: str, context: str = "") -> TurnReport:
        start = self.clock()
        stored = self.speaker.journal.answer(turn_id)
        if stored is not None:  # a retry: resume the answer already given, never a new one
            report = TurnReport("(resumed)", stored)
            for sentence in utterances(stored):
                report.spoken.append((sentence, self.speaker.say(turn_id, sentence)))
            report.seconds["total"] = self.clock() - start
            return report
        question = self.transcriber.transcribe(audio, vocabulary=self.vocabulary)
        report = TurnReport(question, "")
        report.seconds["transcribe"] = self.clock() - start
        sentences: queue.Queue[str | None] = queue.Queue()
        ready = threading.Event()
        failure: list[BaseException] = []
        prompt = f"{context}\n\nQuestion: {question}" if context else question
        thinking = self.clock()

        def produce() -> None:
            chunker = Chunker()
            whole: list[str] = []

            def on_text(piece: str) -> None:
                whole.append(piece)
                for sentence in chunker.feed(piece):
                    if "first_sentence" not in report.seconds:  # when the model wrote it, not when it was heard
                        report.seconds["first_sentence"] = self.clock() - thinking
                    sentences.put(sentence)
                    ready.set()

            try:
                messages = [Message(role="user", content=prompt)]
                if isinstance(self.model, StreamingChatModel):
                    self.model.complete_stream(system=self.system, messages=messages, tools=[], on_text=on_text)
                else:
                    on_text(self.model.complete(system=self.system, messages=messages, tools=[]).text)
                for sentence in chunker.flush():  # only a finished answer ends with an unfinished buffer
                    sentences.put(sentence)
                self.speaker.journal.save_answer(turn_id, "".join(whole).strip())
            except Exception as exc:  # reported after the turn; half a sentence is never spoken
                failure.append(exc)
            finally:
                sentences.put(None)
                ready.set()

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()

        def heard() -> None:  # the moment sound starts, after synthesis and before playback
            report.seconds.setdefault("first_audio", self.clock() - start)

        if not ready.wait(self.budget.first_sentence_s):  # silence is worse than a filler
            report.filler = True
            report.spoken.append((self.filler, self.speaker.say(turn_id, self.filler, before_play=heard)))
        answer: list[str] = []
        try:
            while (next_up := sentences.get()) is not None:
                outcome = self.speaker.say(turn_id, next_up, before_play=heard)
                report.spoken.append((next_up, outcome))
                answer.append(next_up)
        finally:
            worker.join()  # even when playback fails: the answer is journaled once the model finishes
        report.answer = " ".join(answer)
        report.seconds["total"] = self.clock() - start
        limits = {
            "transcribe": self.budget.transcribe_s,
            "first_sentence": self.budget.first_sentence_s,
            "first_audio": self.budget.first_audio_s,
        }
        report.over_budget = [stage for stage, limit in limits.items() if report.seconds.get(stage, 0.0) > limit]
        if failure:
            raise RuntimeError(f"The model failed after {len(answer)} sentences: {failure[0]}") from failure[0]
        return report
