"""Voice: grounding in the plant's vocabulary, speech said at most once, and the cascade's clock."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from aisyslab.core.messages import Message, ModelTurn, ToolSpec
from aisyslab.core.model import ScriptedModel, text_turn
from aisyslab.voice import (
    Audio,
    DeliveryJournal,
    LatencyBudget,
    RecordedTranscriber,
    SilentSynthesizer,
    Speaker,
    SpeechCache,
    VoiceCascade,
    spoken_numbers,
)
from aisyslab.voice.cascade import Chunker, utterances
from aisyslab.voice.cli import RECORDED
from aisyslab.voice.notes import NOTES, VOCABULARY, evaluate

ground = VOCABULARY.ground


def ids(text: str) -> list[str]:
    return [f["id"] for f in ground(text).found if f["kind"] != "line"]


def test_spoken_numbers_and_ids_written_any_way() -> None:
    assert spoken_numbers("P ten forty, line two, forty-two") == "P 10 40, line 2, 42"
    assert spoken_numbers("one oh four oh, oh well") == "1 0 4 0, oh well"
    for heard in ("MC14", "MC-14", "M.C. 14", "M C fourteen", "emcee fourteen", "MC 1 4"):
        assert ids(heard) == ["MC-14"], heard
    for heard in ("P1040", "P-1040", "P 10 40", "P ten forty", "P one oh four oh", "P 1,040"):
        assert ids(heard) == ["P-1040"], heard


def test_what_the_registry_cannot_settle_becomes_a_question() -> None:
    teen_ty = ground("M.C. 40, the welding cell")  # "fourteen" heard as "forty"
    assert {(f["id"], f["how"]) for f in teen_ty.found} >= {("MC-14", "teen/ty"), ("MC-17", "name")}
    wrong_number = ground("We need the relief valve, P 15 50, right now.")
    assert wrong_number.ids("part") == []
    assert wrong_number.unsure == [
        {
            "heard": "P 15 50",
            "candidates": ["P-1550", "P-1515"],
            "why": "the note says relief valve; P-1550 is a pressure sensor",
        }
    ]
    assert ground("MC 99 is broken").unsure == [{"heard": "MC 99", "candidates": [], "why": "not in the registry"}]
    assert ids("The transfer press is knocking") == ["MC-14"] and ids("Press two on line one") == ["MC-12"]
    press = ground("The press on line one is leaking")
    assert press.ids("machine") == [] and press.unsure[0]["candidates"] == ["MC-11", "MC-12"]


def test_the_recorded_run_scores_as_documented() -> None:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))["transcripts"]
    note_of = {texts["vocabulary"]: note_id for note_id, texts in recorded.items()}

    def model(system: str, user: str) -> str:  # a reader that gets every note right, to isolate the code paths
        note = next(n for n in NOTES if n.id == note_of[json.loads(user)["transcript"]])
        return json.dumps({"machine": note.machine, "part": note.part, "urgency": note.urgency, "question": ""})

    result = evaluate(RecordedTranscriber(RECORDED), lambda note: Audio(b"", "mp3", note.id), model)
    right = {name: s["all_right"] for name, s in result["summary"].items()}
    assert right == {"plain": 5, "grounding": 9, "vocabulary": 9, "vocabulary+model": 10}
    plain = result["summary"]["plain"]
    assert (plain["machine"], plain["part"]) == (5, 10)  # IDs written right; machines named by name are lost


class CountingSynthesizer(SilentSynthesizer):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text: str, *, voice: str = "alloy", instructions: str = "") -> Audio:
        self.calls.append((text, voice, instructions))
        return super().synthesize(text, voice=voice, instructions=instructions)


def test_speech_is_synthesized_once_per_text_voice_and_style(tmp_path: Path) -> None:
    synth = CountingSynthesizer()
    cache = SpeechCache(tmp_path, synth)
    first = cache.get("Line two is down.", voice="sage")
    again = cache.get("Line two is down.", voice="sage")
    cache.get("Line two is down.", voice="ash")
    cache.get("Line two is down.", voice="sage", instructions="calm")
    assert first.data == again.data and len(synth.calls) == 3 and (cache.hits, cache.misses) == (1, 3)
    assert SpeechCache(tmp_path, synth).get("Line two is down.", voice="sage").data == first.data  # survives restarts
    assert len(synth.calls) == 3


def test_a_retried_briefing_never_repeats_what_may_have_been_heard(tmp_path: Path) -> None:
    sentences = ["Line two is down.", "The trim press needs a relief valve.", "Press five runs as planned."]
    journal = DeliveryJournal(tmp_path / "journal.sqlite")
    heard: list[int] = []

    def crashing(audio: Audio) -> None:
        if len(heard) == 1:
            raise ConnectionError("speaker lost")  # mid-sentence: the second may or may not have been heard
        heard.append(len(audio.data))

    speaker = Speaker(SpeechCache(tmp_path / "cache", SilentSynthesizer()), journal, crashing)
    assert speaker.say("briefing-0601", sentences[0]) == "played"
    with pytest.raises(ConnectionError):
        speaker.say("briefing-0601", sentences[1])
    retry = Speaker(
        SpeechCache(tmp_path / "cache", SilentSynthesizer()), DeliveryJournal(tmp_path / "journal.sqlite"), heard.append
    )
    outcomes = [retry.say("briefing-0601", s) for s in sentences]
    assert outcomes == ["skipped: delivered before", "skipped: may have been delivered", "played"]
    assert journal.uncertain() == [sentences[1]]
    assert retry.say("briefing-0602", sentences[0]) == "played"  # another briefing says it again


def test_chunking_does_not_depend_on_how_the_text_streamed() -> None:
    text = "Line 2 is down: the trim press needs the relief valve now. It stays down until then. Call me."
    assert utterances(text) == [
        "Line 2 is down:",
        "the trim press needs the relief valve now.",
        "It stays down until then.",
        "Call me.",
    ]
    for seed in range(50):
        r, chunker, out, i = random.Random(seed), Chunker(), [], 0
        while i < len(text):
            n = r.randint(1, 9)
            out += chunker.feed(text[i : i + n])
            i += n
        assert out + chunker.flush() == utterances(text)


class Heard:
    name = "fixed"

    def transcribe(self, audio: Audio, *, vocabulary: Sequence[str] = (), language: str = "en") -> str:
        return "What do I need to know about line two?"


def cascade(tmp_path: Path, answer: str, *, delay: float = 0.0) -> tuple[VoiceCascade, list[int]]:
    calls: list[int] = []

    def policy(_system: str, _messages: Sequence[Message], _tools: Sequence[ToolSpec]) -> ModelTurn:
        calls.append(1)
        time.sleep(delay)
        return text_turn(answer)

    cache = SpeechCache(tmp_path / "cache", SilentSynthesizer())
    speaker = Speaker(cache, DeliveryJournal(tmp_path / "journal.sqlite"), lambda _audio: None)
    budget = LatencyBudget(transcribe_s=1.0, first_sentence_s=0.05, first_audio_s=1.0)
    return VoiceCascade(Heard(), ScriptedModel(policy), speaker, system="Brief the shift lead.", budget=budget), calls


def test_a_slow_model_gets_a_filler_and_the_answer_streams_by_sentence(tmp_path: Path) -> None:
    answer = "Line two is down: the trim press waits for a relief valve. Press five runs as planned."
    voice, calls = cascade(tmp_path, answer, delay=0.2)
    voice.warm()
    report = voice.turn(Audio(b"", "wav", "q"), turn_id="t1")
    assert report.filler and report.spoken[0] == ("One moment, I'm checking.", "played")
    assert [s for s, _ in report.spoken[1:]] == utterances(answer) and calls == [1]
    assert "first_sentence" in report.over_budget
    assert report.seconds["first_audio"] < report.seconds["first_sentence"]  # the filler filled the silence
    assert report.answer == " ".join(utterances(answer))


def test_a_retried_turn_resumes_the_journaled_answer_without_asking_again(tmp_path: Path) -> None:
    answer = "Line two is down. The trim press waits for a relief valve. Press five runs as planned."
    voice, calls = cascade(tmp_path, answer)
    first = voice.turn(Audio(b"", "wav", "q"), turn_id="t1")
    assert [o for _, o in first.spoken] == ["played"] * 3 and calls == [1]
    again, more_calls = cascade(tmp_path, answer)
    resumed = again.turn(Audio(b"", "wav", "q"), turn_id="t1")
    assert resumed.question == "(resumed)" and more_calls == []
    assert [o for _, o in resumed.spoken] == ["skipped: delivered before"] * 3
